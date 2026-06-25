#include <string>
#include <iostream>
#include <mutex>
#include <unordered_set>
#include <ros/ros.h>
#include <pcl_ros/transforms.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PointStamped.h>
#include "markerarray_pub.h"
#include "bgkloctomap.h"

std::mutex map_mutex;

tf::TransformListener *listener;
std::string frame_id("/odom");
la3dm::BGKLOctoMap *map;

la3dm::MarkerArrayPub *m_pub_occ, *m_pub_free, *m_pub_unc, *m_pub_unk, *m_pub_var, *m_pub_alpha_beta, *m_pub_occ_coarse, *m_pub_constraint;
la3dm::TextMarkerArrayPub *m_pub_free_txt, *m_pub_occ_txt, *m_pub_unk_txt;
ros::Publisher viewpoints_pub;

// Tracking for visualization of past processed viewpoints
tf::Vector3 last_position;
tf::Quaternion last_orientation;
std::vector<tf::Transform> past_viewpoints;

// Cached viewpoint message, appended to on each new sonar frame.
// Published on change (from cloudHandler) — the topic is latched so new
// subscribers still receive the full history without periodic republishing.
geometry_msgs::PoseArray viewpoints_msg;

//Universal parameters
std::string map_topic_occ("/occupied_cells_vis_array");
std::string map_topic_occ_coarse("/occupied_cells_coarse_vis_array");
int coarse_depth_steps = 2;   // coarse voxel side = resolution * 2^coarse_depth_steps
std::string map_topic_free("/free_cells_vis_array");
std::string map_topic_free_txt("/free_cells_txt_vis_array");
std::string map_topic_occ_txt("/occupied_cells_txt_vis_array");
std::string map_topic_unk_txt("/unknown_cells_txt_vis_array");
std::string map_topic_unc("/uncertain_cells_vis_array");
std::string map_topic_unk("/unknown_cells_vis_array");
std::string map_topic_var("/variance_vis_array");
std::string map_topic_alpha_beta("/alpha_beta_vis_array");
std::string map_topic_constraint("/constraint_vis_array");
double max_range = 4.0;
double resolution = 0.1;
double ds_resolution = 0.1;
int block_depth = 4;
double sf2 = 0.1;
double ell = 0.2;
double free_resolution = 0.65;
double free_thresh = 0.3;
double occupied_thresh = 0.7;
double min_z = 0;
double max_z = 0;
bool original_size = true;
double max_var_vis = 0.25;
double var_vis_gamma = 1.0;   // power applied to t before color mapping; >1 stretches red→yellow range
double max_constraint_vis = 5.0;
double max_vis_radius = 40.0; // Visualization sphere radius (m); 0 = unlimited
double viz_rate = 1.5;        // Visualization publish rate (Hz); decoupled from sonar callback

//BGKL parameters
float var_thresh = 1.0f;
float prior_A = 1.0f;
float prior_B = 1.0f;
float theta_bw = 0.6f * 3.1415926f / 180.0f;
float phi_bw = 20.0f * 3.1415926f / 180.0f;
bool free_ray_range_weight = false;
float swath_angle = -1.0f;  // Total multibeam swath width (degrees); -1 = azimuth culling disabled
float tau_info = 2.0f;
float tau_var  = 0.01f;
float delta    = 0.05f;

// ---- Pose-level novelty weighting parameters ----
bool  use_pose_level_weighting    = false;
int   pose_history_size        = 20;
float pose_novelty_sigma       = 0.3f;
float pose_w_roll              = 1.0f;
float pose_w_pitch             = 0.6f;
float pose_w_yaw               = 0.05f;
float pose_w_vx_l2             = 0.2f;
float pose_w_vy_l2             = 0.05f;
float pose_w_vz_l2             = 0.5f;

// ---- Ablation flags ----
bool  ablate_directional_weights = false;

// ---- Per-voxel debug tracing ----
bool  debug_voxel_enable = false;
float debug_voxel_x = 0.0f, debug_voxel_y = 0.0f, debug_voxel_z = 0.0f;

void clickedPointHandler(const geometry_msgs::PointStamped::ConstPtr &msg) {
    debug_voxel_x = (float)msg->point.x;
    debug_voxel_y = (float)msg->point.y;
    debug_voxel_z = (float)msg->point.z;
    debug_voxel_enable = true;
    std::lock_guard<std::mutex> lock(map_mutex);
    map->set_debug_voxel(true, debug_voxel_x, debug_voxel_y, debug_voxel_z);
    ROS_INFO("[dbg_voxel] now tracing voxel near (%.3f, %.3f, %.3f)",
             debug_voxel_x, debug_voxel_y, debug_voxel_z);
}

void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &cloud) {

    // ---- Outside mutex: TF and point-cloud transform are fast and read-only ----
    tf::StampedTransform transform;
    try {
        // Wait for the TF at exactly the cloud's stamp. The sonar can run
        // slightly ahead of the TF publisher (common with use_sim_time),
        // so a short wait is needed before the exact-timestamp lookup.
        // Safe to block here: TF lookup is outside map_mutex, so the viz
        // timer and other callbacks are not stalled.
        if (!listener->waitForTransform(frame_id, cloud->header.frame_id,
                                        cloud->header.stamp, ros::Duration(1.0))) {
            ROS_WARN_THROTTLE(1.0, "TF timeout for stamp %.3f, dropping cloud",
                              cloud->header.stamp.toSec());
            return;
        }
        listener->lookupTransform(frame_id, cloud->header.frame_id, cloud->header.stamp, transform);
    } catch (tf::TransformException ex) {
        ROS_WARN_THROTTLE(1.0, "TF lookup failed: %s", ex.what());
        return;
    }

    tf::Vector3 translation = transform.getOrigin();
    tf::Quaternion orientation = transform.getRotation();

    size_t expected = (size_t)cloud->width * cloud->height * cloud->point_step;
    if (cloud->data.size() < expected) {
        ROS_WARN_THROTTLE(1.0, "Malformed PointCloud2: data.size()=%zu < expected=%zu, skipping",
                          cloud->data.size(), expected);
        return;
    }

    la3dm::PCLPointCloud::Ptr pcl_cloud_src(new la3dm::PCLPointCloud());
    pcl::fromROSMsg(*cloud, *pcl_cloud_src);

    la3dm::PCLPointCloud::Ptr pcl_cloud_world(new la3dm::PCLPointCloud());
    pcl_ros::transformPointCloud(*pcl_cloud_src, *pcl_cloud_world, transform);

    if (pcl_cloud_world->size() <= 5)
        return;

    la3dm::point3f origin;
    origin.x() = (float)translation.x();
    origin.y() = (float)translation.y();
    origin.z() = (float)translation.z();

    tf::Matrix3x3 mat(orientation);
    tf::Vector3 up = mat.getColumn(2);
    la3dm::point3f sensor_up(up.x(), up.y(), up.z());

    // ---- Under mutex: map write + shared-state update ----
    {
        std::lock_guard<std::mutex> lock(map_mutex);

        ros::Time start = ros::Time::now();
        map->insert_pointcloud(*pcl_cloud_world, origin, sensor_up,
                               (float)ds_resolution, (float)free_resolution,
                               (float)max_range,
                               (float)orientation.x(), (float)orientation.y(),
                               (float)orientation.z(), (float)orientation.w());
        ros::Time end = ros::Time::now();
        ROS_INFO_STREAM("Map update finished in " << (end - start).toSec() << "s");

        last_position  = translation;
        last_orientation = orientation;
        past_viewpoints.push_back(tf::Transform(orientation, translation));
    }

    // Publish viewpoints outside the mutex: viewpoints_msg is only touched by
    // cloudHandler, and ROS publishers are thread-safe.
    {
        geometry_msgs::Pose pose;
        pose.position.x = translation.x();
        pose.position.y = translation.y();
        pose.position.z = translation.z();
        pose.orientation.x = orientation.x();
        pose.orientation.y = orientation.y();
        pose.orientation.z = orientation.z();
        pose.orientation.w = orientation.w();
        viewpoints_msg.poses.push_back(pose);
        viewpoints_msg.header.frame_id = frame_id;
        viewpoints_msg.header.stamp = ros::Time::now();
        viewpoints_pub.publish(viewpoints_msg);
    }
}

// Periodic map visualization. Viewpoints are published separately from
// cloudHandler (on change), so this timer only drives the map marker refresh.
// Runs on a ros::Timer decoupled from the sonar callback so map updates
// stay at sonar rate even as the map grows and the viz loop becomes heavy.
void publishMapVisualization(const ros::TimerEvent&) {
    if (map == nullptr) return;

    // ---- Map visualisation ----
    {
        std::lock_guard<std::mutex> lock(map_mutex);
        ros::Time start2 = ros::Time::now();

        m_pub_occ->clear();
        m_pub_occ_coarse->clear();
        m_pub_free->clear();
        m_pub_free_txt->clear();
        m_pub_occ_txt->clear();
        m_pub_unk_txt->clear();
        m_pub_unc->clear();
        m_pub_unk->clear();
        m_pub_var->clear();
        m_pub_alpha_beta->clear();
        m_pub_constraint->clear();

        float max_vis_radius_sq = (max_vis_radius > 0) ? (float)(max_vis_radius * max_vis_radius) : -1.0f;

        // Block-level bbox gating: when max_vis_radius is set, only iterate
        // leaves in blocks whose center is within the visualization sphere
        // (plus a block-half-diagonal safety margin). Drops per-tick cost from
        // O(total map) to O(visible blocks). Per-leaf radius check below still
        // catches leaves at block edges that fall outside the sphere.
        la3dm::point3f viz_center(
            (float)last_position.x(), (float)last_position.y(), (float)last_position.z());
        auto leaf_begin = (max_vis_radius > 0)
            ? map->begin_leaf_in_sphere(viz_center, (float)max_vis_radius)
            : map->begin_leaf();

        float coarse_size = (float)resolution * (float)(1 << coarse_depth_steps);
        std::unordered_set<int64_t> seen_coarse;

        for (auto it = leaf_begin; it != map->end_leaf(); ++it) {

            la3dm::point3f p = it.get_loc();

            if (max_vis_radius_sq > 0) {
                float dist_sq = (p.x() - last_position.x()) * (p.x() - last_position.x()) +
                                (p.y() - last_position.y()) * (p.y() - last_position.y()) +
                                (p.z() - last_position.z()) * (p.z() - last_position.z());
                if (dist_sq > max_vis_radius_sq) continue;
            }

            if (it.get_node().get_state() == la3dm::State::OCCUPIED) {
                if (original_size)
                {
                    m_pub_occ->insert_point3d(p.x(), p.y(), p.z(), min_z, max_z, it.get_size());
                    // float dist_sq = (p.x() - last_position.x()) * (p.x() - last_position.x()) +
                    //                 (p.y() - last_position.y()) * (p.y() - last_position.y()) +
                    //                 (p.z() - last_position.z()) * (p.z() - last_position.z());
                    // if (dist_sq < 5.0f) {
                    //     char text[50];
                    //     snprintf(text, sizeof(text), "A:%.2f B:%.2f", it.get_node().get_A(), it.get_node().get_B());
                    //     m_pub_occ_txt->insert_text3d(p.x(), p.y(), p.z(), text, it.get_size());
                    // }
                }
                else
                {
                    auto pruned = it.get_pruned_locs();
                    for (auto n = pruned.cbegin(); n < pruned.cend(); ++n)
                    {
                        m_pub_occ->insert_point3d(n->x(), n->y(), n->z(), min_z, max_z, map->get_resolution());
                    //     float dist_sq = (n->x() - last_position.x()) * (n->x() - last_position.x()) +
                    //                     (n->y() - last_position.y()) * (n->y() - last_position.y()) +
                    //                     (n->z() - last_position.z()) * (n->z() - last_position.z());
                    //     if (dist_sq < 5.0f) {
                    //         char text[50];
                    //         snprintf(text, sizeof(text), "A:%.2f B:%.2f", it.get_node().get_A(), it.get_node().get_B());
                    //         m_pub_occ_txt->insert_text3d(n->x(), n->y(), n->z(), text, map->get_resolution());
                    //     }
                    }
                }
                // Coarse voxel: snap to inflated grid for path-planner safety margin
                int ix = (int)std::floor(p.x() / coarse_size);
                int iy = (int)std::floor(p.y() / coarse_size);
                int iz = (int)std::floor(p.z() / coarse_size);
                int64_t key = ((int64_t)(ix & 0xFFFFF) << 40)
                            | ((int64_t)(iy & 0xFFFFF) << 20)
                            |  (int64_t)(iz & 0xFFFFF);
                if (seen_coarse.insert(key).second) {
                    float cx = (ix + 0.5f) * coarse_size;
                    float cy = (iy + 0.5f) * coarse_size;
                    float cz = (iz + 0.5f) * coarse_size;
                    m_pub_occ_coarse->insert_point3d(cx, cy, cz, min_z, max_z, coarse_size);
                }
            }
            // Free-cell visualization commented out: the free-space volume at 40 m sonar
            // range grows to hundreds of millions of voxels and causes a >1 GB TCPROS
            // message that crashes RViz and disrupts the ROS bridge.  The map data itself
            // is unaffected; re-enable here if you need a bounded debug view.
            // else if(it.get_node().get_state() == la3dm::State::FREE)
            // {
            //     if (original_size)
            //     {
            //         m_pub_free->insert_point3d(p.x(), p.y(), p.z(), min_z, max_z, it.get_size(), it.get_node().get_prob());
            //         float dist_sq = (p.x() - last_position.x()) * (p.x() - last_position.x()) +
            //                         (p.y() - last_position.y()) * (p.y() - last_position.y()) +
            //                         (p.z() - last_position.z()) * (p.z() - last_position.z());
            //         if (dist_sq < 5.0f) {
            //             char text[50];
            //             snprintf(text, sizeof(text), "A:%.2f B:%.2f", it.get_node().get_A(), it.get_node().get_B());
            //             m_pub_free_txt->insert_text3d(p.x(), p.y(), p.z(), text, it.get_size());
            //         }
            //     }
            //     else
            //     {
            //         auto pruned = it.get_pruned_locs();
            //         for (auto n = pruned.cbegin(); n < pruned.cend(); ++n)
            //         {
            //             m_pub_free->insert_point3d(n->x(), n->y(), n->z(), min_z, max_z, map->get_resolution(), it.get_node().get_prob());
            //             float dist_sq = (n->x() - last_position.x()) * (n->x() - last_position.x()) +
            //                             (n->y() - last_position.y()) * (n->y() - last_position.y()) +
            //                             (n->z() - last_position.z()) * (n->z() - last_position.z());
            //             if (dist_sq < 5.0f) {
            //                 char text[50];
            //                 snprintf(text, sizeof(text), "A:%.2f B:%.2f", it.get_node().get_A(), it.get_node().get_B());
            //                 m_pub_free_txt->insert_text3d(n->x(), n->y(), n->z(), text, map->get_resolution());
            //             }
            //         }
            //     }
            // }
            else if(it.get_node().get_state() == la3dm::State::UNCERTAIN)
            {
                if (original_size)
                {
                    m_pub_unc->insert_point3d_color(p.x(), p.y(), p.z(), it.get_size(), 1.0f, 1.0f, 0.0f);
                }
                else
                {
                    auto pruned = it.get_pruned_locs();
                    for (auto n = pruned.cbegin(); n < pruned.cend(); ++n)
                    {
                        m_pub_unc->insert_point3d_color(n->x(), n->y(), n->z(), map->get_resolution(), 1.0f, 1.0f, 0.0f);
                    }
                }
            }
            // else if(it.get_node().get_state() == la3dm::State::UNKNOWN)
            // {
            //     if (original_size)
            //     {
            //         m_pub_unk->insert_point3d(p.x(), p.y(), p.z(), min_z, max_z, it.get_size());
            //         float dist_sq = (p.x() - last_position.x()) * (p.x() - last_position.x()) +
            //                         (p.y() - last_position.y()) * (p.y() - last_position.y()) +
            //                         (p.z() - last_position.z()) * (p.z() - last_position.z());
            //         if (dist_sq < 15.0f) {
            //             char text[50];
            //             snprintf(text, sizeof(text), "A:%.2f B:%.2f", it.get_node().get_A(), it.get_node().get_B());
            //             m_pub_unk_txt->insert_text3d(p.x(), p.y(), p.z(), text, it.get_size());
            //         }
            //     }
            //     else
            //     {
            //         auto pruned = it.get_pruned_locs();
            //         for (auto n = pruned.cbegin(); n < pruned.cend(); ++n)
            //         {
            //             m_pub_unk->insert_point3d(n->x(), n->y(), n->z(), min_z, max_z, map->get_resolution());
            //             float dist_sq = (n->x() - last_position.x()) * (n->x() - last_position.x()) +
            //                             (n->y() - last_position.y()) * (n->y() - last_position.y()) +
            //                             (n->z() - last_position.z()) * (n->z() - last_position.z());
            //             if (dist_sq < 15.0f) {
            //                 char text[50];
            //                 snprintf(text, sizeof(text), "A:%.2f B:%.2f", it.get_node().get_A(), it.get_node().get_B());
            //                 m_pub_unk_txt->insert_text3d(n->x(), n->y(), n->z(), text, map->get_resolution());
            //             }
            //         }
            //     }
            // }

            // auto state = it.get_node().get_state();
            // if (state == la3dm::State::OCCUPIED || state == la3dm::State::FREE || state == la3dm::State::UNCERTAIN) {
            //     std::string ns = (state == la3dm::State::OCCUPIED) ? "occupied" : (state == la3dm::State::FREE) ? "free" : "uncertain";
            //     if (original_size)
            //     {
            //         m_pub_var->insert_color_point3d(p.x(), p.y(), p.z(), 0.0, max_var_vis, it.get_node().get_var(), it.get_size(), ns);
            //     }
            //     else
            //     {
            //         auto pruned = it.get_pruned_locs();
            //         for (auto n = pruned.cbegin(); n < pruned.cend(); ++n)
            //         {
            //             m_pub_var->insert_color_point3d(n->x(), n->y(), n->z(), 0.0, max_var_vis, it.get_node().get_var(), map->get_resolution(), ns);
            //         }
            //     }
            // }

            auto state = it.get_node().get_state();
            if (state == la3dm::State::OCCUPIED || state == la3dm::State::UNCERTAIN) {
                std::string ns = (state == la3dm::State::OCCUPIED) ? "occupied" : "uncertain";
                float t = (float)std::min(std::max(std::pow(it.get_node().get_var() / (float)max_var_vis, (float)var_vis_gamma), 0.0f), 1.0f);
                float A = it.get_node().get_A();
                float B = it.get_node().get_B();
                if (original_size)
                {
                    m_pub_var->insert_point3d_color(p.x(), p.y(), p.z(), it.get_size(), t, 1.0f - t, 0.0f, 1.0f, ns);
                    m_pub_alpha_beta->insert_point3d_color(p.x(), p.y(), p.z(), it.get_size(), A, B, 0.0f, 1.0f, ns);
                }
                else
                {
                    auto pruned = it.get_pruned_locs();
                    for (auto n = pruned.cbegin(); n < pruned.cend(); ++n)
                    {
                        m_pub_var->insert_point3d_color(n->x(), n->y(), n->z(), map->get_resolution(), t, 1.0f - t, 0.0f, 1.0f, ns);
                        m_pub_alpha_beta->insert_point3d_color(n->x(), n->y(), n->z(), map->get_resolution(), A, B, 0.0f, 1.0f, ns);
                    }
                }
            }

            if (state == la3dm::State::OCCUPIED) {
                double lam = it.get_node().get_lambda_max_cache();
                double inv = std::max(0.0, max_constraint_vis - lam);
                if (original_size)
                {
                    m_pub_constraint->insert_color_point3d(p.x(), p.y(), p.z(), 0.0, max_constraint_vis, inv, it.get_size(), "occupied");
                }
                else
                {
                    auto pruned = it.get_pruned_locs();
                    for (auto n = pruned.cbegin(); n < pruned.cend(); ++n)
                    {
                        m_pub_constraint->insert_color_point3d(n->x(), n->y(), n->z(), 0.0, max_constraint_vis, inv, map->get_resolution(), "occupied");
                    }
                }
            }

        }

        m_pub_occ->publish();
        m_pub_occ_coarse->publish();
        m_pub_free->publish();
        m_pub_free_txt->publish();
        m_pub_occ_txt->publish();
        m_pub_unk_txt->publish();
        m_pub_unc->publish();
        m_pub_unk->publish();
        m_pub_var->publish();
        m_pub_alpha_beta->publish();
        m_pub_constraint->publish();

        ros::Time end2 = ros::Time::now();
        ROS_INFO_STREAM("One map published in " << (end2 - start2).toSec() << "s");
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "bgkloctomap_server");
    ros::NodeHandle nh("~");
    //incoming pointcloud topic, this could be put into the .yaml too
    std::string cloud_topic("/sonar_point_cloud");

    //Universal parameters
    nh.param<std::string>("topic", map_topic_occ, map_topic_occ);
    nh.param<std::string>("topic_free", map_topic_free, map_topic_free);
    nh.param<std::string>("topic_free_txt", map_topic_free_txt, map_topic_free_txt);
    nh.param<std::string>("topic_unk", map_topic_unk, map_topic_unk);
    nh.param<double>("max_range", max_range, max_range);
    nh.param<double>("resolution", resolution, resolution);
    nh.param<double>("ds_resolution", ds_resolution, ds_resolution);
    nh.param<int>("block_depth", block_depth, block_depth);
    nh.param<double>("sf2", sf2, sf2);
    nh.param<double>("ell", ell, ell);
    nh.param<double>("free_resolution", free_resolution, free_resolution);
    nh.param<double>("free_thresh", free_thresh, free_thresh);
    nh.param<double>("occupied_thresh", occupied_thresh, occupied_thresh);
    nh.param<double>("min_z", min_z, min_z);
    nh.param<double>("max_z", max_z, max_z);
    nh.param<bool>("original_size", original_size, original_size);
    nh.param<double>("max_var_vis",       max_var_vis,       max_var_vis);
    nh.param<double>("var_vis_gamma",     var_vis_gamma,     var_vis_gamma);
    nh.param<double>("max_constraint_vis", max_constraint_vis, max_constraint_vis);
    nh.param<double>("max_vis_radius", max_vis_radius, max_vis_radius);
    nh.param<double>("viz_rate", viz_rate, viz_rate);
    nh.param<std::string>("frame_id", frame_id, frame_id);
    nh.param<std::string>("topic_occ_coarse", map_topic_occ_coarse, map_topic_occ_coarse);
    nh.param<int>("coarse_depth_steps", coarse_depth_steps, coarse_depth_steps);

    //BKGL parameters
    nh.param<float>("var_thresh",  var_thresh,  var_thresh);
    nh.param<float>("prior_A",     prior_A,     prior_A);
    nh.param<float>("prior_B",     prior_B,     prior_B);
    nh.param<float>("theta_bw",    theta_bw,    theta_bw);
    nh.param<float>("phi_bw",      phi_bw,      phi_bw);
    nh.param<bool>("free_ray_range_weight", free_ray_range_weight, free_ray_range_weight);
    nh.param<float>("swath_angle", swath_angle, swath_angle);
    nh.param<float>("tau_info", tau_info, tau_info);
    nh.param<float>("tau_var",  tau_var,  tau_var);
    nh.param<float>("delta",    delta,    delta);

    // Pose-level novelty weighting
    nh.param<bool> ("use_pose_level_weighting", use_pose_level_weighting, use_pose_level_weighting);
    nh.param<int>  ("pose_history_size",         pose_history_size,        pose_history_size);
    nh.param<float>("pose_novelty_sigma",         pose_novelty_sigma,       pose_novelty_sigma);
    nh.param<float>("pose_w_roll",                pose_w_roll,              pose_w_roll);
    nh.param<float>("pose_w_pitch",               pose_w_pitch,             pose_w_pitch);
    nh.param<float>("pose_w_yaw",                 pose_w_yaw,               pose_w_yaw);
    nh.param<float>("pose_w_vx_per_l2",           pose_w_vx_l2,             pose_w_vx_l2);
    nh.param<float>("pose_w_vy_per_l2",           pose_w_vy_l2,             pose_w_vy_l2);
    nh.param<float>("pose_w_vz_per_l2",           pose_w_vz_l2,             pose_w_vz_l2);

    // Ablation flags
    nh.param<bool>("ablate_directional_weights", ablate_directional_weights, ablate_directional_weights);

    // Per-voxel debug tracing
    nh.param<bool> ("debug_voxel_enable", debug_voxel_enable, debug_voxel_enable);
    nh.param<float>("debug_voxel_x",      debug_voxel_x,      debug_voxel_x);
    nh.param<float>("debug_voxel_y",      debug_voxel_y,      debug_voxel_y);
    nh.param<float>("debug_voxel_z",      debug_voxel_z,      debug_voxel_z);

    ROS_INFO_STREAM("Parameters:" << std::endl <<
            "frame_id: " << frame_id << std::endl <<
            "topic: " << map_topic_occ << std::endl <<
            "max_range: " << max_range << std::endl <<
            "resolution: " << resolution << std::endl <<
            "ds_resolution: " << ds_resolution << std::endl <<
            "block_depth: " << block_depth << std::endl <<
            "sf2: " << sf2 << std::endl <<
            "ell: " << ell << std::endl <<
            "free_resolution: " << free_resolution << std::endl <<
            "free_thresh: " << free_thresh << std::endl <<
            "occupied_thresh: " << occupied_thresh << std::endl <<
            "min_z: " << min_z << std::endl <<
            "max_z: " << max_z << std::endl <<
            "original_size: " << original_size << std::endl <<
            "max_var_vis: " << max_var_vis << std::endl <<
            "max_vis_radius: " << max_vis_radius << std::endl <<
            "viz_rate: " << viz_rate << std::endl <<
            "var_thresh: " << var_thresh << std::endl <<
            "prior_A: " << prior_A << std::endl <<
            "prior_B: " << prior_B << std::endl <<
            "theta_bw: " << theta_bw << std::endl <<
            "phi_bw: " << phi_bw << std::endl <<
            "free_ray_range_weight: " << free_ray_range_weight << std::endl <<
            "swath_angle: " << swath_angle << std::endl <<
            "coarse_depth_steps: " << coarse_depth_steps << std::endl <<
            "use_pose_level_weighting: " << use_pose_level_weighting << std::endl <<
            "pose_history_size: " << pose_history_size << std::endl <<
            "pose_novelty_sigma: " << pose_novelty_sigma << std::endl <<
            "ablate_directional_weights: " << ablate_directional_weights << std::endl <<
            "tau_info: " << tau_info << std::endl <<
            "tau_var: "  << tau_var  << std::endl <<
            "delta: "    << delta
            );

    map = new la3dm::BGKLOctoMap(resolution, block_depth, sf2, ell, free_thresh, occupied_thresh,
                                  var_thresh, prior_A, prior_B, theta_bw, phi_bw, free_ray_range_weight);

    map->configure_pose_level_weighting(use_pose_level_weighting, pose_history_size,
                                        pose_novelty_sigma,
                                        pose_w_roll, pose_w_pitch, pose_w_yaw,
                                        pose_w_vx_l2, pose_w_vy_l2, pose_w_vz_l2);
    map->set_ablate_directional_weights(ablate_directional_weights);
    map->set_debug_voxel(debug_voxel_enable, debug_voxel_x, debug_voxel_y, debug_voxel_z);
    la3dm::OcTreeNode::tau_info = tau_info;
    la3dm::OcTreeNode::tau_var  = tau_var;
    la3dm::OcTreeNode::delta    = delta;

    map->configure_frustum(swath_angle);

    ros::Subscriber point_sub = nh.subscribe<sensor_msgs::PointCloud2>(cloud_topic, 1, cloudHandler);
    ros::Subscriber clicked_sub = nh.subscribe<geometry_msgs::PointStamped>("/clicked_point", 1, clickedPointHandler);
    m_pub_occ = new la3dm::MarkerArrayPub(nh, map_topic_occ, resolution, {"map"}, frame_id);
    m_pub_occ_coarse = new la3dm::MarkerArrayPub(nh, map_topic_occ_coarse, resolution, {"map"}, frame_id);
    m_pub_free = new la3dm::MarkerArrayPub(nh, map_topic_free, resolution, {"map"}, frame_id);
    m_pub_free_txt = new la3dm::TextMarkerArrayPub(nh, map_topic_free_txt, resolution, frame_id);
    m_pub_occ_txt = new la3dm::TextMarkerArrayPub(nh, map_topic_occ_txt, resolution, frame_id);
    m_pub_unk_txt = new la3dm::TextMarkerArrayPub(nh, map_topic_unk_txt, resolution, frame_id);
    m_pub_unc = new la3dm::MarkerArrayPub(nh, map_topic_unc, resolution, {"map"}, frame_id);
    m_pub_unk = new la3dm::MarkerArrayPub(nh, map_topic_unk, resolution, {"map"}, frame_id);
    m_pub_var = new la3dm::MarkerArrayPub(nh, map_topic_var, resolution, {"occupied", "free", "uncertain"}, frame_id);
    m_pub_alpha_beta = new la3dm::MarkerArrayPub(nh, map_topic_alpha_beta, resolution, {"occupied", "free", "uncertain"}, frame_id);
    m_pub_constraint = new la3dm::MarkerArrayPub(nh, map_topic_constraint, resolution, {"occupied"}, frame_id);

    viewpoints_pub = nh.advertise<geometry_msgs::PoseArray>("viewpoints", 1, true);

    listener = new tf::TransformListener();

    // Decouple visualization from the sonar callback: run it on its own timer
    // so map updates stay at sonar rate even when the viz pass becomes heavy.
    ros::Timer viz_timer;
    if (viz_rate > 0.0) {
        viz_timer = nh.createTimer(ros::Duration(1.0 / viz_rate), publishMapVisualization);
    } else {
        ROS_WARN("viz_rate <= 0: map visualization disabled.");
    }

    // Two threads: one for sonar callbacks, one for the viz timer.
    // The sonar callback does TF+PCL outside the mutex, so it can overlap
    // with the viz loop for those fast steps even while viz holds map_mutex.
    ros::AsyncSpinner spinner(2);
    spinner.start();
    ros::waitForShutdown();

    return 0;
}
