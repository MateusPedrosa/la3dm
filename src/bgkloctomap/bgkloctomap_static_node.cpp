#include <string>
#include <iostream>
#include <ros/ros.h>
#include "bgkloctomap.h"
#include "bgkloctree_node.h"
#include "markerarray_pub.h"

void load_pcd(std::string filename, la3dm::point3f &origin, la3dm::point3f &sensor_up,
              float &qx, float &qy, float &qz, float &qw,
              la3dm::PCLPointCloud &cloud) {
    pcl::PCLPointCloud2 cloud2;
    Eigen::Vector4f _origin;
    Eigen::Quaternionf orientation;
    pcl::io::loadPCDFile(filename, cloud2, _origin, orientation);
    pcl::fromPCLPointCloud2(cloud2, cloud);
    origin.x() = _origin[0];
    origin.y() = _origin[1];
    origin.z() = _origin[2];

    Eigen::Matrix3f mat = orientation.toRotationMatrix();
    Eigen::Vector3f up = mat.col(2);
    sensor_up = la3dm::point3f(up.x(), up.y(), up.z());

    qx = orientation.x();
    qy = orientation.y();
    qz = orientation.z();
    qw = orientation.w();
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "bgkloctomap_static_node");
    ros::NodeHandle nh("~");

    std::string dir;
    std::string prefix;
    int scan_num = 0;
    std::string map_topic("/occupied_cells_vis_array");
    std::string map_topic2("/free_cells_vis_array");
    double max_range = -1;
    double resolution = 0.1;
    int block_depth = 4;
    double sf2 = 1.0;
    double ell = 1.0;
    double free_resolution = 0.5;
    double ds_resolution = 0.1;
    double free_thresh = 0.3;
    double occupied_thresh = 0.7;
    double min_z = 0;
    double max_z = 0;
    bool original_size = false;
    float var_thresh = 1.0f;
    float prior_A = 1.0f;
    float prior_B = 1.0f;
    float theta_bw = 0.6f * 3.1415926f / 180.0f;
    float phi_bw = 20.0f * 3.1415926f / 180.0f;
    bool free_ray_range_weight = false;
    float swath_angle = -1.0f;
    float tau_info = 2.0f;
    float tau_var  = 0.01f;
    float delta    = 0.05f;
    bool ablate_directional_weights = false;
    double max_var_vis = 0.25;
    double var_vis_gamma = 1.0;
    std::string map_topic_var("/variance_vis_array");

    nh.param<std::string>("dir", dir, dir);
    nh.param<std::string>("prefix", prefix, prefix);
    nh.param<std::string>("topic", map_topic, map_topic);
    nh.param<std::string>("topic2", map_topic2, map_topic2);
    nh.param<int>("scan_num", scan_num, scan_num);
    nh.param<double>("max_range", max_range, max_range);
    nh.param<double>("resolution", resolution, resolution);
    nh.param<int>("block_depth", block_depth, block_depth);
    nh.param<double>("sf2", sf2, sf2);
    nh.param<double>("ell", ell, ell);
    nh.param<double>("free_resolution", free_resolution, free_resolution);
    nh.param<double>("ds_resolution", ds_resolution, ds_resolution);
    nh.param<double>("free_thresh", free_thresh, free_thresh);
    nh.param<double>("occupied_thresh", occupied_thresh, occupied_thresh);
    nh.param<double>("min_z", min_z, min_z);
    nh.param<double>("max_z", max_z, max_z);
    nh.param<bool>("original_size", original_size, original_size);
    nh.param<float>("var_thresh", var_thresh, var_thresh);
    nh.param<float>("prior_A", prior_A, prior_A);
    nh.param<float>("prior_B", prior_B, prior_B);
    nh.param<float>("theta_bw", theta_bw, theta_bw);
    nh.param<float>("phi_bw", phi_bw, phi_bw);
    nh.param<bool> ("free_ray_range_weight", free_ray_range_weight, free_ray_range_weight);
    nh.param<float>("swath_angle", swath_angle, swath_angle);
    nh.param<float>("tau_info", tau_info, tau_info);
    nh.param<float>("tau_var",  tau_var,  tau_var);
    nh.param<float>("delta",    delta,    delta);
    nh.param<bool> ("ablate_directional_weights", ablate_directional_weights, ablate_directional_weights);
    nh.param<double>("max_var_vis",   max_var_vis,   max_var_vis);
    nh.param<double>("var_vis_gamma", var_vis_gamma, var_vis_gamma);

    ROS_INFO_STREAM("Parameters:" << std::endl <<
            "dir: " << dir << std::endl <<
            "prefix: " << prefix << std::endl <<
            "topic: " << map_topic << std::endl <<
            "scan_num: " << scan_num << std::endl <<
            "max_range: " << max_range << std::endl <<
            "resolution: " << resolution << std::endl <<
            "block_depth: " << block_depth << std::endl <<
            "sf2: " << sf2 << std::endl <<
            "ell: " << ell << std::endl <<
            "free_resolution: " << free_resolution << std::endl <<
            "ds_resolution: " << ds_resolution << std::endl <<
            "free_thresh: " << free_thresh << std::endl <<
            "occupied_thresh: " << occupied_thresh << std::endl <<
            "min_z: " << min_z << std::endl <<
            "max_z: " << max_z << std::endl <<
            "original_size: " << original_size << std::endl <<
            "var_thresh: " << var_thresh << std::endl <<
            "prior_A: " << prior_A << std::endl <<
            "prior_B: " << prior_B << std::endl <<
            "theta_bw: " << theta_bw << std::endl <<
            "phi_bw: " << phi_bw << std::endl <<
            "free_ray_range_weight: " << free_ray_range_weight << std::endl <<
            "swath_angle: " << swath_angle << std::endl <<
            "tau_info: " << tau_info << std::endl <<
            "tau_var: "  << tau_var  << std::endl <<
            "delta: "    << delta    << std::endl <<
            "ablate_directional_weights: " << ablate_directional_weights
            );

    la3dm::BGKLOctoMap map(resolution, block_depth, sf2, ell, free_thresh, occupied_thresh,
                           var_thresh, prior_A, prior_B, theta_bw, phi_bw, free_ray_range_weight);

    map.configure_frustum(swath_angle);
    map.set_ablate_directional_weights(ablate_directional_weights);
    la3dm::OcTreeNode::tau_info = tau_info;
    la3dm::OcTreeNode::tau_var  = tau_var;
    la3dm::OcTreeNode::delta    = delta;

    ros::Time start = ros::Time::now();
    for (int scan_id = 1; scan_id <= scan_num; ++scan_id) {
        la3dm::PCLPointCloud cloud;
        la3dm::point3f origin, sensor_up;
        float qx, qy, qz, qw;
        std::string filename(dir + "/" + prefix + "_" + std::to_string(scan_id) + ".pcd");
        load_pcd(filename, origin, sensor_up, qx, qy, qz, qw, cloud);

        map.insert_pointcloud(cloud, origin, sensor_up,
                              (float)ds_resolution, (float)free_resolution, (float)max_range,
                              qx, qy, qz, qw);
        ROS_INFO_STREAM("Scan " << scan_id << " done");
    }
    ros::Time end = ros::Time::now();
    ROS_INFO_STREAM("Mapping finished in " << (end - start).toSec() << "s");

    ///////// Publish Map /////////////////////
    la3dm::MarkerArrayPub m_pub(nh, map_topic, resolution);
    la3dm::MarkerArrayPub m_pub_var(nh, map_topic_var, resolution, {"occupied", "uncertain"});
    if (min_z == max_z) {
        la3dm::point3f lim_min, lim_max;
        map.get_bbox(lim_min, lim_max);
        min_z = lim_min.z();
        max_z = lim_max.z();
    }

    for (auto it = map.begin_leaf(); it != map.end_leaf(); ++it) {
        la3dm::point3f p = it.get_loc();
        auto state = it.get_node().get_state();

        if (state == la3dm::State::OCCUPIED) {
            if (original_size) {
                m_pub.insert_point3d(p.x(), p.y(), p.z(), min_z, max_z, it.get_size());
            } else {
                auto pruned = it.get_pruned_locs();
                for (auto n = pruned.cbegin(); n < pruned.cend(); ++n)
                    m_pub.insert_point3d(n->x(), n->y(), n->z(), min_z, max_z, map.get_resolution());
            }
        }

        if (state == la3dm::State::OCCUPIED || state == la3dm::State::UNCERTAIN) {
            std::string ns = (state == la3dm::State::OCCUPIED) ? "occupied" : "uncertain";
            float t = (float)std::min(std::max(
                std::pow(it.get_node().get_var() / (float)max_var_vis, (float)var_vis_gamma),
                0.0f), 1.0f);
            if (original_size) {
                m_pub_var.insert_point3d_color(p.x(), p.y(), p.z(), it.get_size(), t, 1.0f - t, 0.0f, 1.0f, ns);
            } else {
                auto pruned = it.get_pruned_locs();
                for (auto n = pruned.cbegin(); n < pruned.cend(); ++n)
                    m_pub_var.insert_point3d_color(n->x(), n->y(), n->z(), map.get_resolution(), t, 1.0f - t, 0.0f, 1.0f, ns);
            }
        }
    }

    m_pub.publish();
    m_pub_var.publish();
    ros::spin();

    return 0;
}
