#include <string>
#include <iostream>
#include <ros/ros.h>
#include <pcl_ros/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include "markerarray_pub.h"
#include "bgkloctomap.h"

tf::TransformListener *listener;
std::string frame_id("/map");
la3dm::BGKLOctoMap *map;

la3dm::MarkerArrayPub *m_pub_occ, *m_pub_free, *m_pub_unc, *m_pub_unk, *m_pub_var;
la3dm::TextMarkerArrayPub *m_pub_free_txt;

//startup parameters
tf::Vector3 last_position;
tf::Quaternion last_orientation;
bool first = true;
double position_change_thresh = 0.1;
double orientation_change_thresh = 0.2;
bool updated = false;

//Universal parameters
std::string map_topic_occ("/occupied_cells_vis_array");
std::string map_topic_free("/free_cells_vis_array");
std::string map_topic_free_txt("/free_cells_txt_vis_array");
std::string map_topic_unc("/uncertain_cells_vis_array");
std::string map_topic_unk("/unknown_cells_vis_array");
std::string map_topic_var("/variance_vis_array");
double max_range = -1;
double resolution = 0.1;
int block_depth = 4;
double sf2 = 0.1;
double ell = 0.2;
double free_resolution = 0.65;
double ds_resolution = 0.1;
double free_thresh = 0.3;
double occupied_thresh = 0.7;
double min_z = 0;
double max_z = 0;
bool original_size = true;
double max_var_vis = 0.25;

//BGKL parameters
float var_thresh = 1.0f;
float prior_A = 1.0f;
float prior_B = 1.0f;
float theta_bw = 0.6f * 3.1415926f / 180.0f;
float phi_bw = 20.0f * 3.1415926f / 180.0f;

void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &cloud) {
    
    tf::StampedTransform transform;
    try {
        listener->waitForTransform(frame_id, cloud->header.frame_id, cloud->header.stamp, ros::Duration(5.0));
        listener->lookupTransform(frame_id, cloud->header.frame_id, cloud->header.stamp, transform); //ros::Time::now() -- Don't use this because processing time delay breaks it
    } catch (tf::TransformException ex) {
        ROS_ERROR("%s", ex.what());
        return;
    }

    ros::Time start = ros::Time::now();
    la3dm::point3f origin;
    tf::Vector3 translation = transform.getOrigin();
    tf::Quaternion orientation = transform.getRotation();

    if (first || orientation.angleShortestPath(last_orientation) > orientation_change_thresh || translation.distance(last_position) > position_change_thresh) 
    {
        ROS_INFO_STREAM("Cloud received");
        
        tf::Matrix3x3 mat(orientation);
        tf::Vector3 up = mat.getColumn(2);

        la3dm::point3f sensor_up(up.x(), up.y(), up.z());
        
        last_position = translation;
        last_orientation = orientation;
        origin.x() = (float) translation.x();
        origin.y() = (float) translation.y();
        origin.z() = (float) translation.z();

        sensor_msgs::PointCloud2 cloud_map;
        pcl_ros::transformPointCloud(frame_id, *cloud, cloud_map, *listener);

        //pointer required for downsampling
        la3dm::PCLPointCloud::Ptr pcl_cloud (new la3dm::PCLPointCloud());
        pcl::fromROSMsg(cloud_map, *pcl_cloud);

        //downsample for faster mapping
        la3dm::PCLPointCloud filtered_cloud;
        pcl::VoxelGrid<pcl::PointXYZI> filterer;
        filterer.setInputCloud(pcl_cloud);
        filterer.setLeafSize(ds_resolution, ds_resolution, ds_resolution);
        filterer.filter(filtered_cloud);

        if(filtered_cloud.size() > 5){
            map->insert_pointcloud(filtered_cloud, origin, sensor_up, (float) resolution, (float) free_resolution, (float) max_range);
        }

        ros::Time end = ros::Time::now();
        ROS_INFO_STREAM("One cloud finished in " << (end - start).toSec() << "s");
        updated = true;
    }


    if (updated) 
    {
        ros::Time start2 = ros::Time::now();

        m_pub_occ->clear();
        m_pub_free->clear();
        m_pub_free_txt->clear();
        m_pub_unc->clear();
        m_pub_unk->clear();
        m_pub_var->clear();

        for (auto it = map->begin_leaf(); it != map->end_leaf(); ++it) {

            la3dm::point3f p = it.get_loc();

            if (it.get_node().get_state() == la3dm::State::OCCUPIED) {
                if (original_size) 
                {
                    m_pub_occ->insert_point3d(p.x(), p.y(), p.z(), min_z, max_z, it.get_size());
                } 
                else 
                {
                    auto pruned = it.get_pruned_locs();
                    for (auto n = pruned.cbegin(); n < pruned.cend(); ++n) 
                    {
                        m_pub_occ->insert_point3d(n->x(), n->y(), n->z(), min_z, max_z, map->get_resolution());
                    }
                }
            }
            else if(it.get_node().get_state() == la3dm::State::FREE)
            {
                if (original_size) 
                {
                    m_pub_free->insert_point3d(p.x(), p.y(), p.z(), min_z, max_z, it.get_size(), it.get_node().get_prob());
                    float dist_sq = (p.x() - last_position.x()) * (p.x() - last_position.x()) + 
                                    (p.y() - last_position.y()) * (p.y() - last_position.y()) + 
                                    (p.z() - last_position.z()) * (p.z() - last_position.z());
                    if (dist_sq < 5.0f) {
                        char text[50];
                        snprintf(text, sizeof(text), "A:%.2f B:%.2f", it.get_node().get_A(), it.get_node().get_B());
                        m_pub_free_txt->insert_text3d(p.x(), p.y(), p.z(), text, it.get_size());
                    }
                } 
                else 
                {
                    auto pruned = it.get_pruned_locs();
                    for (auto n = pruned.cbegin(); n < pruned.cend(); ++n) 
                    {
                        m_pub_free->insert_point3d(n->x(), n->y(), n->z(), min_z, max_z, map->get_resolution(), it.get_node().get_prob());
                        float dist_sq = (n->x() - last_position.x()) * (n->x() - last_position.x()) + 
                                        (n->y() - last_position.y()) * (n->y() - last_position.y()) + 
                                        (n->z() - last_position.z()) * (n->z() - last_position.z());
                        if (dist_sq < 5.0f) {
                            char text[50];
                            snprintf(text, sizeof(text), "A:%.2f B:%.2f", it.get_node().get_A(), it.get_node().get_B());
                            m_pub_free_txt->insert_text3d(n->x(), n->y(), n->z(), text, map->get_resolution());
                        }
                    }
                }
                
            }
            else if(it.get_node().get_state() == la3dm::State::UNCERTAIN)
            {
                if (original_size) 
                {
                    m_pub_unc->insert_point3d(p.x(), p.y(), p.z(), min_z, max_z, it.get_size());
                } 
                else 
                {
                    auto pruned = it.get_pruned_locs();
                    for (auto n = pruned.cbegin(); n < pruned.cend(); ++n) 
                    {
                        m_pub_unc->insert_point3d(n->x(), n->y(), n->z(), min_z, max_z, map->get_resolution());
                    }
                }
            }
            else if(it.get_node().get_state() == la3dm::State::UNKNOWN)
            {
                if (original_size) 
                {
                    m_pub_unk->insert_point3d(p.x(), p.y(), p.z(), min_z, max_z, it.get_size());
                } 
                else 
                {
                    auto pruned = it.get_pruned_locs();
                    for (auto n = pruned.cbegin(); n < pruned.cend(); ++n) 
                        m_pub_unk->insert_point3d(n->x(), n->y(), n->z(), min_z, max_z, map->get_resolution());
                }
            }

            auto state = it.get_node().get_state();
            if (state == la3dm::State::OCCUPIED || state == la3dm::State::FREE || state == la3dm::State::UNCERTAIN) {
                std::string ns = (state == la3dm::State::OCCUPIED) ? "occupied" : (state == la3dm::State::FREE) ? "free" : "uncertain";
                if (original_size) 
                {
                    m_pub_var->insert_color_point3d(p.x(), p.y(), p.z(), 0.0, max_var_vis, it.get_node().get_var(), it.get_size(), ns);
                } 
                else 
                {
                    auto pruned = it.get_pruned_locs();
                    for (auto n = pruned.cbegin(); n < pruned.cend(); ++n) 
                    {
                        m_pub_var->insert_color_point3d(n->x(), n->y(), n->z(), 0.0, max_var_vis, it.get_node().get_var(), map->get_resolution(), ns);
                    }
                }
            }
            
        }
        updated = false;

        m_pub_occ->publish();
        m_pub_free->publish();
        m_pub_free_txt->publish();
        m_pub_unc->publish();
        m_pub_unk->publish();
        m_pub_var->publish();

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
    nh.param<double>("max_var_vis", max_var_vis, max_var_vis);

    //BKGL parameters
    nh.param<float>("var_thresh", var_thresh, var_thresh);
    nh.param<float>("prior_A", prior_A, prior_A);
    nh.param<float>("prior_B", prior_B, prior_B);
    nh.param<float>("theta_bw", theta_bw, theta_bw);
    nh.param<float>("phi_bw", phi_bw, phi_bw);

    ROS_INFO_STREAM("Parameters:" << std::endl <<
            "topic: " << map_topic_occ << std::endl <<
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
            "max_var_vis: " << max_var_vis << std::endl <<
            "var_thresh: " << var_thresh << std::endl <<
            "prior_A: " << prior_A << std::endl <<
            "prior_B: " << prior_B << std::endl <<
            "theta_bw: " << theta_bw << std::endl <<
            "phi_bw: " << phi_bw
            );

    map = new la3dm::BGKLOctoMap(resolution, block_depth, sf2, ell, free_thresh, occupied_thresh, var_thresh, prior_A, prior_B, theta_bw, phi_bw);
    
    ros::Subscriber point_sub = nh.subscribe<sensor_msgs::PointCloud2>(cloud_topic, 1, cloudHandler);
    m_pub_occ = new la3dm::MarkerArrayPub(nh, map_topic_occ, resolution);
    m_pub_free = new la3dm::MarkerArrayPub(nh, map_topic_free, resolution);
    m_pub_free_txt = new la3dm::TextMarkerArrayPub(nh, map_topic_free_txt, resolution);
    m_pub_unc = new la3dm::MarkerArrayPub(nh, map_topic_unc, resolution);
    m_pub_unk = new la3dm::MarkerArrayPub(nh, map_topic_unk, resolution);
    m_pub_var = new la3dm::MarkerArrayPub(nh, map_topic_var, resolution, {"occupied", "free", "uncertain"});

    listener = new tf::TransformListener();
    
    while(ros::ok())
    {
    	ros::spin();
    }

    return 0;
}
