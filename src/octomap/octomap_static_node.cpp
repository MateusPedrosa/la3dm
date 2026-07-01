#include <string>
#include <iostream>
#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <octomap/octomap.h>
#include "markerarray_pub.h"

void load_pcd(const std::string &filename, octomap::point3d &origin, pcl::PointCloud<pcl::PointXYZ> &cloud) {
    pcl::PCLPointCloud2 cloud2;
    Eigen::Vector4f _origin;
    Eigen::Quaternionf orientation;
    pcl::io::loadPCDFile(filename, cloud2, _origin, orientation);
    pcl::fromPCLPointCloud2(cloud2, cloud);
    origin = octomap::point3d(_origin[0], _origin[1], _origin[2]);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "octomap_static_node");
    ros::NodeHandle nh("~");

    std::string dir, prefix;
    int scan_num = 0;
    std::string map_topic("/occupied_cells_vis_array");
    std::string map_topic_prob("/prob_vis_array");
    double max_range = -1, resolution = 0.1, ds_resolution = 0.1;
    double min_z = 0, max_z = 0;
    double occupied_thresh = 0.7;
    double prob_hit = 0.7, prob_miss = 0.4;
    double clamping_thres_min = 0.1192, clamping_thres_max = 0.971;
    double free_bbox_margin = 2.0;

    nh.param<std::string>("dir", dir, dir);
    nh.param<std::string>("prefix", prefix, prefix);
    nh.param<int>("scan_num", scan_num, scan_num);
    nh.param<double>("max_range", max_range, max_range);
    nh.param<double>("resolution", resolution, resolution);
    nh.param<double>("ds_resolution", ds_resolution, ds_resolution);
    nh.param<double>("min_z", min_z, min_z);
    nh.param<double>("max_z", max_z, max_z);
    nh.param<double>("occupied_thresh", occupied_thresh, occupied_thresh);
    nh.param<double>("prob_hit", prob_hit, prob_hit);
    nh.param<double>("prob_miss", prob_miss, prob_miss);
    nh.param<double>("clamping_thres_min", clamping_thres_min, clamping_thres_min);
    nh.param<double>("clamping_thres_max", clamping_thres_max, clamping_thres_max);
    nh.param<double>("free_bbox_margin", free_bbox_margin, free_bbox_margin);

    ROS_INFO_STREAM("Parameters:"
        << "\n  dir: " << dir
        << "\n  prefix: " << prefix
        << "\n  scan_num: " << scan_num
        << "\n  resolution: " << resolution
        << "\n  ds_resolution: " << ds_resolution
        << "\n  max_range: " << max_range
        << "\n  occupied_thresh: " << occupied_thresh
        << "\n  prob_hit: " << prob_hit
        << "\n  prob_miss: " << prob_miss
        << "\n  min_z: " << min_z
        << "\n  max_z: " << max_z);

    octomap::OcTree tree(resolution);
    tree.setOccupancyThres(occupied_thresh);
    tree.setProbHit(prob_hit);
    tree.setProbMiss(prob_miss);
    tree.setClampingThresMin(clamping_thres_min);
    tree.setClampingThresMax(clamping_thres_max);

    ros::Time start = ros::Time::now();
    for (int scan_id = 1; scan_id <= scan_num; ++scan_id) {
        pcl::PointCloud<pcl::PointXYZ> cloud;
        octomap::point3d origin;
        std::string filename(dir + "/" + prefix + "_" + std::to_string(scan_id) + ".pcd");
        load_pcd(filename, origin, cloud);

        pcl::PointCloud<pcl::PointXYZ> sampled;
        if (ds_resolution > 0) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>(cloud));
            pcl::VoxelGrid<pcl::PointXYZ> sor;
            sor.setInputCloud(cloud_ptr);
            sor.setLeafSize((float)ds_resolution, (float)ds_resolution, (float)ds_resolution);
            sor.filter(sampled);
        } else {
            sampled = cloud;
        }

        octomap::Pointcloud octo_cloud;
        for (const auto &pt : sampled)
            octo_cloud.push_back(pt.x, pt.y, pt.z);

        // lazy_eval=true: defer inner node updates until all scans are processed
        tree.insertPointCloud(octo_cloud, origin, max_range, true, false);
        ROS_INFO_STREAM("Scan " << scan_id << " done");
    }
    tree.updateInnerOccupancy();
    ros::Time end = ros::Time::now();
    ROS_INFO_STREAM("Mapping finished in " << (end - start).toSec() << "s");

    la3dm::MarkerArrayPub m_pub(nh, map_topic, (float)resolution);
    la3dm::MarkerArrayPub m_pub_prob(nh, map_topic_prob, (float)resolution, {"occupied", "free"});

    // Compute occupied-only bbox for free cell filtering
    float occ_min_x =  1e9f, occ_min_y =  1e9f, occ_min_z =  1e9f;
    float occ_max_x = -1e9f, occ_max_y = -1e9f, occ_max_z = -1e9f;
    for (auto it = tree.begin_leafs(); it != tree.end_leafs(); ++it) {
        if (!tree.isNodeOccupied(*it)) continue;
        if (it.getX() < occ_min_x) occ_min_x = it.getX();
        if (it.getX() > occ_max_x) occ_max_x = it.getX();
        if (it.getY() < occ_min_y) occ_min_y = it.getY();
        if (it.getY() > occ_max_y) occ_max_y = it.getY();
        if (it.getZ() < occ_min_z) occ_min_z = it.getZ();
        if (it.getZ() > occ_max_z) occ_max_z = it.getZ();
    }
    ROS_INFO_STREAM("Occupied cells bbox: ["
        << occ_min_x << ", " << occ_max_x << "] x ["
        << occ_min_y << ", " << occ_max_y << "] x ["
        << occ_min_z << ", " << occ_max_z << "]");

    if (min_z == max_z) {
        min_z = occ_min_z;
        max_z = occ_max_z;
    }
    float free_min_x = occ_min_x - (float)free_bbox_margin;
    float free_max_x = occ_max_x + (float)free_bbox_margin;
    float free_min_y = occ_min_y - (float)free_bbox_margin;
    float free_max_y = occ_max_y + (float)free_bbox_margin;
    float free_min_z = occ_min_z - (float)free_bbox_margin;
    float free_max_z = occ_max_z + (float)free_bbox_margin;
    ROS_INFO_STREAM("Free cells filter bbox (margin=" << free_bbox_margin << "): ["
        << free_min_x << ", " << free_max_x << "] x ["
        << free_min_y << ", " << free_max_y << "] x ["
        << free_min_z << ", " << free_max_z << "]");

    for (auto it = tree.begin_leafs(); it != tree.end_leafs(); ++it) {
        float prob = (float)it->getOccupancy();
        if (tree.isNodeOccupied(*it)) {
            m_pub.insert_point3d(it.getX(), it.getY(), it.getZ(),
                                 (float)min_z, (float)max_z, (float)it.getSize());
            m_pub_prob.insert_point3d_color(it.getX(), it.getY(), it.getZ(),
                                            (float)it.getSize(), prob, 0.0f, 0.0f, 1.0f, "occupied");
        } else if (it.getX() >= free_min_x && it.getX() <= free_max_x &&
                   it.getY() >= free_min_y && it.getY() <= free_max_y &&
                   it.getZ() >= free_min_z && it.getZ() <= free_max_z) {
            m_pub_prob.insert_point3d_color(it.getX(), it.getY(), it.getZ(),
                                            (float)it.getSize(), prob, 0.0f, 0.0f, 1.0f, "free");
        }
    }
    m_pub.publish();
    m_pub_prob.publish();
    ros::spin();
    return 0;
}
