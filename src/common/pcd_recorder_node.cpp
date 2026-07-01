#include <string>
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <boost/filesystem.hpp>

std::string dir;
std::string prefix;
std::string cloud_topic("/sonar_point_cloud");
std::string fixed_frame("map");
std::string sensor_frame("sonar_link");
double record_rate = 2.0;
bool save_all = false;

std::queue<sensor_msgs::PointCloud2ConstPtr> msg_queue;
std::mutex queue_mutex;
std::condition_variable queue_cv;
bool shutdown_requested = false;

// Callback returns immediately — no blocking work on the ROS spin thread.
void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        msg_queue.push(msg);
    }
    queue_cv.notify_one();
}

void processingThread() {
    tf::TransformListener listener;

    int scan_id = 1;
    ros::Time last_saved(0);

    while (true) {
        sensor_msgs::PointCloud2ConstPtr msg;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [] { return !msg_queue.empty() || shutdown_requested; });
            if (msg_queue.empty())
                break;
            msg = msg_queue.front();
            msg_queue.pop();
        }

        if (!save_all && scan_id > 1 && (msg->header.stamp - last_saved) < ros::Duration(1.0 / record_rate))
            continue;

        tf::StampedTransform transform;
        try {
            listener.waitForTransform(fixed_frame, sensor_frame, msg->header.stamp, ros::Duration(1.0));
            listener.lookupTransform(fixed_frame, sensor_frame, msg->header.stamp, transform);
        } catch (tf::TransformException &ex) {
            ROS_WARN_STREAM("TF lookup failed for scan " << scan_id << ": " << ex.what());
            continue;
        }

        // Transform cloud from sensor frame into the fixed/map frame so that
        // insert_pointcloud receives world-frame hit points (same convention as
        // the _server nodes and the existing demo PCD files).
        sensor_msgs::PointCloud2 cloud_map_msg;
        try {
            pcl_ros::transformPointCloud(fixed_frame, *msg, cloud_map_msg, listener);
        } catch (tf::TransformException &ex) {
            ROS_WARN_STREAM("Cloud transform failed for scan " << scan_id << ": " << ex.what());
            continue;
        }

        pcl::PointCloud<pcl::PointXYZI> cloud;
        pcl::fromROSMsg(cloud_map_msg, cloud);

        if (cloud.empty()) {
            ROS_WARN_STREAM("Scan " << scan_id << " is empty, skipping");
            continue;
        }

        tf::Vector3 t = transform.getOrigin();
        tf::Quaternion q = transform.getRotation();
        cloud.sensor_origin_ = Eigen::Vector4f(t.x(), t.y(), t.z(), 1.0f);
        cloud.sensor_orientation_ = Eigen::Quaternionf(q.w(), q.x(), q.y(), q.z());

        std::string filename = dir + "/" + prefix + "_" + std::to_string(scan_id) + ".pcd";
        if (pcl::io::savePCDFileBinary(filename, cloud) < 0) {
            ROS_ERROR_STREAM("Failed to save " << filename);
            continue;
        }

        ROS_INFO_STREAM("Saved scan " << scan_id << " -> " << filename
                        << " (" << cloud.size() << " pts)"
                        << " origin [" << t.x() << ", " << t.y() << ", " << t.z() << "]"
                        << " (queue depth: " << msg_queue.size() << ")");
        last_saved = msg->header.stamp;
        ++scan_id;
    }

    ROS_INFO_STREAM("PCD Recorder finished. Saved " << (scan_id - 1) << " scans to " << dir
                    << ". Set scan_num: " << (scan_id - 1) << " in your dataset YAML.");
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "pcd_recorder");
    ros::NodeHandle nh("~");

    nh.param<std::string>("dir", dir, ".");
    nh.param<std::string>("prefix", prefix, "scan");
    nh.param<std::string>("cloud_topic", cloud_topic, cloud_topic);
    nh.param<std::string>("fixed_frame", fixed_frame, fixed_frame);
    nh.param<std::string>("sensor_frame", sensor_frame, sensor_frame);
    nh.param<double>("record_rate", record_rate, record_rate);
    nh.param<bool>("save_all", save_all, save_all);

    boost::filesystem::create_directories(dir);

    ROS_INFO_STREAM("PCD Recorder:"
        << "\n  dir:          " << dir
        << "\n  prefix:       " << prefix
        << "\n  cloud_topic:  " << cloud_topic
        << "\n  fixed_frame:  " << fixed_frame
        << "\n  sensor_frame: " << sensor_frame
        << "\n  save_all:     " << (save_all ? "true (every cloud)" : "false")
        << "\n  record_rate:  " << (save_all ? "N/A" : std::to_string(record_rate) + " Hz"));

    std::thread worker(processingThread);

    // Large queue so ROS never drops messages before the worker can pick them up.
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>(cloud_topic, 1000, cloudCallback);

    ros::spin();

    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        shutdown_requested = true;
    }
    queue_cv.notify_one();
    worker.join();

    return 0;
}
