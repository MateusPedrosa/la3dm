#include <pcl_ros/point_cloud.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/ColorRGBA.h>

#include <cmath>
#include <string>

namespace la3dm {
    
    inline std_msgs::ColorRGBA heightMapColor(double h) {

        std_msgs::ColorRGBA color;
        color.a = 1.0;
        // blend over HSV-values (more colors)

        double s = 1.0;
        double v = 1.0;

        h -= floor(h);
        h *= 6;
        int i;
        double m, n, f;

        i = floor(h);
        f = h - i;
        if (!(i & 1))
            f = 1 - f; // if i is even
        m = v * (1 - s);
        n = v * (1 - s * f);

        switch (i) {
            case 6:
            case 0:
                color.r = v;
                color.g = n;
                color.b = m;
                break;
            case 1:
                color.r = n;
                color.g = v;
                color.b = m;
                break;
            case 2:
                color.r = m;
                color.g = v;
                color.b = n;
                break;
            case 3:
                color.r = m;
                color.g = n;
                color.b = v;
                break;
            case 4:
                color.r = n;
                color.g = m;
                color.b = v;
                break;
            case 5:
                color.r = v;
                color.g = m;
                color.b = n;
                break;
            default:
                color.r = 1;
                color.g = 0.5;
                color.b = 0.5;
                break;
        }

        return color;
    }

    class MarkerArrayPub {
        typedef pcl::PointXYZ PointType;
        typedef pcl::PointCloud<PointType> PointCloud;
    public:
        MarkerArrayPub(ros::NodeHandle nh, std::string topic, float resolution, std::vector<std::string> namespaces = {"map"}, std::string frame_id = "map") : nh(nh),
                                                                                  msg(new visualization_msgs::MarkerArray),
                                                                                  topic(topic),
                                                                                  resolution(resolution),
                                                                                  markerarray_frame_id(frame_id) {
            pub = nh.advertise<visualization_msgs::MarkerArray>(topic, 1, true);

            msg->markers.resize(10 * namespaces.size());
            for(size_t n = 0; n < namespaces.size(); ++n) {
                for (int i = 0; i < 10; ++i) {
                    int idx = n * 10 + i;
                    msg->markers[idx].header.frame_id = markerarray_frame_id;
                    msg->markers[idx].ns = namespaces[n];
                    msg->markers[idx].id = i;
                    msg->markers[idx].type = visualization_msgs::Marker::CUBE_LIST;
                    msg->markers[idx].scale.x = resolution * pow(2, i);
                    msg->markers[idx].scale.y = resolution * pow(2, i);
                    msg->markers[idx].scale.z = resolution * pow(2, i);
                    std_msgs::ColorRGBA color;
                    color.r = 0.0;
                    color.g = 0.0;
                    color.b = 1.0;
                    color.a = 1.0;
                    msg->markers[idx].color = color;
                }
            }
        }

        int get_ns_offset(const std::string& ns) const {
            for (size_t i = 0; i < msg->markers.size(); i += 10) {
                if (msg->markers[i].ns == ns) return i;
            }
            return 0;
        }

        void insert_point3d(float x, float y, float z, float min_z, float max_z, float size, std::string ns = "map") {
            geometry_msgs::Point center;
            center.x = x;
            center.y = y;
            center.z = z;

            int depth = 0;
            if (size > 0)
                depth = (int) log2(size /resolution);
            depth += get_ns_offset(ns);

            msg->markers[depth].points.push_back(center);

            if (min_z < max_z) {
                double h = (1.0 - std::min(std::max((z - min_z) / (max_z - min_z), 0.0f), 1.0f)) * 0.8;
                msg->markers[depth].colors.push_back(heightMapColor(h));
            }
        }

        void insert_point3d(float x, float y, float z, float min_z, float max_z, float size, float prob, std::string ns = "map") {
            geometry_msgs::Point center;
            center.x = x;
            center.y = y;
            center.z = z;

            int depth = 0;
            if (size > 0)
                depth = (int) log2(size / resolution);
            depth += get_ns_offset(ns);

            msg->markers[depth].points.push_back(center);

            std_msgs::ColorRGBA color;
            color.a = 1.0;

            if(prob < 0.5){
                color.r = 0.8; 
                color.g = 0.8; 
                color.b = 0.8; 
            }
            else{
                color = heightMapColor(std::min(2.0-2.0*prob, 0.6));
            }

            msg->markers[depth].colors.push_back(color);
        }

        void insert_point3d(float x, float y, float z, float min_z, float max_z, std::string ns = "map") {
            insert_point3d(x, y, z, min_z, max_z, -1.0f, ns);
        }

        void insert_point3d(float x, float y, float z, std::string ns = "map") {
            insert_point3d(x, y, z, 1.0f, 0.0f, -1.0f, ns);
        }

        void insert_color_point3d(float x, float y, float z, double min_v, double max_v, double v, float size = -1.0f, std::string ns = "map") {
            geometry_msgs::Point center;
            center.x = x;
            center.y = y;
            center.z = z;

            int depth = 0;
            if (size > 0)
                depth = (int) log2(size / resolution);
            depth += get_ns_offset(ns);

            msg->markers[depth].points.push_back(center);

            double h = (1.0 - std::min(std::max((v - min_v) / (max_v - min_v), 0.0), 1.0)) * 0.8;
            msg->markers[depth].colors.push_back(heightMapColor(h));
        }

        void insert_point3d_color(float x, float y, float z, float size, float r, float g, float b, float a = 1.0f, std::string ns = "map") {
            geometry_msgs::Point center;
            center.x = x;
            center.y = y;
            center.z = z;

            int depth = 0;
            if (size > 0)
                depth = (int) log2(size / resolution);
            depth += get_ns_offset(ns);

            msg->markers[depth].points.push_back(center);

            std_msgs::ColorRGBA color;
            color.r = r;
            color.g = g;
            color.b = b;
            color.a = a;
            msg->markers[depth].colors.push_back(color);
        }

        void clear() {
            for (size_t i = 0; i < msg->markers.size(); ++i) {
                msg->markers[i].points.clear();
                msg->markers[i].colors.clear();
            }
        }

        void publish() const {
            for (size_t i = 0; i < msg->markers.size(); ++i) {
                msg->markers[i].header.stamp = ros::Time::now();
            }
            pub.publish(*msg);
        }

    private:
        ros::NodeHandle nh;
        ros::Publisher pub;
        visualization_msgs::MarkerArray::Ptr msg;
        std::string markerarray_frame_id;
        std::string topic;
        float resolution;
    };

    class TextMarkerArrayPub {
    public:
        TextMarkerArrayPub(ros::NodeHandle nh, std::string topic, float resolution, std::string frame_id = "map") : nh(nh),
                                                                                  msg(new visualization_msgs::MarkerArray),
                                                                                  topic(topic),
                                                                                  resolution(resolution),
                                                                                  marker_frame_id(frame_id),
                                                                                  marker_id(0) {
            pub = nh.advertise<visualization_msgs::MarkerArray>(topic, 1, true);
        }

        void insert_text3d(float x, float y, float z, const std::string& text, float size) {
            visualization_msgs::Marker marker;
            marker.header.frame_id = marker_frame_id;
            marker.ns = "text";
            marker.id = marker_id++;
            marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.position.x = x;
            marker.pose.position.y = y;
            marker.pose.position.z = z;
            marker.pose.orientation.w = 1.0;
            marker.scale.z = size * 0.2; // Text height
            marker.color.r = 1.0;
            marker.color.g = 1.0;
            marker.color.b = 1.0;
            marker.color.a = 1.0;
            marker.text = text;
            msg->markers.push_back(marker);
        }

        void clear() {
            msg->markers.clear();
            marker_id = 0;
        }

        void publish() const {
            if (msg->markers.empty()) return;
            for (auto& m : msg->markers) {
                m.header.stamp = ros::Time::now();
            }
            pub.publish(*msg);
        }

    private:
        ros::NodeHandle nh;
        ros::Publisher pub;
        visualization_msgs::MarkerArray::Ptr msg;
        std::string topic;
        std::string marker_frame_id;
        float resolution;
        int marker_id;
    };

}
