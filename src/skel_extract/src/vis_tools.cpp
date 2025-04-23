/*

For Visualization

*/

#include "vis_tools.hpp"

VisTools::VisTools(rclcpp::Node::SharedPtr node) : node_(node) {
    pcd_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/pointcloud_repub", 10);
    nrm_pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>("/surface_normals", 10);
    debug_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/debugger_01", 10);
}


void VisTools::publishPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::string &frame_id) {
    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.frame_id = frame_id;
    cloud_msg.header.stamp = node_->now();
    pcd_pub_->publish(cloud_msg);
}

void VisTools::publishNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, std::string &frame_id, double scale) {
    visualization_msgs::msg::MarkerArray marker_array;
    marker_array.markers.reserve(normals->size());

    for (size_t i=0; i<normals->size(); ++i) {
        const auto& pt = cloud->points[i];
        const auto& n = normals->points[i];

        visualization_msgs::msg::Marker arrow;
        arrow.header.frame_id = frame_id;
        arrow.header.stamp = node_->now();
        arrow.ns = "normals";
        arrow.id = i;
        arrow.type = visualization_msgs::msg::Marker::ARROW;
        arrow.action = visualization_msgs::msg::Marker::ADD;

        geometry_msgs::msg::Point start, end;
        start.x = pt.x;
        start.y = pt.y;
        start.z = pt.z;
        end.x = pt.x + n.normal_x * scale;
        end.y = pt.y + n.normal_y * scale;
        end.z = pt.z + n.normal_z * scale;

        arrow.points.push_back(start);
        arrow.points.push_back(end);

        arrow.scale.x = 0.005;  // shaft diameter
        arrow.scale.y = 0.01;   // head diameter
        arrow.scale.z = 0.01;   // head length

        arrow.color.r = 0.0f;
        arrow.color.g = 0.0f;
        arrow.color.b = 1.0f;
        arrow.color.a = 1.0f;

        marker_array.markers.push_back(arrow);
    }
    nrm_pub_->publish(marker_array);
}


void VisTools::publish_cloud_debug(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::string &frame_id) {
    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.frame_id = frame_id;
    cloud_msg.header.stamp = node_->now();
    debug_pub_->publish(cloud_msg);
}
