#ifndef VIS_TOOLS_HPP_
#define VIS_TOOLS_HPP_

#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <pcl/common/common.h>
#include <pcl_conversions/pcl_conversions.h>

class VisTools {
public:
    VisTools(rclcpp::Node::SharedPtr node);

    void publishPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::string &frame_id);
    void publishNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, std::string &frame_id, double scale);

    void publish_cloud_debug(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::string &frame_id);

    void publish_clusters(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::string &frame_id);

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr nrm_pub_;
    
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_pub_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cluster_pub_;

    
};  


#endif //VIS_TOOLS_HPP_