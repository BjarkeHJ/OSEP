/*

Path Planner Node

*/

#include "planner_main.hpp"

#include <chrono>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

class PlannerNode : public rclcpp::Node {
public:
    PlannerNode() : Node("path_planner_node") {
        RCLCPP_INFO(this->get_logger(), "Skeleton Guided Path Planner Node Constructed");
    }

    void init();
    void pcd_callback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud);
    void vertex_callback(const sensor_msgs::msg::PointCloud2::SharedPtr vertex_msg);
    void run();

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr vertex_sub_;
    rclcpp::TimerBase::SharedPtr run_timer_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

private:
    bool run_flag = false;
    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices;
};

void PlannerNode::init() {
    RCLCPP_INFO(this->get_logger(), "Initializing Modules and Data Structures...");
    pcd_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/lidar_scan", 10, std::bind(&PlannerNode::pcd_callback, this, std::placeholders::_1));
    vertex_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/local_vertices", 10, std::bind(&PlannerNode::vertex_callback, this, std::placeholders::_1));

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    vertices.reset(new pcl::PointCloud<pcl::PointXYZ>);
}

void PlannerNode::vertex_callback(const sensor_msgs::msg::PointCloud2::SharedPtr vertex_msg) {
    if (vertex_msg->data.empty()) {
        RCLCPP_INFO(this->get_logger(), "Received empty vertex set");
        return;
    }

    pcl::fromROSMsg(*vertex_msg, *vertices);
    run_flag = true;
}

void PlannerNode::run() {
    if (run_flag) {
        run_flag = false;

    }
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlannerNode>();
    node->init(); // Initialize Modules etc...

    // Spin the node
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}