/* 

ROS2 node 

*/

#include "skel_main.hpp"
#include <chrono>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>

class SkeletonExtractionNode : public rclcpp::Node {
public:
SkeletonExtractionNode() : Node("SkeletonExtractionNode") {
        RCLCPP_INFO(this->get_logger(), "SkeletonExtractionNode Constructed");
    }

    void init();
    void pcd_callback(const sensor_msgs::msg::PointCloud2::SharedPtr pcd_msg);
    void run();

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_sub_;
    rclcpp::TimerBase::SharedPtr run_timer_;

private:
    /* Utils */
    std::shared_ptr<SkelEx> skel_ex;

    /* Params */
    bool run_flag = false;
    int run_timer_ms = 50;

    /* Data */
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_;


};


void SkeletonExtractionNode::init() {
    RCLCPP_INFO(this->get_logger(), "Initializing Modules and Data Structures...");
    /* Subscriber, Publishers, Timers, etc... */
    pcd_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/pointcloud_data", 10, std::bind(&SkeletonExtractionNode::pcd_callback, this, std::placeholders::_1));
    run_timer_ = this->create_wall_timer(std::chrono::milliseconds(run_timer_ms), std::bind(&SkeletonExtractionNode::run, this));

    /* Params */
    // Stuff from launch file (ToDo)...

    /* Data */
    pcd_.reset(new pcl::PointCloud<pcl::PointXYZ>);

    /* Modules */
    skel_ex = std::make_shared<SkelEx>(shared_from_this());
    skel_ex->init();
}

void SkeletonExtractionNode::pcd_callback(const sensor_msgs::msg::PointCloud2::SharedPtr pcd_msg) {
    if (pcd_msg->data.empty()) {
        RCLCPP_INFO(this->get_logger(), "Received empty point cloud...");
        return;
    }
    pcl::fromROSMsg(*pcd_msg, *pcd_);
    skel_ex->SS.pts_ = pcd_;
    run_flag = true;
}

void SkeletonExtractionNode::run() {
    if (run_flag) {
        run_flag = false;
        skel_ex->main();
    }
}


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SkeletonExtractionNode>();
    node->init(); // Initialize Modules etc...

    // Spin the node
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}