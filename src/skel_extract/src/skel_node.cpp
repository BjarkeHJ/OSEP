/* 

ROS2 node 

*/

#include "skel_main.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>

class OsepNode : public rclcpp::Node {
public:
    OsepNode() : Node("OSEP_Node") {
        RCLCPP_INFO(this->get_logger(), "Node Constructed");
    }

    void init();
    void pcd_callback(const sensor_msgs::msg::PointCloud2::SharedPtr pcd_msg);

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_sub_;

private:
    /* Params */

    /* Data */
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_;

    /* Utils */
    std::shared_ptr<OSEP> osep;

};


void OsepNode::init() {
    RCLCPP_INFO(this->get_logger(), "Initializing Modules and Data Structures...");
    /* Params */
    // Stuff from launch file (ToDo)...

    /* Data */
    pcd_.reset(new pcl::PointCloud<pcl::PointXYZ>);

    /* Modules */
    osep = std::make_shared<OSEP>(shared_from_this());
    osep->init();
}

void OsepNode::pcd_callback(const sensor_msgs::msg::PointCloud2::SharedPtr pcd_msg) {
    pcl::fromROSMsg(*pcd_msg, *pcd_);
    osep->SS.pts_ = pcd_;
}


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OsepNode>();
    node->init(); // Initialize Modules etc...

    // Spin the node
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}