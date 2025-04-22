/* 

Main skeleton extraction algorithm 

*/

#include <skel_main.hpp>

OSEP::OSEP(rclcpp::Node::SharedPtr node) : node_(node) 
{

}

void OSEP::init() {
    RCLCPP_INFO(node_->get_logger(), "Initializing Module: Online Skeleton Extraction Planner");
    /* Params */
    // Stuff from launch file (Todo)...

    /* Modules */
    // Vis tools maybe?


    /* Data */
    SS.pts_.reset(new pcl::PointCloud<pcl::PointXYZ>);
}

void OSEP::main() {
    RCLCPP_INFO(node_->get_logger(), "Skeleton Extraction Main Algorithm...");
    

}