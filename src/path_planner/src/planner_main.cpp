/* 

Main Path Skeleton Guided Viewpoint Generation and Path Planning

This file contains the incremental point cloud incrementation

*/

#include "planner_main.hpp"

PathPlanner::PathPlanner(rclcpp::Node::SharedPtr node) : node_(node)
{
}

void PathPlanner::init() {
    RCLCPP_INFO(node_->get_logger(), "Initializing Moduel: Online Skeleton Guided Path Planner");
    /* Param */
    // Stuff from launch file (ToDo)...

    /* Modules */

    /* Data */
    GS.global_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
    GS.global_vertices.reset(new pcl::PointCloud<pcl::PointXYZ>);
    local_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
    local_vertices.reset(new pcl::PointCloud<pcl::PointXYZ>);
}

void PathPlanner::main() {
    auto t_start = std::chrono::high_resolution_clock::now();

    update_skeleton();

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_elapsed = t_end - t_start;
    RCLCPP_INFO(node_->get_logger(), "Time Elapsed: %f seconds", t_elapsed.count());
}

void PathPlanner::update_skeleton() {
    
}

