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
    if (!local_vertices || local_vertices->empty()) {
        RCLCPP_INFO(node_->get_logger(), "No New Vertices...");
        return;
    }

    RCLCPP_INFO(node_->get_logger(), "Updating Global Skeleton...");

    for (const auto& lver : local_vertices->points) {
        std::vector<int> nearest_idx(1);
        std::vector<float> nearest_dist(1);

        // If the nearest neighbor is within threshold distance -> fuse vertices
        if (gskel_tree.nearestKSearch(lver, 1, nearest_idx, nearest_dist) > 0 && nearest_dist[0] < fuse_dist_th * fuse_dist_th) {
            int global_idx = nearest_idx[0];
            lowpass_update(global_idx, lver);
        }
        else {
            add_new_vertex(lver);    
        }
    }
}

void PathPlanner::lowpass_update(int idx, const pcl::PointXYZ& new_pt) {
    pcl::PointXYZ& old_pt = GS.global_vertices->points[idx];
    old_pt.x = (1.0 - fuse_alpha) * old_pt.x + fuse_alpha * new_pt.x;
    old_pt.y = (1.0 - fuse_alpha) * old_pt.y + fuse_alpha * new_pt.y;
    old_pt.z = (1.0 - fuse_alpha) * old_pt.z + fuse_alpha * new_pt.z;
}

void PathPlanner::add_new_vertex(const pcl::PointXYZ& pt) {
    GS.global_vertices->points.push_back(pt);
    gskel_tree.setInputCloud(GS.global_vertices);
}
