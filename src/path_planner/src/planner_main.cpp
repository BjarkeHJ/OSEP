/* 

Main Path Skeleton Guided Viewpoint Generation and Path Planning

This file contains the incremental point cloud incrementation

*/

#include "planner_main.hpp"

PathPlanner::PathPlanner(rclcpp::Node::SharedPtr node) : node_(node)
{
}

void PathPlanner::init() {
    RCLCPP_INFO(node_->get_logger(), "Initializing Module: Online Skeleton Guided Path Planner");
    /* Param */
    // Stuff from launch file (ToDo)...

    /* Modules */

    /* Data */
    GS.global_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
    GS.global_vertices_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    local_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
    local_vertices.reset(new pcl::PointCloud<pcl::PointXYZ>);
}

void PathPlanner::main() {
    auto t_start = std::chrono::high_resolution_clock::now();
    update_skeleton();
    graph_adj();
    mst();
    clean_skeleton_graph();

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

    for (auto &pt : local_vertices->points) {
        Eigen::Vector3d ver(pt.x, pt.y, pt.z);
        bool matched = false;
        for (auto &gver : GS.global_vertices) {
            double dist = (gver.position - ver).norm();
            if (dist < fuse_dist_th) {
                VertexLKF kf(kf_pn, kf_mn);
                kf.initialize(gver.position, gver.covariance);
                kf.update(ver);

                gver.position = kf.getState();
                gver.covariance = kf.getCovariance();
                gver.obs_count++;
                double trace = gver.covariance.trace();
                if (trace < fuse_conf_th) {
                    gver.conf_check = true;
                    gver.unconfirmed_check = 0;
                }
                else {
                    gver.unconfirmed_check++;
                }
                
                if (gver.unconfirmed_check > max_obs_wo_conf) {
                    continue;
                }

                matched = true;
                break;
            }
        }

        if (!matched) {
            SkeletonVertex new_ver;
            new_ver.position = ver;
            new_ver.covariance = Eigen::Matrix3d::Identity();
            new_ver.obs_count = 1;
            new_ver.conf_check = false;
            GS.global_vertices.push_back(new_ver);
        }
    }

    GS.global_vertices_cloud->clear();
    for (auto &gver : GS.global_vertices) {
        if (gver.conf_check) {
            pcl::PointXYZ pt(gver.position(0), gver.position(1), gver.position(2));
            GS.global_vertices_cloud->points.push_back(pt);
        }
    }
}

void PathPlanner::graph_adj() {
    if (!GS.global_vertices_cloud || GS.global_vertices_cloud->empty()) {
        RCLCPP_WARN(node_->get_logger(), "No points in global skeleton. Skipping graph adjacency rebuild.");
        return;
    }

    // Create a new adjacency list
    std::vector<std::vector<int>> new_adj(GS.global_vertices_cloud->size());

    pcl::KdTreeFLANN<pcl::PointXYZ> adj_tree;
    adj_tree.setInputCloud(GS.global_vertices_cloud);

    const int K = 5;          // Number of neighbors
    const float max_dist_th = 2.0 * fuse_dist_th; // Max distance for valid edges (meters)
    const float min_dist_th = 0.5 * fuse_dist_th;

    for (size_t i = 0; i < GS.global_vertices_cloud->size(); ++i) {
        std::vector<int> indices;
        std::vector<float> distances;

        int n_neighbors = adj_tree.nearestKSearch(GS.global_vertices_cloud->points[i], K, indices, distances);

        for (int j = 1; j < n_neighbors; ++j) { // Skip self (index 0)
            int nb_idx = indices[j];
            float dist_to_nb = (GS.global_vertices_cloud->points[i].getVector3fMap() - GS.global_vertices_cloud->points[nb_idx].getVector3fMap()).norm();

            if (dist_to_nb > max_dist_th || dist_to_nb < min_dist_th) continue; // Too far or too close, skip

            bool is_good_neighbor = true;

            // Small geometric consistency check
            for (int k = 1; k < n_neighbors; ++k) {
                if (k == j) continue;

                int other_nb_idx = indices[k];
                float dist_nb_to_other = (GS.global_vertices_cloud->points[nb_idx].getVector3fMap() - GS.global_vertices_cloud->points[other_nb_idx].getVector3fMap()).norm();
                float dist_to_other = (GS.global_vertices_cloud->points[i].getVector3fMap() - GS.global_vertices_cloud->points[other_nb_idx].getVector3fMap()).norm();

                if (dist_nb_to_other < dist_to_nb && dist_to_other < dist_to_nb) {
                    is_good_neighbor = false;
                    break;
                }
            }

            if (is_good_neighbor) {
                new_adj[i].push_back(nb_idx);
                new_adj[nb_idx].push_back(i);
            }
        }
    }

    GS.global_adj = new_adj; // Replace old adjacency
}

void PathPlanner::mst() {
    int N_ver = GS.global_vertices_cloud->size();
    if (N_ver == 0 || GS.global_adj.empty()) {
        RCLCPP_WARN(node_->get_logger(), "Global skeleton is empty, cannot extract MST.");
        return;
    }

    RCLCPP_INFO(node_->get_logger(), "Extracting MST..  .");

    std::vector<Edge> mst_edges;
    for (int i = 0; i < N_ver; ++i) {
        for (int nb : GS.global_adj[i]) {
            if (nb <= i) continue;
            Eigen::Vector3f ver_i = GS.global_vertices_cloud->points[i].getVector3fMap();
            Eigen::Vector3f ver_nb = GS.global_vertices_cloud->points[nb].getVector3fMap();
            double weight = (ver_i - ver_nb).norm();
            mst_edges.push_back({i, nb, weight});
        }
    }

    std::sort(mst_edges.begin(), mst_edges.end());

    UnionFind uf(N_ver);
    std::vector<std::vector<int>> mst_adj(N_ver);

    for (const auto& edge : mst_edges) {
        if (uf.unite(edge.u, edge.v)) {
            mst_adj[edge.u].push_back(edge.v);
            mst_adj[edge.v].push_back(edge.u);
        }
    }

    GS.global_adj = std::move(mst_adj);

    RCLCPP_INFO(node_->get_logger(), "MST extraction complete. Tree edges: %d", N_ver - 1);
}

void PathPlanner::clean_skeleton_graph() {
    if (GS.global_vertices_cloud->empty() || GS.global_adj.empty()) {
        RCLCPP_WARN(node_->get_logger(), "Cannot clean an empty graph.");
        return;
    }

    const int min_branch_nodes = 3; // Minimum number of nodes required to keep a branch
    const int smooth_iter = 1;
    RCLCPP_INFO(node_->get_logger(), "Cleaning skeleton: pruning small branches based on node count and smoothing...");

    // Step 1: Prune small branches (based on number of nodes)
    bool pruned = true;
    while (pruned) {
        pruned = false;
        std::vector<bool> visited(GS.global_vertices_cloud->size(), false);

        for (size_t i = 0; i < GS.global_vertices_cloud->size(); ++i) {
            if (!visited[i] && GS.global_adj[i].size() == 1) { // Leaf node
                std::vector<int> branch;
                int current = i;
                int prev = -1;

                // Walk along the branch
                while (true) {
                    visited[current] = true;
                    branch.push_back(current);

                    if (GS.global_adj[current].empty()) {
                        break;
                    }

                    int next = -1;
                    for (int nb : GS.global_adj[current]) {
                        if (nb != prev) {
                            next = nb;
                            break;
                        }
                    }
                    if (next == -1 || GS.global_adj[current].size() > 2) {
                        break; // Reached junction or end
                    }

                    prev = current;
                    current = next;
                }

                if ((int)branch.size() < min_branch_nodes) {
                    // Prune the entire small branch
                    for (int idx : branch) {
                        for (int nb : GS.global_adj[idx]) {
                            GS.global_adj[nb].erase(std::remove(GS.global_adj[nb].begin(), GS.global_adj[nb].end(), idx), GS.global_adj[nb].end());
                        }
                        GS.global_adj[idx].clear();
                    }
                    pruned = true;
                }
            }
        }
    }

    // Step 2: Optional weighted smoothing
    for (int iter = 0; iter < smooth_iter; ++iter) {
        std::vector<Eigen::Vector3f> new_positions(GS.global_vertices_cloud->size());

        for (size_t i = 0; i < GS.global_vertices_cloud->size(); ++i) {
            if (GS.global_adj[i].empty()) {
                new_positions[i] = GS.global_vertices_cloud->points[i].getVector3fMap();
                continue;
            }

            Eigen::Vector3f avg_pos = GS.global_vertices_cloud->points[i].getVector3fMap();
            float total_weight = 1.0f;

            for (int nb : GS.global_adj[i]) {
                Eigen::Vector3f diff = GS.global_vertices_cloud->points[nb].getVector3fMap() - GS.global_vertices_cloud->points[i].getVector3fMap();
                float dist = diff.norm();
                float weight = std::exp(-dist);
                avg_pos += weight * GS.global_vertices_cloud->points[nb].getVector3fMap();
                total_weight += weight;
            }
            avg_pos /= total_weight;

            new_positions[i] = avg_pos;
        }

        // Update points
        for (size_t i = 0; i < GS.global_vertices_cloud->size(); ++i) {
            GS.global_vertices_cloud->points[i].x = new_positions[i].x();
            GS.global_vertices_cloud->points[i].y = new_positions[i].y();
            GS.global_vertices_cloud->points[i].z = new_positions[i].z();
        }
    }

    RCLCPP_INFO(node_->get_logger(), "Skeleton cleaning complete.");
}

void PathPlanner::select_waypoint() {


}
