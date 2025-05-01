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

    GP.global_waypoints.reset(new pcl::PointCloud<pcl::PointXYZ>);
    GP.current_waypoints.reset(new pcl::PointCloud<pcl::PointXYZ>);
}

void PathPlanner::plan_path() {
    auto t_start = std::chrono::high_resolution_clock::now();
    
    // RUN GENERAL PATH PLANNING SELECTION ECT...
    select_waypoint();
    
    // update_skeleton();
    // graph_adj();
    // mst();
    // clean_skeleton_graph();

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_elapsed = t_end - t_start;
    RCLCPP_INFO(node_->get_logger(), "[Path Planning] Time Elapsed: %f seconds", t_elapsed.count());
}

void PathPlanner::update_skeleton() {
    auto t_start = std::chrono::high_resolution_clock::now();

    skeleton_increment();
    graph_adj();
    mst();
    vertex_merge();
    prune_branches();

    // clean_skeleton_graph();

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_elapsed = t_end - t_start;
    RCLCPP_INFO(node_->get_logger(), "[Skeleton Update] Time Elapsed: %f seconds", t_elapsed.count());
}


void PathPlanner::skeleton_increment() {
    if (!local_vertices || local_vertices->empty()) {
        RCLCPP_INFO(node_->get_logger(), "No New Vertices...");
        return;
    }

    RCLCPP_INFO(node_->get_logger(), "Updating Global Skeleton...");

    for (auto &pt : local_vertices->points) {
        Eigen::Vector3d ver(pt.x, pt.y, pt.z);
        bool matched = false;
        for (auto &gver : GS.prelim_vertices) {
            double sq_dist = (gver.position - ver).squaredNorm();
            if (sq_dist < fuse_dist_th*fuse_dist_th) {
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
                

                // This logic does not hold!!! It will continously occupy the space but never insert a vertice!....
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
            GS.prelim_vertices.push_back(new_ver);
        }
    }

    // Select only confident vertices
    GS.global_vertices_cloud->clear();
    GS.global_vertices.clear();
    for (auto &gver : GS.prelim_vertices) {
        if (gver.conf_check) {
            pcl::PointXYZ pt(gver.position(0), gver.position(1), gver.position(2));
            GS.global_vertices_cloud->points.push_back(pt);
            GS.global_vertices.push_back(gver);
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
    // const float min_dist_th = 0.5 * fuse_dist_th;

    for (size_t i = 0; i < GS.global_vertices_cloud->size(); ++i) {
        std::vector<int> indices;
        std::vector<float> distances;

        int n_neighbors = adj_tree.nearestKSearch(GS.global_vertices_cloud->points[i], K, indices, distances);

        for (int j = 1; j < n_neighbors; ++j) { // Skip self (index 0)
            int nb_idx = indices[j];
            float dist_to_nb = (GS.global_vertices_cloud->points[i].getVector3fMap() - GS.global_vertices_cloud->points[nb_idx].getVector3fMap()).norm();

            // if (dist_to_nb > max_dist_th || dist_to_nb < min_dist_th) continue; // Too far or too close, skip
            if (dist_to_nb > max_dist_th) continue; // Too far or too close, skip

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
    int N_ver = GS.global_vertices.size();
    if (N_ver == 0 || GS.global_adj.empty()) {
        RCLCPP_WARN(node_->get_logger(), "Global skeleton is empty, cannot extract MST.");
        return;
    }

    RCLCPP_INFO(node_->get_logger(), "Extracting MST..  .");

    std::vector<Edge> mst_edges;
    for (int i = 0; i < N_ver; ++i) {
        for (int nb : GS.global_adj[i]) {
            if (nb <= i) continue; // Avoid bi-directional check
            Eigen::Vector3d ver_i = GS.global_vertices[i].position;
            Eigen::Vector3d ver_nb = GS.global_vertices[nb].position;
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

    // Identify leafs and joints
    graph_decomp();

    RCLCPP_INFO(node_->get_logger(), "MST extraction complete. Tree edges: %d", N_ver - 1);
}

void PathPlanner::vertex_merge() {
    auto is_joint = [&](int idx) {
        return std::find(GS.joints.begin(), GS.joints.end(), idx) != GS.joints.end();
    };

    std::set<int> to_delete;
    std::vector<std::pair<int, int>> merge_pairs;  // store (i, j) where j is to be merged into i

    for (int i = 0; i < (int)GS.global_vertices.size(); ++i) {
        if (to_delete.count(i)) continue;
        for (int j = i + 1; j < (int)GS.global_vertices.size(); ++j) {
            if (to_delete.count(j)) continue;

            // If not adjacent, skip
            if (std::find(GS.global_adj[i].begin(), GS.global_adj[i].end(), j) == GS.global_adj[i].end()) continue;

            bool merge = false;

            // Case 1: Both are joints
            if (is_joint(i) && is_joint(j)) {
                merge = true;
                GS.joints.erase(std::remove(GS.joints.begin(), GS.joints.end(), j), GS.joints.end());
            }

            // Case 2: Too close
            double dist = (GS.global_vertices[i].position - GS.global_vertices[j].position).norm();
            if (!merge && dist < 0.5*fuse_dist_th) {
                merge = true;
            }

            if (merge) {
                // Merge j into i
                int obs_i = GS.global_vertices[i].obs_count;
                int obs_j = GS.global_vertices[j].obs_count;
                int total_obs = obs_i + obs_j;

                GS.global_vertices[i].position =
                    (GS.global_vertices[i].position * obs_i + GS.global_vertices[j].position * obs_j) / total_obs;
                GS.global_vertices[i].obs_count = total_obs;

                // Merge neighbors of j into i
                for (int neighbor : GS.global_adj[j]) {
                    if (neighbor != i &&
                        std::find(GS.global_adj[i].begin(), GS.global_adj[i].end(), neighbor) == GS.global_adj[i].end()) {
                        GS.global_adj[i].push_back(neighbor);
                    }
                    auto& nbrs = GS.global_adj[neighbor];
                    nbrs.erase(std::remove(nbrs.begin(), nbrs.end(), j), nbrs.end());
                    if (neighbor != i && std::find(nbrs.begin(), nbrs.end(), i) == nbrs.end()) {
                        nbrs.push_back(i);
                    }
                }

                to_delete.insert(j);
            }
        }
    }

    // === Remove merged vertices and fix indices ===
    std::vector<int> index_map(GS.global_vertices.size(), -1);
    int new_index = 0;
    for (int old_index = 0; old_index < (int)GS.global_vertices.size(); ++old_index) {
        if (!to_delete.count(old_index)) {
            index_map[old_index] = new_index++;
        }
    }

    // Compact vertex list
    std::vector<SkeletonVertex> new_vertices;
    std::vector<std::vector<int>> new_adj;
    for (int i = 0; i < (int)GS.global_vertices.size(); ++i) {
        if (index_map[i] != -1) {
            new_vertices.push_back(GS.global_vertices[i]);
            std::vector<int> new_neighbors;
            for (int nbr : GS.global_adj[i]) {
                if (index_map[nbr] != -1 && index_map[nbr] != index_map[i]) {
                    new_neighbors.push_back(index_map[nbr]);
                }
            }
            std::sort(new_neighbors.begin(), new_neighbors.end());
            new_neighbors.erase(std::unique(new_neighbors.begin(), new_neighbors.end()), new_neighbors.end());
            new_adj.push_back(std::move(new_neighbors));
        }
    }

    // Update joints
    for (int& idx : GS.joints) {
        idx = index_map[idx];
    }
    GS.joints.erase(std::remove(GS.joints.begin(), GS.joints.end(), -1), GS.joints.end());

    GS.global_vertices = std::move(new_vertices);
    GS.global_adj = std::move(new_adj);

    // Update point cloud...
    GS.global_vertices_cloud->clear();
    for (auto &gver : GS.global_vertices) {
        pcl::PointXYZ pt(gver.position(0), gver.position(1), gver.position(2));
        GS.global_vertices_cloud->points.push_back(pt);
    }

    // Update leafs and joints...
    graph_decomp();
}

void PathPlanner::prune_branches() {
    if (GS.global_vertices.empty() || GS.global_adj.empty()) {
        RCLCPP_WARN(node_->get_logger(), "Empty Graph!");
        return;
    }

    int N_ver = GS.global_vertices.size();
    if (N_ver < 5 && GS.joints.size() == 0) return; // Allow skeleton to initialize

    std::vector<bool> to_keep(N_ver, true);
    std::unordered_set<int> joint_set(GS.joints.begin(), GS.joints.end());

    // For each leaf: If it is directly connected to a joint -> Prune the leaf! 
    for (int leaf : GS.leafs) {
        if (GS.global_adj[leaf].size() != 1) continue; // Sanity check

        int nb = GS.global_adj[leaf][0];
        if (joint_set.count(nb)) {
            to_keep[leaf] = false;
        }
    }

    /* Remap the indices */
    std::vector<int> old_to_new(GS.global_vertices.size(), -1);
    std::vector<SkeletonVertex> new_vertices;
    pcl::PointCloud<pcl::PointXYZ>::Ptr new_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<std::vector<int>> new_adj;

    for (size_t i = 0; i < to_keep.size(); ++i) {
        if (to_keep[i]) {
            int new_idx = (int)new_vertices.size();
            old_to_new[i] = new_idx;

            new_vertices.push_back(GS.global_vertices[i]);
            new_cloud->points.emplace_back(GS.global_vertices_cloud->points[i]);
            new_adj.emplace_back();
        }
    }

    for (size_t i = 0; i < to_keep.size(); ++i) {
        if (!to_keep[i]) continue;

        int new_idx = old_to_new[i];
        for (int nb : GS.global_adj[i]) {
            if (to_keep[nb]) {
                int new_nb = old_to_new[nb];
                new_adj[new_idx].push_back(new_nb);
            }
        }
    }

    int pruned_count = (int)GS.global_vertices.size() - (int)new_vertices.size();
    RCLCPP_INFO(node_->get_logger(), "Pruned %d leaf nodes directly connected to joints.", pruned_count);

    GS.global_vertices = std::move(new_vertices);
    GS.global_vertices_cloud = new_cloud;
    GS.global_adj = std::move(new_adj);

    // Update leafs and joints...
    graph_decomp();
}



void PathPlanner::clean_skeleton_graph() {
    if (GS.global_vertices.empty() || GS.global_adj.empty()) {
        RCLCPP_WARN(node_->get_logger(), "Cannot clean an empty graph.");
        return;
    }
    int N_ver = GS.global_vertices.size();

    if (N_ver < 5 && GS.joints.size() == 0) return; // Allow skeleton to build up

    const int min_branch_nodes = 3;
    const int smooth_iter = 1;
    RCLCPP_INFO(node_->get_logger(), "Cleaning skeleton: pruning and smoothing...");

    // --- Step 1: Mark small branches for removal ---
    std::vector<bool> visited(N_ver, false);
    std::vector<bool> to_keep(N_ver, true); // assume keeping all initially

    for (int i = 0; i < N_ver; ++i) {
        if (!visited[i] && GS.global_adj[i].size() == 1) {
            std::vector<int> branch;
            int current = i, prev = -1;

            // Walk from leaf node to junction or termination
            while (true) {
                visited[current] = true;
                branch.push_back(current);

                if (GS.global_adj[current].empty()) break;

                int next = -1;
                for (int nb : GS.global_adj[current]) {
                    if (nb != prev) {
                        next = nb;
                        break;
                    }
                }

                if (next == -1 || GS.global_adj[current].size() > 2) break;

                prev = current;
                current = next;
            }

            if ((int)branch.size() < min_branch_nodes) {
                for (int idx : branch) to_keep[idx] = false;
            }
        }
    }

    // --- Step 2: Remap indices and filter vertices and edges ---
    std::vector<int> old_to_new(GS.global_vertices.size(), -1);
    std::vector<SkeletonVertex> new_vertices;
    pcl::PointCloud<pcl::PointXYZ>::Ptr new_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<std::vector<int>> new_adj;

    for (size_t i = 0; i < to_keep.size(); ++i) {
        if (to_keep[i]) {
            int new_idx = (int)new_vertices.size();
            old_to_new[i] = new_idx;

            new_vertices.push_back(GS.global_vertices[i]);
            new_cloud->points.emplace_back(GS.global_vertices_cloud->points[i]);
            new_adj.emplace_back();
        }
    }

    for (size_t i = 0; i < to_keep.size(); ++i) {
        if (!to_keep[i]) continue;

        int new_idx = old_to_new[i];
        for (int nb : GS.global_adj[i]) {
            if (to_keep[nb]) {
                int new_nb = old_to_new[nb];
                new_adj[new_idx].push_back(new_nb);
            }
        }
    }

    int pruned_count = (int)GS.global_vertices.size() - (int)new_vertices.size();
    RCLCPP_INFO(node_->get_logger(), "Removed %d vertices from pruned branches.", pruned_count);

    // --- Step 3: Smoothing (on remaining vertices) ---
    for (int iter = 0; iter < smooth_iter; ++iter) {
        std::vector<Eigen::Vector3f> new_positions(new_cloud->size());

        for (size_t i = 0; i < new_cloud->size(); ++i) {
            if (new_adj[i].empty()) {
                new_positions[i] = new_cloud->points[i].getVector3fMap();
                continue;
            }

            Eigen::Vector3f avg = new_cloud->points[i].getVector3fMap();
            float total_weight = 1.0f;

            for (int nb : new_adj[i]) {
                Eigen::Vector3f diff = new_cloud->points[nb].getVector3fMap() - new_cloud->points[i].getVector3fMap();
                float dist = diff.norm();
                float w = std::exp(-dist * dist); // Gaussian-like
                avg += w * new_cloud->points[nb].getVector3fMap();
                total_weight += w;
            }

            new_positions[i] = avg / total_weight;
        }

        for (size_t i = 0; i < new_cloud->size(); ++i) {
            new_cloud->points[i].x = new_positions[i].x();
            new_cloud->points[i].y = new_positions[i].y();
            new_cloud->points[i].z = new_positions[i].z();
            new_vertices[i].position = new_positions[i].cast<double>();
        }
    }

    // --- Step 4: Replace global skeleton ---
    GS.global_vertices = std::move(new_vertices);
    GS.global_vertices_cloud = new_cloud;
    GS.global_adj = std::move(new_adj);

    RCLCPP_INFO(node_->get_logger(), "Skeleton cleaning complete.");
}



void PathPlanner::select_waypoint() {
    if (!GS.global_vertices_cloud || GS.global_vertices_cloud->empty()) return;

    GP.current_waypoints->clear();
    const int K_ver = 2;
    double disp_dist = 10;

    pcl::KdTreeFLANN<pcl::PointXYZ> vertex_tree;
    vertex_tree.setInputCloud(GS.global_vertices_cloud);

    const pcl::PointXYZ pos(pose.position(0), pose.position(1), pose.position(2));
    std::vector<int> indices;
    std::vector<float> sq_dists;

    if (vertex_tree.nearestKSearch(pos, K_ver, indices, sq_dists) <= 1) {
        RCLCPP_INFO(node_->get_logger(), "Did not find enough vertices to plan path...");
        return;
    }
    // Check if the vertices are connected in the graph!
    // Check if vertices are marked visited

    Eigen::Vector3d u;
    const Eigen::Vector3d p1 = GS.global_vertices[indices[0]].position;
    const Eigen::Vector3d p2 = GS.global_vertices[indices[1]].position;
    Eigen::Vector3d dir = p2 - p1;

    if (dir.norm() < 1e-2) return;  // Avoid near-zero direction

    Eigen::Vector2d dir_xy = dir.head<2>(); // Direction in xy-plane only

    if (dir_xy.norm() > 0.1) {
        // Case where points are not only in z-direction (vertical)
        dir_xy.normalize();
        Eigen::Vector3d u1(-dir_xy.y(), dir_xy.x(), 0.0);
        Eigen::Vector3d u2(dir_xy.y(), -dir_xy.x(), 0.0);
        double d1 = (pose.position - (p2 + u1)).squaredNorm();
        double d2 = (pose.position - (p2 + u2)).squaredNorm();
        u = (d1 < d2) ? u1 : u2; // if d1 < d2 -> u=u1 else u=u2;
    }
    else {
        Eigen::Vector2d to_drone_xy = (pose.position.head<2>() - p2.head<2>());
        if (to_drone_xy.norm() < 1e-2) return;
        to_drone_xy.normalize();
        u = Eigen::Vector3d(to_drone_xy.x(), to_drone_xy.y(), 0.0);
    }

    // Generate and store the waypoint
    const Eigen::Vector3d wayp = p2 + u * disp_dist;
    GP.current_waypoints->points.emplace_back(wayp.x(), wayp.y(), wayp.z());
    GP.current_waypoints->points.emplace_back(pos);
    for (const auto &id : indices) {
        GP.current_waypoints->points.emplace_back(GS.global_vertices_cloud->points[id]); // adds the two points
    }
}



/* Helper Function */

void PathPlanner::graph_decomp() {
    GS.joints.clear();
    GS.leafs.clear();

    for (int i=0; i<(int)GS.global_adj.size(); ++i) {
        int degree = GS.global_adj[i].size();
        if (degree == 1) {
            GS.leafs.push_back(i);
        }
        if (degree >= 2) {
            GS.joints.push_back(i);
        }
    }
}