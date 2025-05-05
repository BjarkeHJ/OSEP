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

    GP.curr_id = -1;
    GS.gskel_size = 0;
}

void PathPlanner::plan_path() {
    auto t_start = std::chrono::high_resolution_clock::now();
    
    // RUN GENERAL PATH PLANNING SELECTION ECT...
    
    // Manage adjusted points with a flag
    // if adjust_flag: -> adjusted points are recieved and 
    
    select_viewpoints();
    
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
    smooth_vertex_positions();
    mst();
    vertex_merge();
    prune_branches();

    GS.gskel_size = (int)GS.global_vertices.size();
    RCLCPP_INFO(node_->get_logger(), "Global Skeleton Size: %d", GS.gskel_size);

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_elapsed = t_end - t_start;
    RCLCPP_INFO(node_->get_logger(), "[Skeleton Update] Time Elapsed: %f seconds", t_elapsed.count());
}


/* Skeleton Updateing*/
void PathPlanner::skeleton_increment() {
    if (!local_vertices || local_vertices->empty()) {
        RCLCPP_INFO(node_->get_logger(), "No New Vertices...");
        return;
    }

    RCLCPP_INFO(node_->get_logger(), "Updating Global Skeleton...");

    std::vector<int> ids_to_delete;

    for (auto &pt : local_vertices->points) {
        Eigen::Vector3d ver(pt.x, pt.y, pt.z);
        bool matched = false;

        for (int i=0; i<(int)GS.prelim_vertices.size(); ++i) {
            auto &gver = GS.prelim_vertices[i];

            double sq_dist = (gver.position - ver).squaredNorm();
            if (sq_dist < fuse_dist_th*fuse_dist_th) {
                
                if (gver.freeze) {
                    matched = true;
                    continue;
                }

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
                    gver.freeze = true; // Freeze vertex!
                }
                else {
                    gver.unconfirmed_check++;
                }
                
                // Mark ids as invalid and schedule for removal...
                if (gver.unconfirmed_check > max_obs_wo_conf) {
                    ids_to_delete.push_back(i);
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

    // Delete points that did not pass the confidence check (reverse indexing for id consistency)
    for (auto it = ids_to_delete.rbegin(); it != ids_to_delete.rend(); ++it) {
        GS.prelim_vertices.erase(GS.prelim_vertices.begin() + *it);
    }

    // Pass confident vertices to the global skeleton...
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

    for (size_t i = 0; i < GS.global_vertices_cloud->size(); ++i) {
        std::vector<int> indices;
        std::vector<float> distances;

        int n_neighbors = adj_tree.nearestKSearch(GS.global_vertices_cloud->points[i], K, indices, distances);

        for (int j = 1; j < n_neighbors; ++j) { // Skip self (index 0)
            int nb_idx = indices[j];
            float dist_to_nb = (GS.global_vertices_cloud->points[i].getVector3fMap() - GS.global_vertices_cloud->points[nb_idx].getVector3fMap()).norm();

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

void PathPlanner::smooth_vertex_positions() {
    if (GS.global_vertices.empty() || GS.global_adj.empty()) return;

    std::vector<Eigen::Vector3d> new_positions(GS.global_vertices.size());

    for (size_t i = 0; i < GS.global_vertices.size(); ++i) {
        const auto& v = GS.global_vertices[i];
        const auto& nbrs = GS.global_adj[i];

        if (v.type == 1 || v.type == 3 || nbrs.size() < 2) {
            new_positions[i] = v.position;  // Do not smooth leafs or joints
            continue;
        }

        Eigen::Vector3d avg = Eigen::Vector3d::Zero();
        for (int j : nbrs) {
            avg += GS.global_vertices[j].position;
        }
        avg /= static_cast<double>(nbrs.size());

        double blend = 0.9;
        new_positions[i] = (1.0 - blend) * v.position + blend * avg;
    }

    for (size_t i = 0; i < GS.global_vertices.size(); ++i) {
        GS.global_vertices[i].position = new_positions[i];
    }

    // Update point cloud too
    GS.global_vertices_cloud->clear();
    for (const auto& v : GS.global_vertices) {
        pcl::PointXYZ pt(v.position(0), v.position(1), v.position(2));
        GS.global_vertices_cloud->points.push_back(pt);
    }

    RCLCPP_INFO(node_->get_logger(), "Smoothed vertex positions.");
}

void PathPlanner::mst() {
    int N_ver = GS.global_vertices.size();
    if (N_ver == 0 || GS.global_adj.empty()) {
        RCLCPP_WARN(node_->get_logger(), "Global skeleton is empty, cannot extract MST.");
        return;
    }

    struct WeightedEdge {
        int u, v;
        double weight;
        bool operator<(const WeightedEdge& other) const { return weight < other.weight; }
    };

    std::vector<WeightedEdge> mst_edges;

    for (int i = 0; i < N_ver; ++i) {
        const Eigen::Vector3d& pi = GS.global_vertices[i].position;

        for (int nb : GS.global_adj[i]) {
            if (nb <= i) continue;

            const Eigen::Vector3d& pj = GS.global_vertices[nb].position;
            Eigen::Vector3d dir = pj - pi;
            double dist = dir.norm();
            if (dist < 1e-3) continue;
            dir.normalize();

            // Check linearity: compare direction with local tangent
            Eigen::Vector3d smooth_dir = Eigen::Vector3d::Zero();
            int count = 0;

            for (int nnb : GS.global_adj[i]) {
                if (nnb == nb || nnb == i) continue;
                Eigen::Vector3d neighbor_dir = GS.global_vertices[nnb].position - pi;
                if (neighbor_dir.norm() > 1e-3) {
                    smooth_dir += neighbor_dir.normalized();
                    ++count;
                }
            }

            if (count > 0) smooth_dir.normalize();

            // Penalize direction changes
            double angle_penalty = 5.0;
            if (count > 0) {
                double dot = dir.dot(smooth_dir);
                angle_penalty = 1.0 + (1.0 - dot);  // favors alignment (dot=1)
            }

            double weighted_cost = dist * angle_penalty;

            mst_edges.push_back({i, nb, weighted_cost});
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

    graph_decomp();
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

                GS.global_vertices[i].visited_cnt = std::max(GS.global_vertices[i].visited_cnt, GS.global_vertices[j].visited_cnt);

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

void PathPlanner::graph_decomp() {
    /* Assigns vertex type */
    GS.joints.clear();
    GS.leafs.clear();

    for (int i=0; i<(int)GS.global_vertices.size(); ++i) {
        int degree = GS.global_adj[i].size();

        if (degree == 1) {
            GS.leafs.push_back(i);
            GS.global_vertices[i].type = 1;
        }

        else if (degree == 2) {
            GS.global_vertices[i].type = 2;
        }

        else if (degree >= 2) {
            GS.joints.push_back(i);
            GS.global_vertices[i].type = 3;
        }

        else {
            GS.global_vertices[i].type = 0;
        }
    }
}

/* Viewpoint Generation and Path Planning */
void PathPlanner::select_viewpoints() {
    if (!GS.global_vertices_cloud || GS.global_vertices_cloud->empty()) return;
    if (GS.global_vertices.size() < 2) {
        RCLCPP_WARN(node_->get_logger(), "[Select Viewpoints] Too few vertices to plan.");
        return;
    }

    std::vector<Viewpoint> temp_vpts;

    const int k_vpts = 10;

    // Do not add more viewpoints if already at capacity
    if ((int)GP.local_vpts.size() >= k_vpts) {
        RCLCPP_INFO(node_->get_logger(), "[Select Viewpoints] Already have %d viewpoints. Skipping update.", k_vpts);
        return;
    }

    // Initialize the starting vertex if not already set
    if (GP.curr_id == -1) {
        pcl::PointXYZ dpos(pose.position(0), pose.position(1), pose.position(2));
        pcl::KdTreeFLANN<pcl::PointXYZ> vertex_tree;
        vertex_tree.setInputCloud(GS.global_vertices_cloud);
        std::vector<int> id(1);
        std::vector<float> sq_dist(1);
        if (vertex_tree.nearestKSearch(dpos, 1, id, sq_dist) < 1) {
            RCLCPP_WARN(node_->get_logger(), "[Select Viewpoints] Nearest skeleton vertex not found.");
            return;
        }
        GP.curr_id = id[0];
    }

    // Only generate if current vertex is not a leaf
    if (GS.global_vertices[GP.curr_id].type == 1) {
        RCLCPP_INFO(node_->get_logger(), "[Select Viewpoints] Current vertex is a leaf. No further planning.");
        return;
    }

    while ((int)temp_vpts.size() < k_vpts) {
        const auto& current_ver = GS.global_vertices[GP.curr_id];
        int next_id = -1;

        switch (current_ver.type) {
            case 0: {
                RCLCPP_WARN(node_->get_logger(), "[Select Viewpoints] Type 0: Invalid vertex.");
                GP.curr_id = -1;
                return;
            }

            case 1: {
                RCLCPP_INFO(node_->get_logger(), "[Select Viewpoints] Type 1: Leaf node.");
                int nb_id = GS.global_adj[GP.curr_id][0];
                const auto& leaf_nb = GS.global_vertices[nb_id];

                if (leaf_nb.visited_cnt <= current_ver.visited_cnt) {
                    next_id = nb_id;
                }

                else {
                    return;
                }

                break;
            }

            case 2: {
                RCLCPP_INFO(node_->get_logger(), "[Select Viewpoints] Type 2: Branch node.");
                const auto& nbs_ids = GS.global_adj[GP.curr_id];
                int min_visits = std::numeric_limits<int>::max();
                for (int id : nbs_ids) {
                    if (GS.global_vertices[id].visited_cnt < min_visits) {
                        min_visits = GS.global_vertices[id].visited_cnt;
                        next_id = id;
                    }
                }
                break;
            }

            case 3: {
                RCLCPP_INFO(node_->get_logger(), "[Select Viewpoints] Type 3: Junction node.");
                next_id = find_next_toward_furthest_leaf(GP.curr_id);
                if (next_id == -1) {
                    RCLCPP_INFO(node_->get_logger(), "[Select Viewpoints] No path to unvisited leaves from junction.");
                    return;
                }
                break;
            }

            default: {
                RCLCPP_WARN(node_->get_logger(), "[Select Viewpoints] Unknown vertex type.");
                return;
            }
        }

        if (next_id == -1) {
            RCLCPP_WARN(node_->get_logger(), "[Select Viewpoints] No valid next vertex found.");
            return;
        }

        Viewpoint next_wp = generate_viewpoint(GP.curr_id, next_id);
        
        GS.global_vertices[GP.curr_id].visited_cnt++;

        // GP.local_vpts.push(next_wp);
        temp_vpts.push_back(next_wp);
        GP.curr_id = next_id;
    }

    
    // === Find index of closest viewpoint to drone ===
    int best_idx = 0;
    double min_dist = std::numeric_limits<double>::max();
    for (int i = 0; i < (int)temp_vpts.size(); ++i) {
        double d = (temp_vpts[i].position - pose.position).squaredNorm();
        if (d < min_dist) {
            min_dist = d;
            best_idx = i;
        }
    }

    // === Rotate vector so closest viewpoint is first ===
    std::rotate(temp_vpts.begin(), temp_vpts.begin() + best_idx, temp_vpts.end());

    // === Refill queue ===
    while (!GP.local_vpts.empty()) GP.local_vpts.pop();
    for (const auto& vp : temp_vpts) {
        GP.local_vpts.push(vp);
    }


    RCLCPP_INFO(node_->get_logger(), "[Select Viewpoints] Planned up to %d viewpoints.", k_vpts);
}

Viewpoint PathPlanner::generate_viewpoint(int id, int id_next) {
    Viewpoint vp;
    Eigen::Vector3d u;
    double disp_dist = 15.0;

    const Eigen::Vector3d p1 = GS.global_vertices[id].position;
    const Eigen::Vector3d p2 = GS.global_vertices[id_next].position;
    Eigen::Vector3d dir = p2 - p1;
    if (dir.norm() < 1e-2) return Viewpoint{};

    Eigen::Vector2d dir_xy = dir.head<2>();

    if (dir_xy.norm() > 0.5) {
        dir_xy.normalize();
        Eigen::Vector3d u1(-dir_xy.y(), dir_xy.x(), 0.0);
        Eigen::Vector3d u2(dir_xy.y(), -dir_xy.x(), 0.0);
        double d1 = (pose.position - (p2 + u1)).squaredNorm();
        double d2 = (pose.position - (p2 + u2)).squaredNorm();
        u = (d1 < d2) ? u1 : u2;
    }

    else {
        Eigen::Vector2d to_drone_xy = (pose.position.head<2>() - p2.head<2>());
        if (to_drone_xy.norm() < 1e-2) return Viewpoint{};
        to_drone_xy.normalize();
        u = Eigen::Vector3d(to_drone_xy.x(), to_drone_xy.y(), 0.0);
    }

    const Eigen::Vector3d vp_pos = p2 + u * disp_dist;
    const Eigen::Vector3d vp_ori = -u.normalized();

    double yaw = std::atan2(vp_ori.y(), vp_ori.x());
    Eigen::AngleAxisd yaw_rot(yaw, Eigen::Vector3d::UnitZ());

    vp.position = vp_pos;
    vp.orientation = Eigen::Quaterniond(yaw_rot);

    RCLCPP_INFO(node_->get_logger(), "Generated Viewpoint at (%.2f, %.2f, %.2f)",
            vp.position.x(), vp.position.y(), vp.position.z());

    return vp;
}

int PathPlanner::find_next_toward_furthest_leaf(int start_id) {
    int max_depth = -1;
    std::vector<int> best_path;
    std::unordered_set<int> visited;
    std::vector<int> current_path;

    // Recursive depth first search
    std::function<void(int, int)> dfs = [&](int v, int depth) {
        visited.insert(v);
        current_path.push_back(v);
        const auto& node = GS.global_vertices[v];

        if (node.type == 1 && node.visited_cnt == 0) {
            if (depth > max_depth) {
                max_depth = depth;
                best_path = current_path;
            }
        }

        if (v < (int)GS.global_adj.size()) {
            for (int nb : GS.global_adj[v]) {
                if (!visited.count(nb) && GS.global_vertices[nb].visited_cnt == 0) {
                    dfs(nb, depth + 1);
                }
            }
        }

        current_path.pop_back();
    };

    dfs(start_id, 0);

    // If a valid path is found and has at least two nodes, return the second node (next step)
    if (best_path.size() >= 2)
        return best_path[1];

    return -1;  // No valid next step
}




// void PathPlanner::mst() {
//     int N_ver = GS.global_vertices.size();
//     if (N_ver == 0 || GS.global_adj.empty()) {
//         RCLCPP_WARN(node_->get_logger(), "Global skeleton is empty, cannot extract MST.");
//         return;
//     }

//     std::vector<Edge> mst_edges;
//     for (int i = 0; i < N_ver; ++i) {
//         for (int nb : GS.global_adj[i]) {
//             if (nb <= i) continue; // Avoid bi-directional check
//             Eigen::Vector3d ver_i = GS.global_vertices[i].position;
//             Eigen::Vector3d ver_nb = GS.global_vertices[nb].position;
//             double weight = (ver_i - ver_nb).norm();
//             mst_edges.push_back({i, nb, weight});
//         }
//     }

//     std::sort(mst_edges.begin(), mst_edges.end());

//     UnionFind uf(N_ver);
//     std::vector<std::vector<int>> mst_adj(N_ver);

//     for (const auto& edge : mst_edges) {
//         if (uf.unite(edge.u, edge.v)) {
//             mst_adj[edge.u].push_back(edge.v);
//             mst_adj[edge.v].push_back(edge.u);
//         }
//     }

//     GS.global_adj = std::move(mst_adj);

//     // Identify leafs and joints
//     graph_decomp();
// }






// void PathPlanner::select_viewpoints() {
//     if (!GS.global_vertices_cloud || GS.global_vertices_cloud->empty()) return;

//     if (GS.global_vertices.size() < 2) {
//         RCLCPP_WARN(node_->get_logger(), "Too few vertices to plan.");
//         return;
//     }

//     if (GP.curr_id == -1) {
//         /* Initialize Planning (First Waypoint) */
//         pcl::PointXYZ dpos(pose.position(0), pose.position(1), pose.position(2));
//         pcl::KdTreeFLANN<pcl::PointXYZ> vertex_tree;
//         vertex_tree.setInputCloud(GS.global_vertices_cloud);
//         std::vector<int> id(1);
//         std::vector<float> sq_dist(1);
//         if (vertex_tree.nearestKSearch(dpos, 1, id, sq_dist) < 1) {
//             RCLCPP_WARN(node_->get_logger(), "[Select Viewpoints] Did not find a vertex");
//             return;
//         }
//         GP.curr_id = id[0];
//     }

//     if (GS.global_vertices[GP.curr_id].type != 1) {
//         int k_wpts = 5;
//         while ((int)GP.local_vpts.size() <= k_wpts) {
//             auto& current_ver = GS.global_vertices[GP.curr_id];
//             int next_id = -1;

//             // invalid point
//             if (current_ver.type == 0) {
//                 RCLCPP_WARN(node_->get_logger(), "[Select Viewpoints] Type 0: Current Vertice Invalid!");
//                 return;
//             }

//             if (current_ver.type == 1) {
//                 RCLCPP_INFO(node_->get_logger(), "[Select Viewpoints] Type 1: Leaf...");
//                 return;
//             }

//             if (current_ver.type == 2) {
//                 RCLCPP_INFO(node_->get_logger(), "[Select Viewpoints] Type 2: Branch...");
//                 std::vector<int> nbs_ids = GS.global_adj[GP.curr_id];
//                 int vis = 100; // arbitrary large int
//                 for (int id : nbs_ids) {
//                     if (GS.global_vertices[id].visited_cnt < vis) {
//                         vis = GS.global_vertices[id].visited_cnt;
//                         next_id = id;
//                     }
//                 }
//             }

//             if (current_ver.type == 3) {
//                 RCLCPP_INFO(node_->get_logger(), "[Select Viewpoints] Type 3: Junction...");
//                 int temp = find_next_toward_furthest_leaf(GP.curr_id);
//                 if (temp != -1) {
//                     next_id = temp;
//                 }
//                 else {
//                     RCLCPP_INFO(node_->get_logger(), "[Select Viewpoints] No path to unvisited leaves from junction.");
//                 }
//             }



//             Viewpoint next_wp = generate_viewpoint(GP.curr_id, next_id); // Define new viewpoint (set position and orientation)

//             GP.local_vpts.push(next_wp);
//             GP.curr_id = next_id;
//         }
//     }
// }









    // GP.current_waypoints->clear();
    // const int K_ver = 2;
    // double disp_dist = 10;

    // pcl::KdTreeFLANN<pcl::PointXYZ> vertex_tree;
    // vertex_tree.setInputCloud(GS.global_vertices_cloud);

    // const pcl::PointXYZ pos(pose.position(0), pose.position(1), pose.position(2));
    // std::vector<int> indices;
    // std::vector<float> sq_dists;

    // if (vertex_tree.nearestKSearch(pos, K_ver, indices, sq_dists) <= 1) {
    //     RCLCPP_INFO(node_->get_logger(), "Did not find enough vertices to plan path...");
    //     return;
    // }

    // // Check if the vertices are connected in the graph!
    // // Check if vertices are marked visited

    // Eigen::Vector3d u;
    // const Eigen::Vector3d p1 = GS.global_vertices[indices[0]].position;
    // const Eigen::Vector3d p2 = GS.global_vertices[indices[1]].position;
    // Eigen::Vector3d dir = p2 - p1;

    // if (dir.norm() < 1e-2) return;  // Avoid near-zero direction

    // Eigen::Vector2d dir_xy = dir.head<2>(); // Direction in xy-plane only

    // if (dir_xy.norm() > 0.5) {
    //     // Case where points are not only in z-direction (vertical)
    //     dir_xy.normalize();
    //     Eigen::Vector3d u1(-dir_xy.y(), dir_xy.x(), 0.0);
    //     Eigen::Vector3d u2(dir_xy.y(), -dir_xy.x(), 0.0);
    //     double d1 = (pose.position - (p2 + u1)).squaredNorm();
    //     double d2 = (pose.position - (p2 + u2)).squaredNorm();
    //     u = (d1 < d2) ? u1 : u2; // if d1 < d2 -> u=u1 else u=u2;
    // }
    // else {
    //     Eigen::Vector2d to_drone_xy = (pose.position.head<2>() - p2.head<2>());
    //     if (to_drone_xy.norm() < 1e-2) return;
    //     to_drone_xy.normalize();
    //     u = Eigen::Vector3d(to_drone_xy.x(), to_drone_xy.y(), 0.0);
    // }

    // // Generate and store the waypoint
    // const Eigen::Vector3d wayp = p2 + u * disp_dist;
    // GP.current_waypoints->points.emplace_back(wayp.x(), wayp.y(), wayp.z());
    // GP.current_waypoints->points.emplace_back(pos);
    // for (const auto &id : indices) {
    //     GP.current_waypoints->points.emplace_back(GS.global_vertices_cloud->points[id]); // adds the two points
    // }