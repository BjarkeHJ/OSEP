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

    GP.curr_id = -1;
    GS.gskel_size = 0;
    N_new_vers = 0;
}

void PathPlanner::update_skeleton() {
    auto t_start = std::chrono::high_resolution_clock::now();

    skeleton_increment();
    graph_adj();
    mst();
    vertex_merge();
    smooth_vertex_positions();
    prune_branches();

    N_new_vers = (int)GS.global_vertices.size() - GS.gskel_size; // Number of new vertices added
    RCLCPP_INFO(node_->get_logger(), "New Vertices: %d", N_new_vers);

    GS.gskel_size = (int)GS.global_vertices.size(); // Update total number of vertices
    RCLCPP_INFO(node_->get_logger(), "Global Skeleton Size: %d", GS.gskel_size);

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_elapsed = t_end - t_start;
    RCLCPP_INFO(node_->get_logger(), "[Skeleton Update] Time Elapsed: %f seconds", t_elapsed.count());
}


void PathPlanner::plan_path() {
    auto t_start = std::chrono::high_resolution_clock::now();
    if (GS.global_vertices.size() < 2) {
        RCLCPP_WARN(node_->get_logger(), "[Select Viewpoints] Too few vertices to plan.");
        return;
    }

    // RUN GENERAL PATH PLANNING SELECTION ECT...

    // If: only first (start) vertex is found
    // Do: Fly towards it
    // Manage adjusted points with a flag
    // if adjust_flag: -> adjusted points are recieved and 

    // Wait for number of vertices before planning?
    
    // Sample Viewpoints

    viewpoint_sampling();
    viewpoint_filtering();
    generate_path();

    RCLCPP_INFO(node_->get_logger(), "Number of Viewpoints Generated: %d", (int)GP.global_vpts.size());

    N_new_vers = 0; // Reset

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_elapsed = t_end - t_start;
    RCLCPP_INFO(node_->get_logger(), "[Path Planning] Time Elapsed: %f seconds", t_elapsed.count());
}


/* Skeleton Updating*/
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

        double blend = 0.8;
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
        GS.global_vertices[i].updated = false; // Reset update flag for each vertex

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
            double angle_penalty = 10.0;
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
        int prev_type = GS.global_vertices[i].type;
        int degree = GS.global_adj[i].size();
        int new_type = 0;
        
        if (degree == 1) { // Leaf
            GS.leafs.push_back(i);
            // GS.global_vertices[i].type = 1;
            new_type = 1;
        }

        else if (degree == 2) { // branch
            // GS.global_vertices[i].type = 2;
            new_type = 2;
        }

        else if (degree > 2) { // joint
            GS.joints.push_back(i);
            // GS.global_vertices[i].type = 3;
            new_type = 3;
        }

        // else { // Invalid
        //     GS.global_vertices[i].type = 0;
        // }

        GS.global_vertices[i].type = new_type;

        // Mark updated vertices if they are updated in this iteration
        if (GS.global_vertices[i].updated) continue;
        GS.global_vertices[i].updated = GS.global_vertices[i].updated || (GS.global_vertices[i].type != prev_type);
    }
}


/* Viewpoint Generation and Path Planning */
void PathPlanner::viewpoint_sampling() {
    // if (N_new_vers == 0) return;

    for (int i=0; i<GS.gskel_size; ++i) {
        if (!GS.global_vertices[i].updated || GS.global_vertices[i].type == 0) continue;

        std::vector<Viewpoint> leaf_vps = generate_viewpoint(i);
        if (leaf_vps.empty()) continue;

        for (int i=0; i<(int)leaf_vps.size(); ++i) {
            GP.global_vpts.push_back(leaf_vps[i]);
        }
    }
}

void PathPlanner::viewpoint_filtering() {
    if (GS.global_pts->empty()) {
        GP.global_vpts.clear();
        RCLCPP_WARN(node_->get_logger(), "No VoxelMap - Skipping Viewpoint Generation");
        return;
    }

    std::vector<Viewpoint> filtered;

    // Step 1: Always preserve locked viewpoints
    for (const auto& vp : GP.global_vpts) {
        if (vp.locked) {
            filtered.push_back(vp);
        }
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> voxel_tree;
    voxel_tree.setInputCloud(GS.global_pts);

    // Step 2: Process only unlocked viewpoints
    for (const auto& vp : GP.global_vpts) {
        if (vp.locked) continue;  // Already handled

        if (!viewpoint_check(vp, voxel_tree)) continue;

        bool discard = false;
        bool merged = false;

        for (auto& kept : filtered) {
            if (viewpoint_similarity(vp, kept)) {
                if (kept.locked) {
                    discard = true; // Similar to locked → delete this viewpoint
                    break;
                } else {
                    // Merge into unlocked kept
                    kept.position = 0.5 * (kept.position + vp.position);

                    Eigen::Quaterniond qa(kept.orientation);
                    Eigen::Quaterniond qb(vp.orientation);
                    qa.normalize(); qb.normalize();
                    kept.orientation = qa.slerp(0.5, qb);

                    merged = true;
                    break;
                }
            }
        }

        if (!discard && !merged) {
            filtered.push_back(vp);
        }
    }

    GP.global_vpts = std::move(filtered);
}

// void PathPlanner::viewpoint_filtering() {
//     if (GS.global_pts->empty()) {
//         GP.global_vpts.clear();
//         RCLCPP_WARN(node_->get_logger(), "No VoxelMap - Skipping Viewpoint Generation");
//         return;
//     }

//     std::vector<Viewpoint> filtered;
//     pcl::KdTreeFLANN<pcl::PointXYZ> voxel_tree;
//     voxel_tree.setInputCloud(GS.global_pts);

//     for (const auto& vp : GP.global_vpts) {
//         if (vp.locked) continue;

//         if (!viewpoint_check(vp, voxel_tree)) continue;

//         bool merged = false;
//         for (auto& kept : filtered) {
//             if (viewpoint_similarity(vp, kept)) {
//                 // Merge vp into kept
//                 kept.position = 0.5 * (kept.position + vp.position);

//                 // Merge orientation (optional — here averaging direction vectors)
//                 Eigen::Quaterniond qa(kept.orientation);
//                 Eigen::Quaterniond qb(vp.orientation);
//                 qa.normalize(); qb.normalize();
//                 kept.orientation = qa.slerp(0.5, qb);

//                 merged = true;
//                 break;
//             }
//         }

//         if (!merged) {
//             filtered.push_back(vp);
//         }
//     }

//     GP.global_vpts = std::move(filtered);
// }


void PathPlanner::generate_path() {
    if (GS.global_vertices.empty() || GP.global_vpts.empty()) return;

    GP.local_path.clear();

    // --- Find closest skeleton vertex to drone
    pcl::PointXYZ current_pos(pose.position.x(), pose.position.y(), pose.position.z());
    pcl::KdTreeFLANN<pcl::PointXYZ> vertex_tree;
    vertex_tree.setInputCloud(GS.global_vertices_cloud);
    std::vector<int> nearest_id(1);
    std::vector<float> sq_dist(1);
    if (vertex_tree.nearestKSearch(current_pos, 1, nearest_id, sq_dist) < 1) {
        RCLCPP_WARN(node_->get_logger(), "No nearby skeleton vertex found...");
        return;
    }
    int start_id = nearest_id[0];

    // --- Plan forward path
    const int max_steps = 50;
    std::vector<int> local_path_ids = find_next_toward_furthest_leaf(start_id, max_steps);
    if (local_path_ids.empty()) {
        RCLCPP_WARN(node_->get_logger(), "Could not generate local path...");
        return;
    }
    std::unordered_set<int> path_vertex_set(local_path_ids.begin(), local_path_ids.end());

    // --- Get drone XY direction (forward facing)
    Eigen::Vector3d drone_dir_3d = pose.orientation * Eigen::Vector3d::UnitX();
    Eigen::Vector2d drone_dir_xy = drone_dir_3d.head<2>().normalized();
    Eigen::Vector3d drone_pos = pose.position;

    const double cos_90deg = std::cos(M_PI / 2.0);  // 0.0

    // --- Score and filter candidate viewpoints
    std::vector<Viewpoint> candidate_vpts;
    for (auto& vp : GP.global_vpts) {
        if (!path_vertex_set.count(vp.corresp_vertex_id)) continue;

        Eigen::Vector3d vp_facing = vp.orientation * Eigen::Vector3d::UnitX();
        Eigen::Vector2d vp_dir_xy = vp_facing.head<2>().normalized();

        double cos_angle = drone_dir_xy.dot(vp_dir_xy);
        if (cos_angle < cos_90deg) continue;

        score_viewpoint(vp);
        candidate_vpts.push_back(vp);
    }

    if (candidate_vpts.empty()) {
        RCLCPP_WARN(node_->get_logger(), "No valid viewpoints aligned with current drone orientation.");
        return;
    }

    // --- Sort by score and take top N
    std::sort(candidate_vpts.begin(), candidate_vpts.end(),
              [](const Viewpoint& a, const Viewpoint& b) { return a.score > b.score; });

    const int N_max = 5;
    std::vector<Viewpoint> selected_vpts;
    for (auto& vp : candidate_vpts) {
        if ((int)selected_vpts.size() >= N_max) break;
        vp.locked = true;
        selected_vpts.push_back(vp);
    }

    // --- Sort selected viewpoints by distance from drone
    std::sort(selected_vpts.begin(), selected_vpts.end(),
              [&](const Viewpoint& a, const Viewpoint& b) {
                  return (a.position - drone_pos).norm() < (b.position - drone_pos).norm();
              });

    // --- Final local path
    GP.local_path = std::move(selected_vpts);
}

std::vector<Viewpoint> PathPlanner::generate_viewpoint(int id) {
    double disp_dist = 6;
    std::vector<Viewpoint> output_vps;

    if (GS.global_vertices[id].type == 1) {

        // leaf
        int id_adj = GS.global_adj[id][0];
        const Eigen::Vector3d p1 = GS.global_vertices[id].position;
        const Eigen::Vector3d p2 = GS.global_vertices[id_adj].position;
        Eigen::Vector3d dir = p2 - p1;

        if (dir.norm() < 1e-2) return output_vps;
        dir.normalize();
        
        Eigen::Vector2d dir_xy = dir.head<2>();
        if (dir_xy.norm() > 0.5) {
            dir_xy.normalize();
            Eigen::Vector3d u1(-dir_xy.y(), dir_xy.x(), 0.0);
            Eigen::Vector3d u2(dir_xy.y(), -dir_xy.x(), 0.0);
            Eigen::Vector3d u3(-dir_xy.x(), -dir_xy.y(), 0.0);
            std::vector<Eigen::Vector3d> dirs = {u1, u2, u3*1.5};
            output_vps = vp_sample(p1, dirs, disp_dist, id);
        }
        else {
            Eigen::Vector2d to_drone_xy = (pose.position.head<2>() - p1.head<2>());
            if (to_drone_xy.norm() < 1e-2) return output_vps;
            to_drone_xy.normalize();
            Eigen::Vector3d u1(to_drone_xy.x(), to_drone_xy.y(), 0.0);
            Eigen::Vector3d u2 = -u1;
            Eigen::Vector3d u3(-u1.y(), u1.x(), 0.0);
            Eigen::Vector3d u4(u1.y(), -u1.x(), 0.0);
            std::vector<Eigen::Vector3d> dirs = {u1, u2, u3, u4};
            output_vps = vp_sample(p1, dirs, disp_dist, id);
        }
    }

    if (GS.global_vertices[id].type == 2) {
        // branch
        int id_adj1 = GS.global_adj[id][0];
        int id_adj2 = GS.global_adj[id][1];
    
        const Eigen::Vector3d p1 = GS.global_vertices[id].position;
        const Eigen::Vector3d p2_1 = GS.global_vertices[id_adj1].position;
        const Eigen::Vector3d p2_2 = GS.global_vertices[id_adj2].position;
        Eigen::Vector3d dir = p2_1 - p2_2;

        if (dir.norm() < 1e-2) return output_vps;
        dir.normalize();
        
        Eigen::Vector2d dir_xy = dir.head<2>();
        if (dir_xy.norm() > 0.5) {
            dir_xy.normalize();
            Eigen::Vector3d u1(-dir_xy.y(), dir_xy.x(), 0.0);
            Eigen::Vector3d u2(dir_xy.y(), -dir_xy.x(), 0.0);
            std::vector<Eigen::Vector3d> dirs = {u1, u2};
            output_vps = vp_sample(p1, dirs, disp_dist, id);
        }
        else {
            Eigen::Vector2d to_drone_xy = (pose.position.head<2>() - p1.head<2>());
            if (to_drone_xy.norm() < 1e-2) return output_vps;
            to_drone_xy.normalize();
            Eigen::Vector3d u1(to_drone_xy.x(), to_drone_xy.y(), 0.0);
            Eigen::Vector3d u2 = -u1;
            Eigen::Vector3d u3(-u1.y(), u1.x(), 0.0);
            Eigen::Vector3d u4(u1.y(), -u1.x(), 0.0);
            std::vector<Eigen::Vector3d> dirs = {u1, u2, u3, u4};
            output_vps = vp_sample(p1, dirs, disp_dist, id);
        }
    }

    if (GS.global_vertices[id].type == 3) {
        // joint
        const Eigen::Vector3d p1 = GS.global_vertices[id].position;
        int N_nbs = GS.global_adj[id].size();
        std::vector<int> nbs = GS.global_adj[id];

        for (int i=0; i<N_nbs; ++i) {
            Eigen::Vector3d p2_1 = GS.global_vertices[nbs[i]].position;
            for (int j=i+1; j<N_nbs; ++j) {
                Eigen::Vector3d p2_2 = GS.global_vertices[nbs[j]].position;
                Eigen::Vector3d dir = p2_1 - p2_2;
                if (dir.norm() < 1e-2) continue;
                dir.normalize();

                Eigen::Vector2d dir_xy = dir.head<2>();
                if (dir_xy.norm() > 0.5) {
                    dir_xy.normalize();
                    Eigen::Vector3d u1(-dir_xy.y(), dir_xy.x(), 0.0);
                    Eigen::Vector3d u2(dir_xy.y(), -dir_xy.x(), 0.0);
                    std::vector<Eigen::Vector3d> dirs = {u1, u2};
                    auto vps = vp_sample(p1, dirs, disp_dist, id);
                    output_vps.insert(output_vps.end(), vps.begin(), vps.end());
                }
                else {
                    Eigen::Vector2d to_drone_xy = (pose.position.head<2>() - p1.head<2>());
                    if (to_drone_xy.norm() < 1e-2) return output_vps;
                    to_drone_xy.normalize();
                    Eigen::Vector3d u1(to_drone_xy.x(), to_drone_xy.y(), 0.0);
                    Eigen::Vector3d u2 = -u1;
                    Eigen::Vector3d u3(-u1.y(), u1.x(), 0.0);
                    Eigen::Vector3d u4(u1.y(), -u1.x(), 0.0);
                    std::vector<Eigen::Vector3d> dirs = {u1, u2, u3, u4};
                    auto vps = vp_sample(p1, dirs, disp_dist, id);
                    output_vps.insert(output_vps.end(), vps.begin(), vps.end());
                }
            }
        }
    }
    return output_vps;
}

std::vector<Viewpoint> PathPlanner::vp_sample(const Eigen::Vector3d& origin, const std::vector<Eigen::Vector3d>& directions, double disp_dist, int vertex_id) {
    std::vector<Viewpoint> viewpoints;
    for (const auto& u : directions) {
        Viewpoint vp;
        vp.position = origin + u * disp_dist;
        double yaw = std::atan2(-u.y(), -u.x());
        vp.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
        vp.corresp_vertex_id = vertex_id; // OBS: Possible issues with merging of vertices
        viewpoints.push_back(vp);
    }
    return viewpoints;
}

bool PathPlanner::viewpoint_check(const Viewpoint& vp, pcl::KdTreeFLANN<pcl::PointXYZ> voxel_tree) {
    std::vector<int> ids;
    std::vector<float> dsq;
    pcl::PointXYZ query_point(vp.position.x(), vp.position.y(), vp.position.z());
    if (voxel_tree.radiusSearch(query_point, min_view_dist, ids, dsq) > 0) {
        return false;
    }
    return true;
}

bool PathPlanner::viewpoint_similarity(const Viewpoint& a, const Viewpoint& b) {
    return ((a.position - b.position).norm() < viewpoint_merge_dist);
}

void PathPlanner::score_viewpoint(Viewpoint &vp) {
    if (GS.global_pts->empty()) {
        RCLCPP_WARN(node_->get_logger(), "Global Points is empty!");
        return;
    }

    /* From a Viewpoint Candidate - Map the camera FOV and identify seen voxels */
    int info_steps = 5;

    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    tree.setInputCloud(GS.global_pts);
    
    pcl::PointXYZ vp_pos(vp.position.x(), vp.position.y(), vp.position.z());
    Eigen::Vector3d cam_dir = vp.orientation * Eigen::Vector3d::UnitX();
    cam_dir.normalize();
    
    // Map FoV cone onto structure 
    float cos_half_h = std::cos((fov_h * 0.5f) * M_PI / 180.0f);
    float cos_half_v = std::cos((fov_v * 0.5f) * M_PI / 180.0f);
    std::unordered_set<VoxelIndex, VoxelIndexHash> seen_here;
    int unseen_voxels = 0;

    for (const auto &pt : GS.global_pts->points) {
        Eigen::Vector3d target(pt.x, pt.y, pt.z);
        Eigen::Vector3d vec = target - vp.position;

        double dist = vec.norm();
        if (dist < min_view_dist || dist > max_view_dist) continue;

        Eigen::Vector3d dir = vec.normalized();
        double cos_angle_h = dir.dot(cam_dir);
        if (cos_angle_h < cos_half_h) continue;

        double cos_angle_v = std::abs(dir.dot(Eigen::Vector3d::UnitZ()));
        if (cos_angle_v < cos_half_v) continue;

        // It is inside the FoV-cone
        // Do ray-casting from the viewpoint towards the vertex (in steps)

        for (int i=1; i<=info_steps; ++i) {
            Eigen::Vector3d sample = vp.position + dir * (dist * i / double(info_steps));
            VoxelIndex key = {
                int(std::floor(sample.x() / voxel_size)),
                int(std::floor(sample.y() / voxel_size)),
                int(std::floor(sample.z() / voxel_size))
            };

            if (seen_here.count(key)) continue; // voxel
            seen_here.insert(key);

            if (GS.voxels.count(key) == 0) continue; // Skip free-space voxels...

            vp.visible_voxels.insert(key); // Assign visible voxels to the viewpoint
            
            auto it = GS.seen_voxels.find(key);
            if (it == GS.seen_voxels.end() || it->second < 1) {
                unseen_voxels++;
            }

            break;
        }
    }

    double score = std::min(1.0f, static_cast<float>(unseen_voxels) / 50.0f);
    vp.score = score; // Set the score of the viewpoint
}

std::vector<int> PathPlanner::find_next_toward_furthest_leaf(int start_id, int max_steps) {
    // long-term reasoning and short-term control
    int best_visited_cnt = std::numeric_limits<int>::max();
    int max_depth = -1;
    std::vector<int> best_path;
    std::unordered_set<int> visited;
    std::vector<int> current_path;

    std::function<void(int, int)> dfs = [&](int v, int depth) {
        visited.insert(v);
        current_path.push_back(v);
        const auto& node = GS.global_vertices[v];

        if (node.type == 3) {
            current_path.pop_back();
            return;
        }

        if (node.type == 1) {
            if (node.visited_cnt < best_visited_cnt || 
               (node.visited_cnt == best_visited_cnt && depth > max_depth)) {
                best_visited_cnt = node.visited_cnt;
                max_depth = depth;
                best_path = current_path;
            }
        }

        for (int nb : GS.global_adj[v]) {
            if (!visited.count(nb)) {
                dfs(nb, depth + 1);
            }
        }

        current_path.pop_back();
    };

    dfs(start_id, 0);

    // Truncate to N steps ahead if longer
    if (best_path.size() > static_cast<size_t>(max_steps + 1)) {
        best_path.resize(max_steps + 1);  // +1 to include start_id
    }

    return best_path;
}



/* Voxel Grid Map */

void PathPlanner::global_cloud_handler() {
    for (const auto &pt : local_pts->points) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
        VoxelIndex idx = {
            static_cast<int>(std::floor(pt.x / voxel_size)),
            static_cast<int>(std::floor(pt.y / voxel_size)),
            static_cast<int>(std::floor(pt.z / voxel_size))
        };

        GS.voxel_point_count[idx]++;

        if (GS.voxel_point_count[idx] >= 3) {
            GS.voxels.insert(idx);
        }
    }

    GS.global_pts->clear();
    for (const auto &v : GS.voxels) {
        GS.global_pts->points.emplace_back(
            (v.x + 0.5f) * voxel_size,
            (v.y + 0.5f) * voxel_size,
            (v.z + 0.5f) * voxel_size
        );
    }
}







// void PathPlanner::mst() {
//     double loop_min_length = 20.0;
//     int N_ver = GS.global_vertices.size();
//     if (N_ver == 0 || GS.global_adj.empty()) {
//         RCLCPP_WARN(node_->get_logger(), "Global skeleton is empty, cannot extract relaxed MST.");
//         return;
//     }

//     struct WeightedEdge {
//         int u, v;
//         double weight;
//         double raw_dist;
//         bool operator<(const WeightedEdge& other) const { return weight < other.weight; }
//     };

//     std::vector<WeightedEdge> all_edges;

//     for (int i = 0; i < N_ver; ++i) {
//         GS.global_vertices[i].updated = false;
//         const Eigen::Vector3d& pi = GS.global_vertices[i].position;

//         for (int nb : GS.global_adj[i]) {
//             if (nb <= i) continue;

//             const Eigen::Vector3d& pj = GS.global_vertices[nb].position;
//             Eigen::Vector3d dir = pj - pi;
//             double dist = dir.norm();
//             if (dist < 1e-3) continue;
//             dir.normalize();

//             Eigen::Vector3d smooth_dir = Eigen::Vector3d::Zero();
//             int count = 0;
//             for (int nnb : GS.global_adj[i]) {
//                 if (nnb == nb || nnb == i) continue;
//                 Eigen::Vector3d neighbor_dir = GS.global_vertices[nnb].position - pi;
//                 if (neighbor_dir.norm() > 1e-3) {
//                     smooth_dir += neighbor_dir.normalized();
//                     ++count;
//                 }
//             }

//             if (count > 0) smooth_dir.normalize();
//             double angle_penalty = (count > 0) ? 1.0 + (1.0 - dir.dot(smooth_dir)) : 10.0;

//             double weighted_cost = dist * angle_penalty;
//             all_edges.push_back({i, nb, weighted_cost, dist});
//         }
//     }

//     std::sort(all_edges.begin(), all_edges.end());

//     UnionFind uf(N_ver);
//     std::vector<std::vector<int>> relaxed_adj(N_ver);

//     for (const auto& edge : all_edges) {
//         bool merged = uf.unite(edge.u, edge.v);
//         if (merged) {
//             // standard MST edge
//             relaxed_adj[edge.u].push_back(edge.v);
//             relaxed_adj[edge.v].push_back(edge.u);
//         } else if (edge.raw_dist >= loop_min_length) {
//             // allow long loop edge
//             relaxed_adj[edge.u].push_back(edge.v);
//             relaxed_adj[edge.v].push_back(edge.u);
//         }
//         // else skip short loops
//     }

//     GS.global_adj = std::move(relaxed_adj);
//     graph_decomp();
// }




// std::vector<int> PathPlanner::find_next_toward_furthest_leaf(int start_id) {
//     int best_visited_cnt = std::numeric_limits<int>::max();
//     int max_depth = -1;
//     std::vector<int> best_path;
//     std::unordered_set<int> visited;
//     std::vector<int> current_path;

//     std::function<void(int, int)> dfs = [&](int v, int depth) {
//         visited.insert(v);
//         current_path.push_back(v);
//         const auto& node = GS.global_vertices[v];

//         if (node.type == 3) {
//             current_path.pop_back();
//             return;
//         }

//         if (node.type == 1) {
//             // Prefer lower visited_cnt; break ties using depth
//             if (node.visited_cnt < best_visited_cnt || (node.visited_cnt == best_visited_cnt && depth > max_depth)) {
//                 best_visited_cnt = node.visited_cnt;
//                 max_depth = depth;
//                 best_path = current_path;
//             }
//         }

//         if (v < static_cast<int>(GS.global_vertices.size())) {
//             for (int nb : GS.global_adj[v]) {
//                 if (!visited.count(nb)) {
//                     dfs(nb, depth + 1);
//                 }
//             }
//         }

//         current_path.pop_back();
//     };

//     dfs(start_id, 0);
//     return best_path;
// }