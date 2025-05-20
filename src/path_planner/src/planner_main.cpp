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
    GS.global_seen_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    GS.global_vertices_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

    local_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
    local_vertices.reset(new pcl::PointCloud<pcl::PointXYZ>);

    GP.curr_id = -1;
    GP.curr_branch = -1;
    GS.gskel_size = 0;
}

void PathPlanner::update_skeleton() {
    auto t_start = std::chrono::high_resolution_clock::now();

    int prev_size = (int)GS.global_vertices.size();
    std::vector<int> prev_type;
    prev_type.reserve(prev_size);
    for (int i=0; i<(int)GS.global_vertices.size(); ++i) {
        GS.global_vertices[i].updated = false;
        prev_type[i] = GS.global_vertices[i].type;
    }

    skeleton_increment();
    graph_adj();
    mst();
    vertex_merge();
    prune_branches();
    smooth_vertex_positions();
    extract_branches();

    for (int i=0; i<prev_size; ++i) {
        if (GS.global_vertices[i].type != prev_type[i]) {
            GS.global_vertices[i].updated = true;
        }
    }


    GS.gskel_size = (int)GS.global_vertices.size(); // Update total number of vertices
    // RCLCPP_INFO(node_->get_logger(), "Global Skeleton Size: %d", GS.gskel_size);

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

    viewpoint_sampling();
    viewpoint_filtering();
    viewpoint_connections();

    int path_length = (int)GP.local_path.size();
    if (path_length < MAX_HORIZON) {
        generate_path_test();
    }
    refine_path();


    RCLCPP_INFO(node_->get_logger(), "Current Local Path Lenght: %d", (int)GP.local_path.size());

    RCLCPP_INFO(node_->get_logger(), "Number of Viewpoints Generated: %d", (int)GP.global_vpts.size());

    if (!GS.global_pts->points.empty()) {
        double current_coverage = static_cast<double>(GS.global_seen_cloud->points.size()) / static_cast<double>(GS.global_pts->points.size());
        RCLCPP_INFO(node_->get_logger(), "Current Coverage Percentage: %f", current_coverage);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_elapsed = t_end - t_start;
    RCLCPP_INFO(node_->get_logger(), "[Path Planning] Time Elapsed: %f seconds", t_elapsed.count());
}


/* Skeleton Updating*/
void PathPlanner::skeleton_increment() {
    if (!local_vertices || local_vertices->empty()) return;

    RCLCPP_INFO(node_->get_logger(), "Updating Global Skeleton...");
    std::vector<int> ids_to_delete;

    for (auto &v : GS.prelim_vertices) {
        v.just_approved = false;
    }

    for (auto &pt : local_vertices->points) {
        Eigen::Vector3d ver(pt.x, pt.y, pt.z);
        bool matched = false;

        for (int i=0; i<(int)GS.prelim_vertices.size(); ++i) {
            auto &gver = GS.prelim_vertices[i];

            double sq_dist = (gver.position - ver).squaredNorm();
            if (sq_dist < fuse_dist_th*fuse_dist_th) {
                
                if (gver.freeze) {
                    matched = true;
                    break;
                }

                VertexLKF kf(kf_pn, kf_mn);
                kf.initialize(gver.position, gver.covariance);
                kf.update(ver);

                gver.position = kf.getState();
                gver.covariance = kf.getCovariance();
                gver.obs_count++;
                double trace = gver.covariance.trace();

                if (!gver.conf_check && trace < fuse_conf_th) {
                    gver.conf_check = true;
                    gver.just_approved = true;
                    gver.freeze = true; // Freeze vertex!
                    gver.unconfirmed_check = 0;
                }
                else {
                    gver.unconfirmed_check++;
                }
                
                // Mark ids as invalid and schedule for removal if not confident after a few runs...
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
            new_ver.freeze = false;
            new_ver.just_approved = false;
            new_ver.unconfirmed_check = 0;
            GS.prelim_vertices.push_back(new_ver);
        }
    }

    // Delete points that did not pass the confidence check (reverse indexing for id consistency)
    for (auto it = ids_to_delete.rbegin(); it != ids_to_delete.rend(); ++it) {
        GS.prelim_vertices.erase(GS.prelim_vertices.begin() + *it);
    }

    // Pass confident vertices to the global skeleton...
    GS.new_vertex_indices.clear();
    for (int i=0; i<(int)GS.prelim_vertices.size(); ++i) {
        auto &v = GS.prelim_vertices[i];
        if (v.just_approved) {
            GS.global_vertices.push_back(v);
            GS.global_vertices_cloud->points.emplace_back(v.position.x(), v.position.y(), v.position.z());
            GS.new_vertex_indices.push_back((int)GS.global_vertices.size() - 1);
            v.just_approved = false;
            v.updated = true;
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

    const int K = 10;          // Number of neighbors
    const float max_dist_th = 2.5 * fuse_dist_th; // Max distance for valid edges (meters)

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

void PathPlanner::mst() {
    int N_ver = GS.global_vertices.size();
    if (N_ver == 0 || GS.global_adj.empty()) {
        RCLCPP_WARN(node_->get_logger(), "Global skeleton is empty, cannot extract MST.");
        return;
    }

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
}

void PathPlanner::vertex_merge() {
    int N_ver = GS.global_vertices.size();
    int new_ver = GS.new_vertex_indices.size();
    if (N_ver == 0 || new_ver == 0) return;
    if (static_cast<double>(new_ver) / static_cast<double>(N_ver) > 0.5) return;

    // if (N_ver < 5) return;

    auto is_joint = [&](int idx) {
        return std::find(GS.joints.begin(), GS.joints.end(), idx) != GS.joints.end();
    };

    std::set<int> new_set(GS.new_vertex_indices.begin(), GS.new_vertex_indices.end());
    std::set<int> to_delete;

    for (int j_new : GS.new_vertex_indices) {
        for (int i : GS.global_adj[j_new]) {
            if (i == j_new || to_delete.count(i) || to_delete.count(j_new)) {
                continue;
            }

            bool do_merge = false;

            if (is_joint(i) && is_joint(j_new)) {
                do_merge = true;
                GS.joints.erase(std::remove(GS.joints.begin(), GS.joints.end(), j_new), GS.joints.end());
            }

            double dist = (GS.global_vertices[i].position - GS.global_vertices[j_new].position).norm();
            if (!do_merge && dist < 0.5 * fuse_dist_th) {
                do_merge = true;
            }

            if (!do_merge) continue;

            merge_into(i, j_new); //merge j_new into i

            //Maybe: To generate new viewpoints for moved vertex??
            // GS.global_vertices[i].updated = true;

            to_delete.insert(j_new);
            break;
        }
    }

    if (to_delete.empty()) return;

    std::vector<int> del_idx(to_delete.begin(), to_delete.end());
    std::sort(del_idx.rbegin(), del_idx.rend());

    for (int idx : del_idx) {
        GS.global_vertices.erase(GS.global_vertices.begin() + idx);
        GS.global_vertices_cloud->points.erase(GS.global_vertices_cloud->points.begin() + idx);

        GS.global_adj.erase(GS.global_adj.begin() + idx);
        for (auto &nbrs : GS.global_adj) {
            nbrs.erase(std::remove(nbrs.begin(), nbrs.end(), idx), nbrs.end());
            for (auto &v : nbrs) {
                if (v > idx) --v;
            }
        }

        GS.new_vertex_indices.erase(std::remove(GS.new_vertex_indices.begin(), GS.new_vertex_indices.end(), idx), GS.new_vertex_indices.end());
        for (auto& id : GS.new_vertex_indices) {
            if (id > idx) --id;
        }
    }

    graph_decomp(); // update leafs and joints
}

void PathPlanner::prune_branches() {
    int N_ver = GS.global_vertices.size();
    int new_ver = GS.new_vertex_indices.size();
    if (N_ver == 0 || new_ver == 0) return;
    if (static_cast<double>(new_ver) / static_cast<double>(N_ver) > 0.5) return;
    
    
    std::unordered_set<int> joint_set(GS.joints.begin(), GS.joints.end());
    std::unordered_set<int> new_set(GS.new_vertex_indices.begin(), GS.new_vertex_indices.end());
    std::vector<bool> to_delete(N_ver, false); // Mask for deleting

    for (int leaf : GS.leafs) {
        if (!new_set.count(leaf)) continue; // Not a new leaf

        if (GS.global_adj[leaf].size() != 1) continue; // Sanity check

        int nb = GS.global_adj[leaf][0];

        // If leaf nb is a joint -> remove leaf...
        if (joint_set.count(nb)) {
            to_delete[leaf] = true;
        }
    }

    std::vector<int> del_idx;
    for (int i=0; i<N_ver; ++i) {
        if (to_delete[i]) {
            del_idx.push_back(i);
        }
    }

    if (del_idx.empty()) return;

    std::sort(del_idx.rbegin(), del_idx.rend());

    for (int idx : del_idx) {
        GS.global_vertices.erase(GS.global_vertices.begin() + idx);
        GS.global_vertices_cloud->points.erase(GS.global_vertices_cloud->points.begin() + idx);
        
        GS.global_adj.erase(GS.global_adj.begin() + idx);
        for (auto& nbrs : GS.global_adj) {
            nbrs.erase(std::remove(nbrs.begin(), nbrs.end(), idx), nbrs.end());
            for (auto &v : nbrs) {
                if (v > idx) --v;
            }
        }
        
        GS.new_vertex_indices.erase(std::remove(GS.new_vertex_indices.begin(), GS.new_vertex_indices.end(), idx), GS.new_vertex_indices.end());
        for (auto& id : GS.new_vertex_indices) {
            if (id > idx) --id;
        }
    }

    graph_decomp();
}

void PathPlanner::smooth_vertex_positions() {
    if (GS.global_vertices.empty() || GS.global_adj.empty()) return;

    size_t N = GS.global_vertices.size();
    std::vector<Eigen::Vector3d> new_positions(N);
  
    // 1) compute the new positions, but only for those with smooth_iters_left > 0
    for (size_t i = 0; i < N; ++i) {
      auto &v    = GS.global_vertices[i];
      auto &nbrs = GS.global_adj[i];
  
      // never move joints or leaves
      if (v.type == 1 || v.type == 3 || nbrs.size() < 2) {
        new_positions[i] = v.position;
        continue;
      }
  
      // only smooth if this vertex still has budget
      if (v.smooth_iters_left > 0) {
        Eigen::Vector3d avg = Eigen::Vector3d::Zero();
        for (int j : nbrs) {
          avg += GS.global_vertices[j].position;
        }
        avg /= double(nbrs.size());
  
        double blend = 0.3;  // how strongly to pull toward the average
        new_positions[i] = (1.0 - blend)*v.position + blend*avg;
  
        // consume one smoothing iteration
        --v.smooth_iters_left;
      }
      else {
        new_positions[i] = v.position;
      }
    }
  
    // 2) write them back in place (so you don't lose any other vertex state)
    for (size_t i = 0; i < N; ++i) {
      GS.global_vertices[i].position = new_positions[i];
    }
  
    // 3) update your point cloud
    GS.global_vertices_cloud->clear();
    for (const auto &v : GS.global_vertices) {
      GS.global_vertices_cloud->points.emplace_back(
        v.position.x(), v.position.y(), v.position.z()
      );
    }
}

void PathPlanner::extract_branches() {
    const int min_branch_length = 3;
    int N = (int)GS.global_vertices.size();
    if (N == 0) return;

    GS.branches.clear();

    std::set<std::pair<int,int>> visited_edges;

    // Helper function for determining endpoints
    auto is_endpoint = [&](int vidx) {
        return GS.global_vertices[vidx].type == 1 || GS.global_vertices[vidx].type == 3;
        // return std::find(GS.leafs.begin(), GS.leafs.end(), vidx) != GS.leafs.end() ||
        //        std::find(GS.joints.begin(), GS.joints.end(), vidx) != GS.joints.end();
    };

    // For every endpoint: walk each unvisited nb
    for (int endp : GS.leafs) {
        for (int nb : GS.global_adj[endp]) {
            int a = endp;
            int b = nb;
            if (a > b) std::swap(a,b); // ensure that edge a->b and b->a is interpreted the same
            if (visited_edges.count({a,b})) continue; // edge already walked

            std::vector<int> branch;
            branch.push_back(endp);

            int prev = endp;
            int curr = nb;

            while (true) {
                branch.push_back(curr);

                if (is_endpoint(curr) && curr != endp) break; // stop if another endpoint is hit

                int next = -1;

                for (int nn : GS.global_adj[curr]) {
                    if (nn != prev) {
                        next = nn;
                        break;
                    }
                }

                if (next == -1) break; // dead end

                prev = curr;
                curr = next;
            }

            for (int i=1; i<(int)branch.size(); ++i) {
                int x = branch[i-1];
                int y = branch[i];
                visited_edges.insert({x,y});
            }

            GS.branches.push_back(branch);
        }
    }

    // repeat for joints to catch joint-joint bounded branches
    for (int endp : GS.joints) {
        for (int nb : GS.global_adj[endp]) {
            int a = endp;
            int b = nb;
            if (a > b) std::swap(a,b);
            if (visited_edges.count({a,b})) continue;

            std::vector<int> branch;
            branch.push_back(endp);

            int prev = endp;
            int curr = nb;

            while(true) {
                branch.push_back(curr);
                if (is_endpoint(curr) && curr != endp) break;

                int next = -1;
                for (int nn : GS.global_adj[curr]) {
                    if (nn != prev) {
                        next = nn;
                        break;
                    }
                }

                if (next == -1) break;

                prev = curr;
                curr = next;
            }

            for (int i=1; i<(int)branch.size(); ++i) {
                int x = branch[i-1];
                int y = branch[i];
                visited_edges.insert({x,y});
            }
            GS.branches.push_back(branch);
        }
    }

    // sort branches for ascending order
    std::sort(GS.branches.begin(), GS.branches.end(), 
        [](const std::vector<int>& a, const std::vector<int>&b) {
            return a.size() < b.size();
        }
    );

    // prune small branches
    GS.branches.erase(
        std::remove_if(
            GS.branches.begin(), GS.branches.end(),
            [min_branch_length](const std::vector<int>& branch) {
                return branch.size() < min_branch_length;
            }
        ),
        GS.branches.end()
    );
}

void PathPlanner::graph_decomp() {
    /* Assigns vertex type */
    GS.joints.clear();
    GS.leafs.clear();

    for (int i=0; i<(int)GS.global_vertices.size(); ++i) {
        // int prev_type = GS.global_vertices[i].type;
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

        GS.global_vertices[i].type = new_type;

        // Mark updated vertices if their type is updated in this iteration
        // if (GS.global_vertices[i].updated) continue;
        // GS.global_vertices[i].updated = GS.global_vertices[i].updated || (GS.global_vertices[i].type != prev_type);
    }
}

void PathPlanner::merge_into(int id_keep, int id_del) {
    auto &Vi = GS.global_vertices[id_keep];
    auto &Vj = GS.global_vertices[id_del];

    int tot = Vi.obs_count + Vi.obs_count;
    Vi.position = (Vi.position * Vi.obs_count + Vj.position * Vj.obs_count) / tot;
    Vi.obs_count = tot;
    Vi.visited_cnt = std::max(Vi.visited_cnt, Vj.visited_cnt);

    for (int nb: GS.global_adj[id_del]) {
        if (nb == id_keep) continue;
        auto &nbs_i = GS.global_adj[id_keep];
        if (std::find(nbs_i.begin(), nbs_i.end(), nb) == nbs_i.end()) {
            nbs_i.push_back(nb); // if nb of id_del is not already nb to id_keep
        }

        auto &nbs_nb = GS.global_adj[nb];
        std::replace(nbs_nb.begin(), nbs_nb.end(), id_del, id_keep);
    }
}


/* Viewpoint Generation and Path Planning */

void PathPlanner::viewpoint_sampling() {
    // Collect pointers to old viewpoints for removal
    std::unordered_set<Viewpoint*> to_remove;

    for (int i = 0; i < GS.gskel_size; ++i) {
        auto &vertex = GS.global_vertices[i];

        // Mark as updated if this vertex had no viewpoints
        if (vertex.assigned_vpts.empty()) {
            vertex.updated = true;
        }
        // Skip unchanged or non-relevant vertices
        if (!vertex.updated || vertex.type == 0) 
            continue;

        // Record all currently assigned viewpoints for removal
        for (Viewpoint* vp : vertex.assigned_vpts) {
            to_remove.insert(vp);
        }
        vertex.assigned_vpts.clear();

        // Generate new viewpoints
        std::vector<Viewpoint> new_vpts = generate_viewpoint(i);
        if (new_vpts.empty()) 
            continue;

        // Append them into a std::list for pointer stability
        for (auto &vp : new_vpts) {
            GP.global_vpts.emplace_back(std::move(vp));
        }
    }

    // Remove any old viewpoints that were cleared above
    if (!to_remove.empty()) {
        GP.global_vpts.remove_if([&](const Viewpoint &vp) {
            return to_remove.count(const_cast<Viewpoint*>(&vp)) > 0;
        });
    }
}

void PathPlanner::viewpoint_filtering() {
    if (GS.global_pts->empty()) {
        GP.global_vpts.clear();
        RCLCPP_WARN(node_->get_logger(), "No VoxelMap - Skipping Viewpoint Generation");
        return;
    }

    for (auto &v : GS.global_vertices) {
        v.assigned_vpts.clear();
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> voxel_tree;
    voxel_tree.setInputCloud(GS.global_pts);

    for (auto it = GP.global_vpts.begin(); it != GP.global_vpts.end(); /**/) {
        Viewpoint &vp = *it;

        if (!viewpoint_check(vp, voxel_tree)) {
            it = GP.global_vpts.erase(it);
            continue;
        }

        for (auto jt = std::next(it); jt != GP.global_vpts.end(); /**/) {
            if (viewpoint_similarity(vp, *jt)) {
                vp.position = 0.5 * (vp.position + jt->position);
                vp.orientation = vp.orientation.slerp(0.5, jt->orientation);
                jt = GP.global_vpts.erase(jt);
            }
            else {
                ++jt;
            }
        }

        int vid = vp.corresp_vertex_id;
        GS.global_vertices[vid].assigned_vpts.push_back(&vp);

        ++it; 
    }

    // Clean up potential dangling pointerss
    {
        std::unordered_set<Viewpoint*> valid;
        valid.reserve(GP.global_vpts.size());
        for (auto &vp : GP.global_vpts) valid.insert(&vp);

        auto &path = GP.local_path;
        path.erase(
          std::remove_if(path.begin(), path.end(),
            [&](Viewpoint *p){ return valid.count(p) == 0; }),
          path.end()
        );
    }
}

void PathPlanner::refine_path() {

    std::vector<Viewpoint*> prev_path = GP.local_path;

    // Purge potential dangling pointer due to viewpoints deleted in filtering
    {
        std::unordered_set<Viewpoint*> valid;
        valid.reserve(GP.global_vpts.size());
        for (auto &vp : GP.global_vpts) valid.insert(&vp);

        auto &path = GP.local_path;
        path.erase(
          std::remove_if(path.begin(), path.end(),
            [&](Viewpoint *p){ return valid.count(p) == 0; }),
          path.end()
        );
    }

    std::vector<Viewpoint*> remaining = std::move(GP.local_path);
    GP.local_path.clear();

    Eigen::Vector3d last_pos = pose.position;

    while (!remaining.empty()) {
        // find the nearest reachable viewpoint
        auto best_it = remaining.end();
        double best_d2 = std::numeric_limits<double>::infinity();

        for (auto it = remaining.begin(); it != remaining.end(); ++it) {
            Viewpoint* vp = *it;
            double d2 = (vp->position - last_pos).squaredNorm();
            if (d2 >= best_d2) 
                continue;  // not closer than current best
            // check occlusion
            if (line_obstructed(last_pos, vp->position)) 
                continue;  // skip blocked views
            // this one is both closer *and* unobstructed
            best_d2 = d2;
            best_it = it;
        }

        if (best_it == remaining.end()) {
            // no reachable viewpoint left → stop reordering
            break;
        }

        // append the best one
        Viewpoint* next_vp = *best_it;
        GP.local_path.push_back(next_vp);
        last_pos = next_vp->position;
        remaining.erase(best_it);
    }

    for (auto *vp : remaining) {
        vp->in_path = false;
    }

    // Remove coverage by voxels no longer in path!
    std::unordered_set<Viewpoint*> new_path_set(GP.local_path.begin(), GP.local_path.end());

    for (auto* vp : prev_path) {
        if (new_path_set.count(vp) == 0) {
            for (const auto& idx : vp->covered_voxels) {
                auto it = GS.seen_voxels.find(idx);
                if (it != GS.seen_voxels.end()) {
                    it->second -= 1;
                    if (it->second <= 0)
                        GS.seen_voxels.erase(it);
                }
            }
        }
    }
}

void PathPlanner::viewpoint_connections() {
    for (int vid=0; vid<(int)GS.global_vertices.size(); ++vid) {
        if (!GS.global_vertices[vid].updated) continue;

        auto& vertex = GS.global_vertices[vid];
        auto& vpts = vertex.assigned_vpts;

        for (auto* vp : vpts) vp->adj.clear();

        if (vertex.type == 1) {
            // Handles leaf viewpoint connections
            int N = vpts.size();
            if (N <= 1) continue;

            std::vector<bool> used(N, false);
            std::vector<int> path;
            path.reserve(N);

            int curr = 0;
            used[curr] = true;
            path.push_back(curr);

            for (int step = 1; step<N; ++step) {
                double best_dist = std::numeric_limits<double>::max();
                int best_j = -1;
                for (int j=0; j<N; ++j) {
                    if (used[j]) continue;
                    double dist = (vpts[curr]->position - vpts[j]->position).squaredNorm();
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_j = j;
                    }
                }
                curr = best_j;
                used[curr] = true;
                path.push_back(curr);
            }

            for (int i=0; i+1 < N; ++i) {
                vpts[path[i]]->adj.push_back(vpts[path[i+1]]);
                vpts[path[i+1]]->adj.push_back(vpts[path[i]]);
            }
        }

        else if (vertex.type == 2) {
            // Handles branch-branch, branch-leaf, branch-joint connections
            for (int nb_vid : GS.global_adj[vid]) {
                auto& nb_vertex = GS.global_vertices[nb_vid];
                auto& nb_vpts = nb_vertex.assigned_vpts;

                // Erase any neigbor's adjacency connection to this vertex... (will be rebuild)
                for (auto* nb_vp : nb_vpts) {
                    nb_vp->adj.erase(
                        std::remove_if(
                            nb_vp->adj.begin(), nb_vp->adj.end(),
                            [&](Viewpoint* adj_vp) {
                                return std::find(vpts.begin(), vpts.end(), adj_vp) != vpts.end();
                            }
                        ),
                        nb_vp->adj.end()
                    );
                }

                // branch-branch
                if (nb_vertex.type == 2) {
                    for (Viewpoint* vp_a : vpts) {
                        for (Viewpoint* vp_b : nb_vpts) {
                            Eigen::Vector3d dir_a = vp_a->orientation*Eigen::Vector3d::UnitX();
                            Eigen::Vector3d dir_b = vp_b->orientation*Eigen::Vector3d::UnitX();
                            double dot = dir_a.normalized().dot(dir_b.normalized());
                            if (dot > 0.0) {
                                vp_a->adj.push_back(vp_b);
                                vp_b->adj.push_back(vp_a);
                            }
                        }
                    }
                }

                // branch-leaf
                else if (nb_vertex.type == 1 && !vpts.empty() && !nb_vpts.empty()) {
                    double best_dist = std::numeric_limits<double>::max();
                    int best_a = -1;
                    int best_b = -1;

                    for (int i=0; i<(int)vpts.size(); ++i) {
                        for (int j=0; j<(int)nb_vpts.size(); ++j) {
                            double dist = (vpts[i]->position - nb_vpts[j]->position).squaredNorm();
                            if (dist < best_dist) {
                                best_dist = dist;
                                best_a = i;
                                best_b = j;
                            }
                        }
                    }
                    if (best_a >= 0 && best_b >= 0) {
                        vpts[best_a]->adj.push_back(vpts[best_b]);
                        nb_vpts[best_b]->adj.push_back(vpts[best_a]);
                    }
                }

                // branch-joint
                else if (nb_vertex.type == 3 && !vpts.empty() && !nb_vpts.empty()) {
                    for (Viewpoint* vp_a : vpts) {
                        for (Viewpoint* vp_b : nb_vpts) {
                            if (!line_obstructed(vp_a->position, vp_b->position)) {
                                vp_a->adj.push_back(vp_b);
                                vp_b->adj.push_back(vp_a);
                            }
                        }
                    }
                }
            }
        }

        else if (vertex.type == 3) {
            // Handles leaf-joint connections
            for (int nb_vid : GS.global_adj[vid]) {
                auto& nb_vertex = GS.global_vertices[nb_vid];
                auto& nb_vpts = nb_vertex.assigned_vpts;

                // Erase any neigbor's adjacency connection to this vertex... (will be rebuild)
                for (auto* nb_vp : nb_vpts) {
                    nb_vp->adj.erase(
                        std::remove_if(
                            nb_vp->adj.begin(), nb_vp->adj.end(),
                            [&](Viewpoint* adj_vp) {
                                return std::find(vpts.begin(), vpts.end(), adj_vp) != vpts.end();
                            }
                        ),
                        nb_vp->adj.end()
                    );
                }

                if (nb_vertex.type == 1 && !vpts.empty() && !nb_vpts.empty()) {
                    for (Viewpoint* vp_a : vpts) {
                        double best_dist = std::numeric_limits<double>::max();
                        Viewpoint* best_leaf_vp = nullptr;
                        for (Viewpoint* vp_b : nb_vpts) {
                            if (!line_obstructed(vp_a->position, vp_b->position)) {
                                double dist = (vp_a->position - vp_b->position).squaredNorm();
                                if (dist < best_dist) {
                                    best_dist = dist;
                                    best_leaf_vp = vp_b;
                                }
                            }
                        }
                        if (best_leaf_vp) {
                            vp_a->adj.push_back(best_leaf_vp);
                            best_leaf_vp->adj.push_back(vp_a);
                        }
                    }
                }
            }
        }
    }
}

void PathPlanner::generate_path_test() {
    if (GP.global_vpts.empty()) return;

    // --- Reference drone yaw direction and position
    Eigen::Vector3d fwd = pose.orientation * Eigen::Vector3d::UnitX();
    Eigen::Vector2d ref_dir_xy = fwd.head<2>().normalized();
    Eigen::Vector3d ref_pos = pose.position;

    Viewpoint* best_start = nullptr;
    if (!GP.local_path.empty()) {
        Viewpoint* last = GP.local_path.back();
        if (last && !last->visited) {
            best_start = last;
        }
    }
    else {
        double best_dist = std::numeric_limits<double>::max();
        for (auto& vp : GP.global_vpts) {
            if (vp.visited) continue;
            double dist = (vp.position - ref_pos).squaredNorm();
            if (dist < best_dist) {
                best_dist = dist;
                best_start = &vp;
            }
        }
    }
    
    if (!best_start) return; // no vertex found 

    int slots = MAX_HORIZON - (int)GP.local_path.size();

    std::vector<Viewpoint*> cands;
    cands.reserve(slots);
    vpt_adj_step(best_start, slots, ref_dir_xy, cands);

    if (cands.empty()) {
        RCLCPP_WARN(node_->get_logger(), "No candidate viewpoints found!");
        return;
    }

    // GP.local_path.clear();
    for (auto* vp : cands) {
        if ((int)GP.local_path.size() >= MAX_HORIZON) break;

        GP.local_path.push_back(vp);
        vp->in_path = true;
    }
}

void PathPlanner::vpt_adj_step(Viewpoint* start, int steps, const Eigen::Vector2d& ref_dir_xy, std::vector<Viewpoint*>& out_vps) {
    if (!start) return;

    std::unordered_set<Viewpoint*> visited;
    Viewpoint* current = start;
    Eigen::Vector3d prev_pos = start->position;

    for (int i=0; i<steps && current; ++i) {
        out_vps.push_back(current);
        current->in_path = true;
        visited.insert(current);

        Viewpoint* next = nullptr;

        double best_score = -std::numeric_limits<double>::max();
        for (Viewpoint* nb : current->adj) {
            if (nb->visited || nb->in_path) continue;
            if (visited.count(nb)) continue;

            Eigen::Vector2d dir = (nb->position - prev_pos).head<2>().normalized();
            double score = ref_dir_xy.dot(dir);

            if (score > best_score) {
                best_score = score;
                next = nb;
            }
        }

        current = next;
        if (current) {
            prev_pos = current->position;
        }
    }

}





std::vector<Viewpoint> PathPlanner::generate_viewpoint(int id) {
    double disp_dist = 10;
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
            Eigen::Vector3d u4 = (u1 + u3).normalized();
            Eigen::Vector3d u5 = (u2 + u3).normalized();

            std::vector<Eigen::Vector3d> dirs = {u1, u2, u3, u4, u5};
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
        if (dir_xy.norm() > 0.2) {
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
        vp.corresp_vertex_id = vertex_id;
        viewpoints.push_back(vp);
    }
    return viewpoints;
}

bool PathPlanner::viewpoint_check(const Viewpoint& vp, pcl::KdTreeFLANN<pcl::PointXYZ>& voxel_tree) {
    std::vector<int> ids;
    std::vector<float> dsq;
    pcl::PointXYZ query_point(vp.position.x(), vp.position.y(), vp.position.z());
    if (voxel_tree.radiusSearch(query_point, safe_dist, ids, dsq) > 0) {
        return false;
    }
    return true;
}

bool PathPlanner::viewpoint_similarity(const Viewpoint& a, const Viewpoint& b) {
    return ((a.position - b.position).norm() < viewpoint_merge_dist);
}

void PathPlanner::score_viewpoint(Viewpoint *vp) {
    if (GS.global_pts->empty()) {
        RCLCPP_WARN(node_->get_logger(), "Global Points is empty!");
        vp->score = 0.0;
        vp->covered_voxels.clear();
        return;
    }
    

    // Track what this VP touched
    int new_count   = 0;
    int total_count = 0;
    vp->covered_voxels.clear();

    // Ray‐cast from VP towards each voxel‐center in global_pts
    for (const auto &pt : GS.global_pts->points) {
        Eigen::Vector3d target(pt.x, pt.y, pt.z);
        Eigen::Vector3d vec = target - vp->position;

        double dist = vec.norm();
        if (dist < safe_dist || dist > max_view_dist) {
            continue;
        }

        Eigen::Vector3d dir = vec / dist;
        // Eigen::Vector3d cam_dir = (vp->orientation * Eigen::Vector3d::UnitX()).normalized();
        Eigen::Vector3d cam_dir = vp->orientation.inverse() * vec;
        const double theta_h = std::atan2(cam_dir.y(), cam_dir.x()); // left/right
        const double theta_v = std::atan2(cam_dir.z(), cam_dir.x()); // up/down

        if (std::abs(theta_h) > (fov_h * 0.5 * M_PI / 180.0)) continue;
        if (std::abs(theta_v) > (fov_v * 0.5 * M_PI / 180.0)) continue;

        
        // Target voxel index
        VoxelIndex tgt_idx {
            int(std::floor(target.x() / voxel_size)),
            int(std::floor(target.y() / voxel_size)),
            int(std::floor(target.z() / voxel_size))
        };

        // sample along the ray to find *first* occupied voxel
        bool occ = false;        
        int info_steps = int(2 * dist / (voxel_size * 0.5));

        for (int i = 1; i <= info_steps; ++i) {
            Eigen::Vector3d sample = vp->position + dir * (dist * i / double(info_steps));
            VoxelIndex sample_idx {
                int(std::floor(sample.x() / voxel_size)),
                int(std::floor(sample.y() / voxel_size)),
                int(std::floor(sample.z() / voxel_size))
            };

            if (sample_idx == tgt_idx) break; // arrived at target...

            // If occluded - break
            if (GS.voxels.count(sample_idx) > 0) {
                occ = true;
                break;
            }
        }

        if (!occ && GS.voxels.count(tgt_idx) > 0) {
            ++total_count;
            vp->covered_voxels.push_back(tgt_idx); // Log seen voxels by this viewpoint
            auto it = GS.seen_voxels.find(tgt_idx);
            if (it == GS.seen_voxels.end() || it->second < 1) {
                ++new_count;
            }
        }
    }

    // ratio of new to seen
    vp->score = (total_count > 0)
              ? double(new_count) / double(total_count)
              : 0.0;
}

void PathPlanner::dfs_collect(int vertex_id, int& slots_left, Eigen::Vector2d& ref_dir_xy, Eigen::Vector3d& last_pos, std::vector<Viewpoint*>& out_vps, std::unordered_set<int>& seen) {
    if (slots_left <= 0) return;
    if (!seen.insert(vertex_id).second) return;  // already here this run

    const double MAX_JUMP2 = MAX_JUMP * MAX_JUMP;
    const double cos120 = std::cos(2.0 * M_PI / 3.0);

    auto  &vertex = GS.global_vertices[vertex_id];
    vertex.visited_cnt++;
    
    // --- Gather valid viewpoints
    std::vector<Viewpoint*> cands;
    cands.reserve(vertex.assigned_vpts.size());
    for (auto *vp : vertex.assigned_vpts) {
        if (vp->in_path || vp->visited) continue; // Check if already in path 

        Eigen::Vector2d d2 = (vp->orientation * Eigen::Vector3d::UnitX()).head<2>().normalized();
        if (ref_dir_xy.dot(d2) < cos120) continue; // check if sufficiently similar yaw

        if ((int)GP.local_path.size() > 1 && (vp->position - last_pos).squaredNorm() > MAX_JUMP2) continue; // check if too far between vpts

        if (line_obstructed(last_pos, vp->position)) continue; // check if line between two vpts are obstructed by voxels

        score_viewpoint(vp);
        if (vp->score < 0.3) {
            RCLCPP_WARN(node_->get_logger(), "[DFS COLLECT] SCORE TOO LOW");
            continue;
        }
        cands.push_back(vp);
    }
    

    // --- Sort by best alignment then nearest distance
    std::sort(cands.begin(), cands.end(),
        [&](Viewpoint* a, Viewpoint* b) {
            Eigen::Vector2d a2 = (a->orientation * Eigen::Vector3d::UnitX()).head<2>().normalized();
            Eigen::Vector2d b2 = (b->orientation * Eigen::Vector3d::UnitX()).head<2>().normalized();
            double ca = ref_dir_xy.dot(a2);
            double cb = ref_dir_xy.dot(b2);
            if (ca != cb) return ca > cb;
            double da = (a->position - last_pos).squaredNorm();
            double db = (b->position - last_pos).squaredNorm();
            return da < db;
        });

    // --- Accept up to slots_left
    for (auto *vp : cands) {
        if (slots_left <= 0) break;
        vp->in_path = true;
        out_vps.push_back(vp);
        --slots_left;
        last_pos = vp->position;
        Eigen::Vector3d d3 = vp->orientation * Eigen::Vector3d::UnitX();
        ref_dir_xy = d3.head<2>().normalized();
    }

    if (slots_left <= 0) return;

    // Sort neighbor after visitation amount
    std::vector<int> nbs = GS.global_adj[vertex_id];
    std::sort(nbs.begin(), nbs.end(), 
        [&](int a, int b){
            return GS.global_vertices[a].visited_cnt < GS.global_vertices[b].visited_cnt;
        });

    // --- Recurse into neighbors
    for (int nb : nbs) {
        // if (nb == parent_id) continue; // Never trace backwards to parent
        dfs_collect(nb, slots_left, ref_dir_xy, last_pos, out_vps, seen);
        if (slots_left <= 0) break;
    }
}

bool PathPlanner::line_obstructed(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2) {
    /* Draw a line between two points and checks if it is obstructed by the voxel grid */
    const double step = voxel_size;
    Eigen::Vector3d dir = (p2 - p1).normalized();
    double len = (p2 - p1).norm();
    int n = static_cast<int>(std::ceil(len/step));
    for (int k=1; k<=n; ++k) {
        Eigen::Vector3d sample = p1 + dir * (k * step);
        VoxelIndex idx = {
            static_cast<int>(std::floor(sample.x() / voxel_size)),
            static_cast<int>(std::floor(sample.y() / voxel_size)),
            static_cast<int>(std::floor(sample.z() / voxel_size))};
        if (GS.voxels.count(idx)) {
            return true;
        }
    }
    return false;
}

/* Voxel Grid Map */
void PathPlanner::global_cloud_handler() {
    // Maybe upgrade from hash-map to octree structure??
    for (const auto &pt : local_pts->points) {
        if (pt.z < gnd_th) continue;
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

void PathPlanner::update_seen_cloud(Viewpoint *vp) {
    if (!GS.global_seen_cloud) {
        GS.global_seen_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    };

    for (const auto& idx : vp->covered_voxels) {
        if (GS.global_seen_voxels.insert(idx).second) {
            GS.global_seen_cloud->points.emplace_back(
                (idx.x + 0.5f) * voxel_size,
                (idx.y + 0.5f) * voxel_size,
                (idx.z + 0.5f) * voxel_size);
        }
    }
}






// void PathPlanner::generate_path() {
//     if (GP.global_vpts.empty()) return;

//     int slots = MAX_HORIZON - (int)GP.local_path.size();
//     if (slots <= 0) return; // Return if too many points in current path...
    
//     if (GP.curr_branch < 0) {
//         // pick longest branch
//         GP.curr_branch = 0;
//     }

//     const auto& current_branch = GS.branches[GP.curr_branch];
    
//     if (get_branch_vertex) {
//         int min_idx = -1;
//         double min_dist = std::numeric_limits<double>::max();
//         for (int vidx : current_branch) {
//             const auto& v = GS.global_vertices[vidx];
//             double d = (v.position - pose.position).squaredNorm();
//             if (d < min_dist) {
//                 min_dist = d;
//                 min_idx = vidx;
//             }
//         }

//         if (min_idx < 0) {
//             RCLCPP_WARN(node_->get_logger(), "No vertex found on selected branch!");
//             get_branch_vertex = true;
//             return;
//         }

//         get_branch_vertex = false;
//         GP.curr_id = min_idx;
//     }

//     auto branch_it = std::find(current_branch.begin(), current_branch.end(), GP.curr_id);
//     if (branch_it == current_branch.end()) {
//         RCLCPP_WARN(node_->get_logger(), "Current vertex not found in branch!");
//         get_branch_vertex = true;
//         return;
//     }

//     const double MAX_JUMP2 = MAX_JUMP * MAX_JUMP;
//     const double cos120 = std::cos(2.0 * M_PI / 3.0);

//     // --- Reference direction in XY-Plane
//     Eigen::Vector3d fwd = pose.orientation * Eigen::Vector3d::UnitX();
//     Eigen::Vector2d ref_dir_xy = fwd.head<2>().normalized();
//     Eigen::Vector3d last_pos = pose.position;

//     if (!GP.local_path.empty()) {
//         auto *last = GP.local_path.back();
//         Eigen::Vector3d lv = last->orientation * Eigen::Vector3d::UnitX();
//         ref_dir_xy = lv.head<2>().normalized();
//         last_pos = last->position;
//     }

//     // --- Walk along the current branch
//     std::vector<Viewpoint*> new_cands;
//     int slots_left = slots;

//     for (; branch_it != current_branch.end() && slots_left > 0; ++branch_it) {
//         int vidx = *branch_it;
//         auto& vertex = GS.global_vertices[vidx];
//         vertex.visited_cnt++;
        
//         std::vector<Viewpoint*> cands;
//         cands.reserve(vertex.assigned_vpts.size());
//         for (auto *vp : vertex.assigned_vpts) {
//             if (vp->in_path || vp->visited) continue;

//             Eigen::Vector2d d2 = (vp->orientation * Eigen::Vector3d::UnitX()).head<2>().normalized();
//             if (ref_dir_xy.dot(d2) < cos120) continue;

//             if (!new_cands.empty() && (vp->position - last_pos).squaredNorm() > MAX_JUMP2) continue;

//             if (line_obstructed(last_pos, vp->position)) continue;

//             // score_viewpoint(vp);
//             // if (vp->score < 0.2) continue;
//             cands.push_back(vp);
//         }

//         for (auto* vp : cands) {
//             if (slots_left <= 0) break;
//             vp->in_path = true;
//             new_cands.push_back(vp);
//             --slots_left;
//             last_pos = vp->position;
//             ref_dir_xy = (vp->orientation * Eigen::Vector3d::UnitX()).head<2>().normalized();
//         }
//     }

//     for (auto* vp : new_cands) {
//         GP.local_path.push_back(vp);
//         for (const auto& idx : vp->covered_voxels) GS.seen_voxels[idx] += 1;
//     }

//     if (!GP.local_path.empty()) {
//         GP.curr_id = GP.local_path.back()->corresp_vertex_id;
//     }

// }



// std::vector<int> PathPlanner::find_next_toward_furthest_leaf(int start_id, int max_steps) {
//     // A single state in the priority queue:
//     struct State {
//         int vid;           // current vertex
//         int parent;        // predecessor vid
//         int depth;         // steps from start
//         int visited_cnt;   // GS.global_vertices[vid].visited_cnt
//         double score;      // visited_cnt - alpha * depth
//     };
//     // lower score = higher priority
//     struct Compare {
//         bool operator()(State const &a, State const &b) const {
//             return a.score > b.score;
//         }
//     };

//     // Bookkeeping: back‐pointers and depths
//     std::unordered_map<int,int> parent_of;
//     std::unordered_map<int,int> depth_of;
//     std::unordered_set<int>    closed; 
//     std::priority_queue<State, std::vector<State>, Compare> pq;

//     // Initialize
//     double alpha = 0.1;  // weight for depth vs. visited_cnt, tune as needed
//     int vc0 = GS.global_vertices[start_id].visited_cnt;
//     parent_of[start_id] = -1;
//     depth_of [start_id] =  0;
//     closed.insert(start_id);
//     pq.push(State{start_id, -1, 0, vc0, vc0 - alpha * 0});

//     State best = pq.top();

//     while (!pq.empty()) {
//         State cur = pq.top(); pq.pop();
//         best = cur;

//         // Stop if we've reached a leaf or exceeded lookahead
//         const auto &nbrs = GS.global_adj[cur.vid];
//         if (cur.depth >= max_steps || nbrs.size() == 1) {
//             break;
//         }

//         // Expand neighbors
//         for (int nb : nbrs) {
//             if (closed.count(nb)) continue;
//             int d    = cur.depth + 1;
//             int vc   = GS.global_vertices[nb].visited_cnt;
//             double sc = vc - alpha * d;

//             parent_of[nb] = cur.vid;
//             depth_of [nb] = d;
//             closed.insert(nb);
//             pq.push(State{nb, cur.vid, d, vc, sc});
//         }
//     }

//     // Reconstruct path from best.vid back to start_id
//     std::vector<int> path;
//     int walker = best.vid;
//     while (walker != -1) {
//         path.push_back(walker);
//         walker = parent_of[walker];
//     }
//     std::reverse(path.begin(), path.end());
//     return path;
// }








// std::vector<int> PathPlanner::find_next_toward_furthest_leaf(int start_id, int max_steps) {
//     // long-term reasoning and short-term control
//     int best_visited_cnt = std::numeric_limits<int>::max();
//     int max_depth = -1;
//     std::vector<int> best_path;
//     std::unordered_set<int> visited;
//     std::vector<int> current_path;

//     max_steps = 0;

//     std::function<void(int, int)> dfs = [&](int v, int depth) {
//         visited.insert(v);
//         current_path.push_back(v);
//         const auto& node = GS.global_vertices[v];

//         // if (node.type == 3) {
//         //     current_path.pop_back();
//         //     return;
//         // }

//         if (node.type == 1) {
//             if (node.visited_cnt < best_visited_cnt || 
//                (node.visited_cnt == best_visited_cnt && depth > max_depth)) {
//                 best_visited_cnt = node.visited_cnt;
//                 max_depth = depth;
//                 best_path = current_path;
//             }
//         }

//         for (int nb : GS.global_adj[v]) {
//             if (!visited.count(nb)) {
//                 dfs(nb, depth + 1);
//             }
//         }

//         current_path.pop_back();
//     };

//     dfs(start_id, 0);

//     // Truncate to N steps ahead if longer
//     // if (best_path.size() > static_cast<size_t>(max_steps + 1)) {
//     //     best_path.resize(max_steps + 1);  // +1 to include start_id
//     // }

//     return best_path;
// }




// if (GS.global_vertices.empty() || GP.global_vpts.empty()) return;

//     int diff = horizon_max - GP.local_path.size();
//     if (diff == 0) return;

//     // --- Find closest skeleton vertex to drone to initialize
//     if (first_plan || GP.local_path.empty()) {
//         pcl::PointXYZ current_pos(pose.position.x(), pose.position.y(), pose.position.z());
//         pcl::KdTreeFLANN<pcl::PointXYZ> vertex_tree;
//         vertex_tree.setInputCloud(GS.global_vertices_cloud);
//         std::vector<int> nearest_id(1);
//         std::vector<float> sq_dist(1);
//         if (vertex_tree.nearestKSearch(current_pos, 1, nearest_id, sq_dist) < 1) {
//             RCLCPP_WARN(node_->get_logger(), "No nearby skeleton vertex found...");
//             return;
//         }
//         GP.curr_id = nearest_id[0];
//         first_plan = false;
//     }

//     // --- Plan forward path
//     const int max_steps = 10;
//     std::vector<int> local_path_ids = find_next_toward_furthest_leaf(GP.curr_id, max_steps);
//     // std::vector<int> local_path_ids = find_next_toward_furthest_leaf(GP.curr_id, diff);
//     if (local_path_ids.empty()) {
//         RCLCPP_WARN(node_->get_logger(), "Could not generate local path...");
//         return;
//     }
    
//     // --- Get reference direction
//     Eigen::Vector3d ref_dir = pose.orientation * Eigen::Vector3d::UnitX();
//     Eigen::Vector2d ref_dir_xy = ref_dir.head<2>().normalized();
    
//     // --- Reference direction for the next viewpoint appended to the path
//     if (!GP.local_path.empty()) {
//         ref_dir = GP.local_path.back()->orientation * Eigen::Vector3d::UnitX();
//         ref_dir_xy = ref_dir.head<2>().normalized();
//     }

//     const double cos_120deg = std::cos(2 * M_PI / 3.0);
    
//     // --- Score and filter valid candidate viewpoints
//     std::vector<Viewpoint*> candidate_vpts; // Construc a vector of pointers to the viewpoints!
//     for (int id : local_path_ids) {
//         SkeletonVertex &current_vertex = GS.global_vertices[id];
//         for (auto& vp : current_vertex.assigned_vpts) {
//             if (vp->in_path || vp->visited) continue;
            
//             Eigen::Vector3d vp_dir = vp->orientation * Eigen::Vector3d::UnitX();
//             Eigen::Vector2d vp_dir_xy = vp_dir.head<2>().normalized();
//             double cos_angle = ref_dir_xy.dot(vp_dir_xy);
//             if (cos_angle < cos_120deg) continue;
//             candidate_vpts.push_back(vp); // address of vp
//             ref_dir_xy = vp_dir_xy;
//         }
//     }

//     if (candidate_vpts.empty()) {
//         RCLCPP_WARN(node_->get_logger(), "No candidate viewpoints!");
//         return;
//     }
    
//     // Sort ascending by squared distance
//     Eigen::Vector3d curr_pos = pose.position;
//     std::sort(candidate_vpts.begin(), candidate_vpts.end(),
//             [&](const Viewpoint *a, const Viewpoint *b) {
//         return (a->position - curr_pos).squaredNorm() < (b->position - curr_pos).squaredNorm();
//     });

//     // Insert allowed n new viewpoints
//     int n = std::min(diff, (int)candidate_vpts.size());
//     for (int i=0; i<n; ++i) {
//         Viewpoint *vp = candidate_vpts[i];
//         GP.local_path.push_back(vp);
//         vp->in_path=true;
//     }

//     // Set current point for path incrementation to the end of the current path
//     if (!GP.local_path.empty()) {
//         GP.curr_id = GP.local_path.back()->corresp_vertex_id;
//     }