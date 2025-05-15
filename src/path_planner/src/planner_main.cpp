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
    prune_branches();
    smooth_vertex_positions();

    N_new_vers = (int)GS.global_vertices.size() - GS.gskel_size; // Number of new vertices added
    // RCLCPP_INFO(node_->get_logger(), "New Vertices: %d", N_new_vers);

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

    // RUN GENERAL PATH PLANNING SELECTION ECT...

    // If: only first (start) vertex is found
    // Do: Fly towards it
    // Manage adjusted points with a flag
    // if adjust_flag: -> adjusted points are recieved and 

    // Wait for number of vertices before planning?
    viewpoint_sampling();
    viewpoint_filtering();
    int path_length = (int)GP.local_path.size();
    if (path_length < horizon_max) {
        generate_path();
    }
    refine_path();

    RCLCPP_INFO(node_->get_logger(), "Current Local Path Lenght: %d", (int)GP.local_path.size());

    RCLCPP_INFO(node_->get_logger(), "Number of Viewpoints Generated: %d", (int)GP.global_vpts.size());

    N_new_vers = 0; // Reset

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

        GS.global_vertices[i].type = new_type;

        // Mark updated vertices if their type is updated in this iteration
        if (GS.global_vertices[i].updated) continue;
        GS.global_vertices[i].updated = GS.global_vertices[i].updated || (GS.global_vertices[i].type != prev_type);
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
    // if (N_new_vers == 0) return;

    for (int i=0; i<GS.gskel_size; ++i) {
        if (!GS.global_vertices[i].updated || GS.global_vertices[i].type == 0) continue;

        std::vector<Viewpoint> vpts = generate_viewpoint(i);
        if (vpts.empty()) continue;

        for (auto &vp : vpts) {
            GP.global_vpts.push_back(std::move(vp));
        }
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
}

void PathPlanner::generate_path() {
    if (GS.global_vertices.empty() || GP.global_vpts.empty()) return;

    int diff = horizon_max - GP.local_path.size();
    if (diff == 0) return;

    // --- Find closest skeleton vertex to drone to initialize
    if (first_plan || GP.local_path.empty()) {
        pcl::PointXYZ current_pos(pose.position.x(), pose.position.y(), pose.position.z());
        pcl::KdTreeFLANN<pcl::PointXYZ> vertex_tree;
        vertex_tree.setInputCloud(GS.global_vertices_cloud);
        std::vector<int> nearest_id(1);
        std::vector<float> sq_dist(1);
        if (vertex_tree.nearestKSearch(current_pos, 1, nearest_id, sq_dist) < 1) {
            RCLCPP_WARN(node_->get_logger(), "No nearby skeleton vertex found...");
            return;
        }
        GP.curr_id = nearest_id[0];
        first_plan = false;
    }

    // --- Plan forward path
    const int max_steps = 10;
    std::vector<int> local_path_ids = find_next_toward_furthest_leaf(GP.curr_id, max_steps);
    // std::vector<int> local_path_ids = find_next_toward_furthest_leaf(GP.curr_id, diff);
    if (local_path_ids.empty()) {
        RCLCPP_WARN(node_->get_logger(), "Could not generate local path...");
        return;
    }
    
    // --- Get reference direction
    Eigen::Vector3d ref_dir = pose.orientation * Eigen::Vector3d::UnitX();
    Eigen::Vector2d ref_dir_xy = ref_dir.head<2>().normalized();
    
    // --- Reference direction for the next viewpoint appended to the path
    if (!GP.local_path.empty()) {
        ref_dir = GP.local_path.back()->orientation * Eigen::Vector3d::UnitX();
        ref_dir_xy = ref_dir.head<2>().normalized();
    }

    const double cos_120deg = std::cos(2 * M_PI / 3.0);
    
    // --- Score and filter valid candidate viewpoints
    std::vector<Viewpoint*> candidate_vpts; // Construc a vector of pointers to the viewpoints!
    for (int id : local_path_ids) {
        SkeletonVertex &current_vertex = GS.global_vertices[id];
        for (auto& vp : current_vertex.assigned_vpts) {
            if (vp->in_path || vp->visited) continue;
            
            Eigen::Vector3d vp_dir = vp->orientation * Eigen::Vector3d::UnitX();
            Eigen::Vector2d vp_dir_xy = vp_dir.head<2>().normalized();
            double cos_angle = ref_dir_xy.dot(vp_dir_xy);
            if (cos_angle < cos_120deg) continue;
            candidate_vpts.push_back(vp); // address of vp
            ref_dir_xy = vp_dir_xy;
        }
    }

    if (candidate_vpts.empty()) {
        RCLCPP_WARN(node_->get_logger(), "No candidate viewpoints!");
        return;
    }
    
    // Sort ascending by squared distance
    Eigen::Vector3d curr_pos = pose.position;
    std::sort(candidate_vpts.begin(), candidate_vpts.end(),
            [&](const Viewpoint *a, const Viewpoint *b) {
        return (a->position - curr_pos).squaredNorm() < (b->position - curr_pos).squaredNorm();
    });

    // Insert allowed n new viewpoints
    int n = std::min(diff, (int)candidate_vpts.size());
    for (int i=0; i<n; ++i) {
        Viewpoint *vp = candidate_vpts[i];
        GP.local_path.push_back(vp);
        vp->in_path=true;
    }

    // Set current point for path incrementation to the end of the current path
    if (!GP.local_path.empty()) {
        GP.curr_id = GP.local_path.back()->corresp_vertex_id;
    }
}

void PathPlanner::refine_path() {
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
    Eigen::Vector3d last_pos = pose.position;  // start from the drone

    while (!remaining.empty()) {
        // find the remaining viewpoint closest to last_pos
        auto best_it = std::min_element(
            remaining.begin(), remaining.end(),
            [&](Viewpoint* a, Viewpoint* b) {
                return (a->position - last_pos).squaredNorm()
                     < (b->position - last_pos).squaredNorm();
            });
        Viewpoint* next_vp = *best_it;

        // append it, update last_pos, and remove from pool
        GP.local_path.push_back(next_vp);
        last_pos = next_vp->position;
        remaining.erase(best_it);
    }
}



std::vector<Viewpoint> PathPlanner::generate_viewpoint(int id) {
    double disp_dist = 7;
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
            std::vector<Eigen::Vector3d> dirs = {u1, u2, u3*2.0};
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
        if (dist < safe_dist || dist > max_view_dist) continue;

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
    // A single state in the priority queue:
    struct State {
        int vid;           // current vertex
        int parent;        // predecessor vid
        int depth;         // steps from start
        int visited_cnt;   // GS.global_vertices[vid].visited_cnt
        double score;      // visited_cnt - alpha * depth
    };
    // lower score = higher priority
    struct Compare {
        bool operator()(State const &a, State const &b) const {
            return a.score > b.score;
        }
    };

    // Bookkeeping: back‐pointers and depths
    std::unordered_map<int,int> parent_of;
    std::unordered_map<int,int> depth_of;
    std::unordered_set<int>    closed; 
    std::priority_queue<State, std::vector<State>, Compare> pq;

    // Initialize
    double alpha = 0.1;  // weight for depth vs. visited_cnt, tune as needed
    int vc0 = GS.global_vertices[start_id].visited_cnt;
    parent_of[start_id] = -1;
    depth_of [start_id] =  0;
    closed.insert(start_id);
    pq.push(State{start_id, -1, 0, vc0, vc0 - alpha * 0});

    State best = pq.top();

    while (!pq.empty()) {
        State cur = pq.top(); pq.pop();
        best = cur;

        // Stop if we've reached a leaf or exceeded lookahead
        const auto &nbrs = GS.global_adj[cur.vid];
        if (cur.depth >= max_steps || nbrs.size() == 1) {
            break;
        }

        // Expand neighbors
        for (int nb : nbrs) {
            if (closed.count(nb)) continue;
            int d    = cur.depth + 1;
            int vc   = GS.global_vertices[nb].visited_cnt;
            double sc = vc - alpha * d;

            parent_of[nb] = cur.vid;
            depth_of [nb] = d;
            closed.insert(nb);
            pq.push(State{nb, cur.vid, d, vc, sc});
        }
    }

    // Reconstruct path from best.vid back to start_id
    std::vector<int> path;
    int walker = best.vid;
    while (walker != -1) {
        path.push_back(walker);
        walker = parent_of[walker];
    }
    std::reverse(path.begin(), path.end());
    return path;
}

/* Voxel Grid Map */
void PathPlanner::global_cloud_handler() {
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

// std::vector<int> PathPlanner::find_next_toward_furthest_leaf(int start_id, int max_steps) {
//     // long-term reasoning and short-term control
//     int best_visited_cnt = std::numeric_limits<int>::max();
//     int max_depth = -1;
//     std::vector<int> best_path;
//     std::unordered_set<int> visited;
//     std::vector<int> current_path;

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
//     if (best_path.size() > static_cast<size_t>(max_steps + 1)) {
//         best_path.resize(max_steps + 1);  // +1 to include start_id
//     }

//     return best_path;
// }

