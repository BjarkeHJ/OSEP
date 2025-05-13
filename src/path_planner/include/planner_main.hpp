#ifndef PLANNER_MAIN_
#define PLANNER_MAIN_

#include "kalman_vertex_fusion.hpp"

#include <algorithm>
#include <queue>
#include <rclcpp/rclcpp.hpp>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Core>


struct VoxelIndex {
    int x, y, z;
    bool operator==(const VoxelIndex &other) const {
        return std::tie(x, y, z) == std::tie(other.x, other.y, other.z);
    }

    bool operator<(const VoxelIndex &other) const {
        return std::tie(x, y, z) < std::tie(other.x, other.y, other.z);
    }
};

struct VoxelIndexHash {
    std::size_t operator()(const VoxelIndex &k) const {
        return std::hash<int>()(k.x) ^ std::hash<int>()(k.y << 1) ^ std::hash<int>()(k.z << 2);
    }
};

struct Edge
{
    int u, v; // Vertex indices of the edge
    double w; // Lenght / weight of the edge
    bool operator<(const Edge &other) const {
        return w < other.w;
    } // comparison of the weight of this edge with another
};

struct UnionFind {
    std::vector<int> parent;  // parent[i] tells you who the parent of node i is

    // Constructor: initially, every node is its own parent (disconnected)
    UnionFind(int n) : parent(n) {
        for (int i = 0; i < n; ++i) parent[i] = i;
    }

    // Find the "representative" of the component that x belongs to
    int find(int x) {
        // If x is not its own parent, follow the chain recursively
        if (parent[x] != x)
            parent[x] = find(parent[x]);  // Path compression for speed
        return parent[x];
    }

    // Try to merge the sets that x and y belong to
    bool unite(int x, int y) {
        int rx = find(x);  // root of x
        int ry = find(y);  // root of y
        if (rx == ry) return false;  // Already in the same set — adding this edge would create a cycle
        parent[ry] = rx;  // Union: make one root the parent of the other
        return true;
    }
};

struct SkeletonVertex {
    Eigen::Vector3d position;
    Eigen::Matrix3d covariance;
    int obs_count = 0;
    int unconfirmed_check = 0;
    bool conf_check = false;
    bool freeze = false;
    
    int type = -1; // "0: invalid", "1: leaf", "2: branch", "3: joint" 
    bool updated = false;
    int visited_cnt = 0;
    int invalid = false; // If no proper viewpoint can be generated??
};

struct DronePose {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
};

struct Viewpoint {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    std::set<VoxelIndex> visible_voxels;
    int corresp_vertex_id;
    double score = 0.0f;
    bool locked = false;
    bool visited = false;
};

struct GlobalSkeleton {
    std::unordered_set<VoxelIndex, VoxelIndexHash> voxels;
    std::unordered_map<VoxelIndex, int, VoxelIndexHash> voxel_point_count;
    std::unordered_map<VoxelIndex, int, VoxelIndexHash> seen_voxels;
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_pts;

    std::vector<SkeletonVertex> prelim_vertices;
    std::vector<SkeletonVertex> global_vertices;
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_vertices_cloud; // For visualizing

    std::vector<int> joints;
    std::vector<int> leafs;

    std::vector<std::vector<int>> global_adj;

    int gskel_size;
};

struct GlobalPath {
    bool started = false;
    int vertex_start_id;
    int curr_id;

    std::vector<int> vertex_nbs_id;
    Viewpoint start;

    std::vector<Viewpoint> global_vpts;
    std::vector<Viewpoint> local_path; // Current local path being published
    std::vector<Viewpoint> traced_path; // Add only when drone reaches the vpt
};

class PathPlanner {
public:
    PathPlanner(rclcpp::Node::SharedPtr node);

    void init();
    void plan_path();
    void update_skeleton();
        
    /* Occupancy */
    void global_cloud_handler();
    void mark_viewpoint_visited(const Viewpoint& reached_vp);

    /* Data */
    GlobalSkeleton GS;
    GlobalPath GP;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_pts;
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_vertices;

    DronePose pose;

    bool planning_started = false;

private:
    rclcpp::Node::SharedPtr node_;

    /* Updating Skeleton */
    void skeleton_increment();
    void graph_adj();
    void mst();
    void vertex_merge();
    void prune_branches();
    
    void smooth_vertex_positions();
    void graph_decomp();
    
    /* Waypoint Generation and PathPlanning*/
    void viewpoint_sampling();
    void viewpoint_filtering();
    
    void generate_path();
    std::vector<Viewpoint> generate_viewpoint(int id);
    std::vector<Viewpoint> vp_sample(const Eigen::Vector3d& origin, const std::vector<Eigen::Vector3d>& directions, double disp_distance, int vertex_id);
    bool viewpoint_check(const Viewpoint& vp, pcl::KdTreeFLANN<pcl::PointXYZ> voxel_tree);
    bool viewpoint_similarity(const Viewpoint& a, const Viewpoint& b);


    void score_viewpoint(Viewpoint &vp);    
    std::vector<int> find_next_toward_furthest_leaf(int start_id, int max_steps);


    /* Data */
    bool planner_flag = false;
    bool first_plan = true;
    int N_new_vers; // store number of new vertices for each iteration...


    /* Params */
    int max_obs_wo_conf = 3; // Maximum number of iters without passing conf check before discarding...
    double fuse_dist_th = 2.0;
    double fuse_conf_th = 0.5;
    double kf_pn = 0.00001;
    double kf_mn = 0.1;

    double voxel_size = 1.0;
    double fov_h = 90;
    double fov_v = 60;
    double max_view_dist = 25;
    double min_view_dist = 8;
    double viewpoint_merge_dist = 3.0;
};

#endif //PLANNER_MAIN_