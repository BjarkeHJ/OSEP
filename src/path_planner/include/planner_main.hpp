#ifndef PLANNER_MAIN_
#define PLANNER_MAIN_

#include "kalman_vertex_fusion.hpp"

#include <algorithm>
#include <rclcpp/rclcpp.hpp>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Core>


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

struct GlobalSkeleton {
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_pts;
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_vertices_cloud;
    std::vector<SkeletonVertex> global_vertices;

    std::vector<std::vector<int>> global_adj;
    std::vector<bool> visited;
};

class PathPlanner {
public:
    PathPlanner(rclcpp::Node::SharedPtr node);
    void init();
    void main();
    void update_skeleton();
    void graph_adj();
    void mst();
    void clean_skeleton_graph();
    void select_waypoint();

    /* Data */
    GlobalSkeleton GS;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_pts;
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_vertices;

private:
    rclcpp::Node::SharedPtr node_;

    /* Functions */
    
    /* Data */
    pcl::KdTreeFLANN<pcl::PointXYZ> gskel_tree;

    /* Params */
    int max_obs_wo_conf = 3; // Maximum number of iters without passing conf check before discarding...
    double fuse_dist_th = 2.0;
    double fuse_conf_th = 0.5;
    double kf_pn = 0.0001;
    double kf_mn = 0.1;




    double fuse_alpha = 0.5;
};

#endif //PLANNER_MAIN_