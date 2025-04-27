#ifndef PLANNER_MAIN_
#define PLANNER_MAIN_

#include <rclcpp/rclcpp.hpp>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Core>

struct GlobalSkeleton {
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_pts;
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_vertices;
};

class PathPlanner {
public:
    PathPlanner(rclcpp::Node::SharedPtr node);
    void init();
    void main();
    void update_skeleton();

    /* Data */
    GlobalSkeleton GS;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_pts;
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_vertices;

private:
    rclcpp::Node::SharedPtr node_;

    /* Functions */
    void lowpass_update(int idx, const pcl::PointXYZ& new_pt);
    void add_new_vertex(const pcl::PointXYZ& pt);

    /* Data */
    pcl::KdTreeFLANN<pcl::PointXYZ> gskel_tree;

    /* Params */
    double fuse_dist_th = 3.0;
    double fuse_alpha = 0.3;

    double kf_pn = 0.001; // LKF process noise
    double kf_mn = 0.1; // LKF measurement noise
    double kf_fuse_th = 1.0; // Distance threshold for fusing vertices
};

#endif //PLANNER_MAIN_