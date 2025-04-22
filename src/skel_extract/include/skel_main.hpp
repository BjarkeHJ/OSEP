#ifndef __main__
#define __main__

#include <rclcpp/rclcpp.hpp>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/features/normal_3d.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>

#include <Eigen/Core>

struct SkeletonStructure {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pts_;
    pcl::PointCloud<pcl::Normal>::Ptr normals_;
};


class SkelEx {
public: 
SkelEx(rclcpp::Node::SharedPtr node); // Constructor with ROS2 node parsed

    /* Functions */
    void init();
    void main();
    void normal_estimation();

    /* Data */
    SkeletonStructure SS;
    
    /* Params */

private:
    rclcpp::Node::SharedPtr node_;

    /* Params */
    int ne_KNN = 20;
    double leaf_size_ds = 0.001;
    int max_points = 300;

    /* Data */
    int pcd_size_;
    double norm_scale;
    Eigen::Vector4d centroid;
};

#endif