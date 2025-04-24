#ifndef PLANNER_MAIN_
#define PLANNER_MAIN_

#include <rclcpp/rclcpp.hpp>
#include <pcl/common/common.h>

struct GlobalSkeleton {
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_pts;
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_vertices;
};

class PathPlanner {
public:
    PathPlanner(rclcpp::Node::SharedPtr node);
    void init();
    void main();

    /* Data */
    GlobalSkeleton GS;
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_pts;
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_vertices;

private:
    rclcpp::Node::SharedPtr node_;

};

#endif //PLANNER_MAIN_