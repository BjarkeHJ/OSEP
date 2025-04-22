#ifndef __main__
#define __main__

#include <rclcpp/rclcpp.hpp>
#include <pcl/common/common.h>


struct SkeletonStructure {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pts_;

};


class OSEP {
public: 
    OSEP(rclcpp::Node::SharedPtr node); // Constructor with ROS2 node parsed

    /* Functions */
    void init();
    void main();

    /* Data */
    SkeletonStructure SS;
    
    /* Params */

private:
    rclcpp::Node::SharedPtr node_;


};

#endif