#ifndef __node__
#define __node__

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>



class OsepNode : public rclcpp::Node {
public:
    OsepNode() : Node("OSEP_Node") {
        init();
    }
    void init();

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_sub_;

    

};




#endif