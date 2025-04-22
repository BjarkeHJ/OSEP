#ifndef __main__
#define __main__

#include <rclcpp/rclcpp.hpp>

class OSEP {
public: 
    OSEP(rclcpp::Node::SharedPtr node); // Constructor with ROS2 node parsed

    void init();
    void main();

private:
    rclcpp::Node::SharedPtr nnode_;


};

#endif