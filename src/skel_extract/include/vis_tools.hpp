#ifndef __vis__
#define __vis__

#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

class VisTools {
public:

    void publish_pointcloud();
    void publish_normals();
    void publish_graph_adj();

private:


};


#endif