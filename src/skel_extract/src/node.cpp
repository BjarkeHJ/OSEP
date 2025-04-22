/* 

ROS2 node 

*/

#include "node.hpp"
#include "main.hpp"
#include "vis_tools.hpp"

void OsepNode::init() {
    osep.reset(new OsepNode);
    osep->init();
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OsepNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}