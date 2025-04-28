/*

Node for publishing Real-Time Lidar data
Simple republisher to keep format...

*/

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>

class LidarPublisher : public rclcpp::Node {
public:
    LidarPublisher() : Node("lidar_publisher_node") {
        pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/lidar_scan", 10);
        // sub_ = create_subscription<sensor_msgs::msg::PointCloud2>("/pointcloud", 10, std::bind(&LidarPublisher::publishPointCloud, this, std::placeholders::_1));
        sub_ = create_subscription<sensor_msgs::msg::PointCloud2>("/isaac/point_cloud_0", 10, std::bind(&LidarPublisher::publishPointCloud, this, std::placeholders::_1));
    }

private:
    void publishPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr pcd_msg) {
        pcd_msg->header.frame_id = "lidar_frame";
        // pcd_msg->header.stamp = this->now();
        pcd_msg->header.stamp = pcd_msg->header.stamp;
        pub_->publish(*pcd_msg);
    }

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarPublisher>());
    rclcpp::shutdown();
    return 0;
  }