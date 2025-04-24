#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>

class PCDPublisher : public rclcpp::Node {
public:
  PCDPublisher() : Node("pcd_publisher_node"), publish_rate_hz_(1.0) {
    // Set the .pcd file path (change this accordingly)
    std::string pcd_path = "src/data_publisher/data/05_horizontal_wing_side.pcd";
    // std::string pcd_path = "src/data_publisher/data/06_horizontal_wing_side_behind.pcd";
    // std::string pcd_path = "src/data_publisher/data/07_vertical_wing_side_behind.pcd";
    // std::string pcd_path = "src/data_publisher/data/08_nacelle_side.pcd";
    // std::string pcd_path = "src/data_publisher/data/09_wings_only_front.pcd";
    publish_rate_hz_ = 10;

    // Load the PCD file
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path, cloud_) == -1) {
      RCLCPP_ERROR(get_logger(), "Couldn't read file: %s", pcd_path.c_str());
      rclcpp::shutdown();
    } else {
      RCLCPP_INFO(get_logger(), "Loaded PCD file: %s with %zu points", pcd_path.c_str(), cloud_.points.size());
      RCLCPP_INFO(get_logger(), "Starting Publishing PointCloud at a Rate of %f Hz", publish_rate_hz_);
    }

    // Create publisher
    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/lidar_scan", 1);

    // Create timer to publish periodically
    timer_ = create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate_hz_)),
      std::bind(&PCDPublisher::publishPointCloud, this)
    );
  }

private:
  void publishPointCloud() {
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(cloud_, output);
    output.header.frame_id = "map";  // Adjust as needed
    output.header.stamp = now();
    pub_->publish(output);
  }

  pcl::PointCloud<pcl::PointXYZ> cloud_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  double publish_rate_hz_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PCDPublisher>());
  rclcpp::shutdown();
  return 0;
}