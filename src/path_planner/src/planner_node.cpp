/*

Path Planner Node

*/

#include "planner_main.hpp"

#include <chrono>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Core>

class PlannerNode : public rclcpp::Node {
public:
    PlannerNode() : Node("path_planner_node") {
        RCLCPP_INFO(this->get_logger(), "Skeleton Guided Path Planner Node Constructed");
    }

    void init();
    void pcd_callback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg);
    void vertex_callback(const sensor_msgs::msg::PointCloud2::SharedPtr vertex_msg);
    void run();
    void set_transform();

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr vertex_sub_;
    rclcpp::TimerBase::SharedPtr run_timer_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

private:
    std::shared_ptr<PathPlanner> planner;

    bool run_flag = false;
    int run_timer_ms = 50;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices;

    geometry_msgs::msg::TransformStamped curr_tf;
};

void PlannerNode::init() {
    RCLCPP_INFO(this->get_logger(), "Initializing Modules and Data Structures...");

    /* Subscriber, Publishers, Timers, etc... */
    pcd_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/lidar_scan", 10, std::bind(&PlannerNode::pcd_callback, this, std::placeholders::_1));
    vertex_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/local_vertices", 10, std::bind(&PlannerNode::vertex_callback, this, std::placeholders::_1));

    run_timer_ = this->create_wall_timer(std::chrono::milliseconds(run_timer_ms), std::bind(&PlannerNode::run, this));

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    /* Params */
    // Stuff from launch file (ToDo)...

    /* Data */
    vertices.reset(new pcl::PointCloud<pcl::PointXYZ>);

    /* Modules */
    planner = std::make_shared<PathPlanner>(shared_from_this());
    planner->init();
}

void PlannerNode::pcd_callback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
    if (cloud_msg->data.empty()) {
        RCLCPP_INFO(this->get_logger(), "Received empty point cloud");
        return;
    }
    pcl::fromROSMsg(*cloud_msg, *cloud);
    planner->local_pts = cloud;
}

void PlannerNode::vertex_callback(const sensor_msgs::msg::PointCloud2::SharedPtr vertex_msg) {
    if (vertex_msg->data.empty()) {
        RCLCPP_INFO(this->get_logger(), "Received empty vertex set");
        return;
    }
    pcl::fromROSMsg(*vertex_msg, *vertices);
    planner->local_vertices = vertices;

    try {
        curr_tf = tf_buffer_->lookupTransform("World", "lidar_frame", tf2::TimePointZero);
        set_transform();
    }
    catch (const tf2::TransformException &ex) {
        RCLCPP_ERROR(this->get_logger(), "Transform Lookup Failed: %s", ex.what());
        return;
    }

    run_flag = true;
}

void PlannerNode::set_transform() {
    Eigen::Quaterniond q(curr_tf.transform.rotation.w,
                         curr_tf.transform.rotation.x,
                         curr_tf.transform.rotation.y,
                         curr_tf.transform.rotation.z);
    Eigen::Matrix3d R = q.toRotationMatrix();
    Eigen::Vector3d t(curr_tf.transform.translation.x,
                      curr_tf.transform.translation.y,
                      curr_tf.transform.translation.z);
    planner->tf_rot = R;
    planner->tf_trans = t;
}

void PlannerNode::run() {
    if (run_flag) {
        run_flag = false;
        planner->main();
    }
}




int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlannerNode>();
    node->init(); // Initialize Modules etc...

    // Spin the node
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}