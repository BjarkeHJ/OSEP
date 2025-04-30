/*

Path Planner Node

*/

#include "planner_main.hpp"

#include <chrono>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Core>

class PlannerNode : public rclcpp::Node {
public:
    PlannerNode() : Node("path_planner_node") {
        RCLCPP_INFO(this->get_logger(), "Skeleton Guided Path Planner Node Constructed");
    }

    void init();
    void pcd_callback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg);
    void vertex_callback(const sensor_msgs::msg::PointCloud2::SharedPtr vertex_msg);
    void publish_gskel();
    void run();

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr vertex_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr gskel_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr adj_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr wayp_pub_;
    rclcpp::TimerBase::SharedPtr run_timer_;
    rclcpp::TimerBase::SharedPtr gskel_timer_;
    
private:

    std::shared_ptr<PathPlanner> planner;

    bool run_flag = false;
    int run_timer_ms = 50;
    int gskel_timer_ms = 50;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices;

    pcl::VoxelGrid<pcl::PointXYZ> vgf_ds;

    std::string global_frame_id = "World";
    // std::string global_frame_id = "odom";
};

void PlannerNode::init() {
    RCLCPP_INFO(this->get_logger(), "Initializing Modules and Data Structures...");

    /* Subscriber, Publishers, Timers, etc... */
    pcd_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/local_points", 10, std::bind(&PlannerNode::pcd_callback, this, std::placeholders::_1));
    vertex_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/local_vertices", 10, std::bind(&PlannerNode::vertex_callback, this, std::placeholders::_1));
    gskel_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/global_skeleton", 10);
    adj_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/adjacency_graph", 10);
    cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/global_points", 10);
    wayp_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/next_waypoint", 10);

    run_timer_ = this->create_wall_timer(std::chrono::milliseconds(run_timer_ms), std::bind(&PlannerNode::run, this));
    gskel_timer_ = this->create_wall_timer(std::chrono::milliseconds(gskel_timer_ms), std::bind(&PlannerNode::publish_gskel, this));


    /* Params */
    // Stuff from launch file (ToDo)...

    /* Data */
    vertices.reset(new pcl::PointCloud<pcl::PointXYZ>);
    cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

    /* Modules */
    planner = std::make_shared<PathPlanner>(shared_from_this());
    planner->init();
}

void PlannerNode::pcd_callback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
    if (!cloud_msg ||cloud_msg->data.empty()) {
        RCLCPP_INFO(this->get_logger(), "Received empty point cloud");
        return;
    }
    cloud->clear();
    pcl::fromROSMsg(*cloud_msg, *cloud);
    *(planner->GS.global_pts) += *cloud;

    vgf_ds.setInputCloud(planner->GS.global_pts);
    vgf_ds.setLeafSize(2.0, 2.0, 2.0);
    vgf_ds.filter(*planner->GS.global_pts);

    sensor_msgs::msg::PointCloud2 output_msg;
    pcl::toROSMsg(*planner->GS.global_pts, output_msg);
    output_msg.header.frame_id = global_frame_id;
    output_msg.header.stamp = cloud_msg->header.stamp;
    cloud_pub_->publish(output_msg);
}

void PlannerNode::vertex_callback(const sensor_msgs::msg::PointCloud2::SharedPtr vertex_msg) {
    if (!vertex_msg || vertex_msg->data.empty()) {
        RCLCPP_INFO(this->get_logger(), "Received empty vertex set");
        return;
    }
    pcl::fromROSMsg(*vertex_msg, *vertices);
    planner->local_vertices = vertices;
    run_flag = true;
}

void PlannerNode::publish_gskel() {
    if (planner->GS.global_vertices_cloud && !planner->GS.global_vertices_cloud->empty()) {
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*planner->GS.global_vertices_cloud, output);
        output.header.frame_id = global_frame_id;
        output.header.stamp = now();
        gskel_pub_->publish(output);

        visualization_msgs::msg::Marker lines;
        lines.header.frame_id = global_frame_id;
        lines.header.stamp = this->get_clock()->now();
        lines.type = visualization_msgs::msg::Marker::LINE_LIST;
        lines.action = visualization_msgs::msg::Marker::ADD;
        lines.pose.orientation.w = 1.0;
        lines.scale.x = 0.05;
        lines.color.r = 0.2;
        lines.color.g = 0.8;
        lines.color.b = 0.4;
        lines.color.a = 1.0;

        std::set<std::pair<int, int>> published_edges;
        geometry_msgs::msg::Point p1, p2;
        for (int i = 0; i < (int)planner->GS.global_adj.size(); ++i) {
            const auto& neighbors = planner->GS.global_adj[i];
            for (int j : neighbors) {
                // Make sure we only publish each edge once
                if (i < j) {
                    p1.x = planner->GS.global_vertices_cloud->points[i].x;
                    p1.y = planner->GS.global_vertices_cloud->points[i].y;
                    p1.z = planner->GS.global_vertices_cloud->points[i].z;

                    p2.x = planner->GS.global_vertices_cloud->points[j].x;
                    p2.y = planner->GS.global_vertices_cloud->points[j].y;
                    p2.z = planner->GS.global_vertices_cloud->points[j].z;

                    lines.points.push_back(p1);
                    lines.points.push_back(p2);

                    published_edges.insert({i, j});
                }
            }
        }
        adj_pub_->publish(lines);
    }
    else RCLCPP_INFO(this->get_logger(), "WARNING: No Global Skeleton Available");
}

void PlannerNode::run() {
    // Change update logic to:
    // If new vertices -> Update skeleton
    // If no new vertices -> Still perform refinements and path planning computations etc...

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