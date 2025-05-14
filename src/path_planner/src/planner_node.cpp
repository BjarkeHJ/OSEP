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
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>

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
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr odom_msg);

    void publish_gskel();
    void publish_viewpoints();
    void publish_path();
    void init_path();
    void drone_tracking();
    void adjust_viewpoints(const nav_msgs::msg::Path::SharedPtr adjusted_vpts);

    void run();

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr vertex_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr gskel_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr adj_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr viewpoint_pub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr adjusted_vpts_sub_;
    
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr seen_voxels_pub_;
    
    rclcpp::TimerBase::SharedPtr run_timer_;

private:
    std::string topic_prefix = "/osep";

    std::shared_ptr<PathPlanner> planner;

    bool update_skeleton_flag = false;
    bool planner_flag = false;
    bool path_init = false;
    int run_cnt;

    int run_timer_ms = 100;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices;

    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;

    // std::string global_frame_id = "World";
    std::string global_frame_id = "odom";
};

void PlannerNode::init() {
    RCLCPP_INFO(this->get_logger(), "Initializing Modules and Data Structures...");

    /* Subscriber, Publishers, Timers, etc... */
    pcd_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(topic_prefix+"/local_points", 10, std::bind(&PlannerNode::pcd_callback, this, std::placeholders::_1));
    vertex_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(topic_prefix+"/local_vertices", 10, std::bind(&PlannerNode::vertex_callback, this, std::placeholders::_1));
    gskel_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(topic_prefix+"/global_skeleton", 10);
    adj_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(topic_prefix+"/adjacency_graph", 10);
    cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(topic_prefix+"/global_points", 10);

    path_pub_ = this->create_publisher<nav_msgs::msg::Path>(topic_prefix+"/viewpoints", 10);
    viewpoint_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(topic_prefix+"/all_viewpoints", 10);

    seen_voxels_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(topic_prefix+"/seen_voxel", 10); // When a vpt is popped published the seen voxels 
    adjusted_vpts_sub_ = this->create_subscription<nav_msgs::msg::Path>("/planner/viewpoints_adjusted", 10, std::bind(&PlannerNode::adjust_viewpoints, this, std::placeholders::_1)); 

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/isaac/odom", 10, std::bind(&PlannerNode::odom_callback, this, std::placeholders::_1));

    run_timer_ = this->create_wall_timer(std::chrono::milliseconds(run_timer_ms), std::bind(&PlannerNode::run, this));

    /* Params */
    // Stuff from launch file (ToDo)...

    /* Data */
    vertices.reset(new pcl::PointCloud<pcl::PointXYZ>);
    cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

    /* Modules */
    planner = std::make_shared<PathPlanner>(shared_from_this());
    planner->init();

    run_cnt = 0;

    RCLCPP_INFO(this->get_logger(), "Initialization Done...");
}

void PlannerNode::pcd_callback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
    if (!cloud_msg ||cloud_msg->data.empty()) {
        return;
    }
    pcl::fromROSMsg(*cloud_msg, *planner->local_pts);
    planner->global_cloud_handler();

    sensor_msgs::msg::PointCloud2 global_msg;
    pcl::toROSMsg(*planner->GS.global_pts, global_msg);
    global_msg.header.frame_id = global_frame_id;
    global_msg.header.stamp = cloud_msg->header.stamp;
    cloud_pub_->publish(global_msg);
}

void PlannerNode::vertex_callback(const sensor_msgs::msg::PointCloud2::SharedPtr vertex_msg) {
    if (!vertex_msg || vertex_msg->data.empty()) {
        return;
    }
    pcl::fromROSMsg(*vertex_msg, *vertices);
    planner->local_vertices = vertices;
    update_skeleton_flag = true;
}

void PlannerNode::odom_callback(const nav_msgs::msg::Odometry::SharedPtr odom_msg) {
    if (!odom_msg) {
        return;
    }
    position(0) = odom_msg->pose.pose.position.x;
    position(1) = odom_msg->pose.pose.position.y;
    position(2) = odom_msg->pose.pose.position.z;
    
    orientation.w() = odom_msg->pose.pose.orientation.w;
    orientation.x() = odom_msg->pose.pose.orientation.x;
    orientation.y() = odom_msg->pose.pose.orientation.y;
    orientation.z() = odom_msg->pose.pose.orientation.z;
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
}

void PlannerNode::publish_viewpoints() {
    // Publish all generated viewpoints
    if (!planner->GP.global_vpts.empty()) {
        geometry_msgs::msg::PoseArray gvps_msg;
        gvps_msg.header.frame_id = global_frame_id;
        gvps_msg.header.stamp = now();
    
        // for (const auto& vp : planner->GP.all_vpts) {
        for (const auto& vp : planner->GP.global_vpts) {
            geometry_msgs::msg::Pose vp_pose;
            vp_pose.position.x = vp.position.x();
            vp_pose.position.y = vp.position.y();
            vp_pose.position.z = vp.position.z();
            vp_pose.orientation.x = vp.orientation.x();
            vp_pose.orientation.y = vp.orientation.y();
            vp_pose.orientation.z = vp.orientation.z();
            vp_pose.orientation.w = vp.orientation.w();
            gvps_msg.poses.push_back(vp_pose);
        }
        viewpoint_pub_->publish(gvps_msg);    
    }
}

void PlannerNode::publish_path() {
    const auto& vpts = planner->GP.local_path;
    if (!vpts.empty()) {
        nav_msgs::msg::Path path_msg;
        path_msg.header.frame_id = global_frame_id;
        path_msg.header.stamp = now();

        for (const auto& vp : vpts) {
            geometry_msgs::msg::PoseStamped pose_msg;
            pose_msg.header = path_msg.header;

            pose_msg.pose.position.x = vp.position.x();
            pose_msg.pose.position.y = vp.position.y();
            pose_msg.pose.position.z = vp.position.z();

            pose_msg.pose.orientation.x = vp.orientation.x();
            pose_msg.pose.orientation.y = vp.orientation.y();
            pose_msg.pose.orientation.z = vp.orientation.z();
            pose_msg.pose.orientation.w = vp.orientation.w();

            path_msg.poses.push_back(pose_msg);
        }
        path_pub_->publish(path_msg);
    }
}

void PlannerNode::init_path() {
    // Guiding the drone towards the structure
    Viewpoint first_vp;
    Viewpoint second_vp;
    Viewpoint third_vp;
    Viewpoint fourth_vp;
    
    // first_vp.position = Eigen::Vector3d(0.0, 0.0, 50.0);
    // first_vp.orientation = Eigen::Quaterniond::Identity();

    second_vp.position = Eigen::Vector3d(0.0, 0.0, 120.0);
    second_vp.orientation = Eigen::Quaterniond::Identity();

    third_vp.position = Eigen::Vector3d(100.0, 0.0, 120.0);
    third_vp.orientation = Eigen::Quaterniond::Identity();

    fourth_vp.position = Eigen::Vector3d(170.0, 0.0, 120.0);
    fourth_vp.orientation = Eigen::Quaterniond::Identity();

    // planner->GP.local_path.push_back(first_vp);
    planner->GP.local_path.push_back(second_vp);
    planner->GP.local_path.push_back(third_vp);
    planner->GP.local_path.push_back(fourth_vp);
}

void PlannerNode::drone_tracking() {
    // if (planner->GP.local_path.empty()) return;

    // const double dist_check_th = 1.5;

    // while (!planner->GP.local_path.empty()) {
    //     const Viewpoint& current_next = planner->GP.local_path.front();
    //     double distance_to_drone = (current_next.position - position).norm();

    //     if (distance_to_drone < dist_check_th) {
    //         RCLCPP_INFO(this->get_logger(), "Arrived at Viewpoint - Removing from path");
    //         // planner->GP.local_path[0].visited = true;
    //         planner->mark_viewpoint_visited(planner->GP.local_path[0]);
    //         planner->GP.traced_path.push_back(planner->GP.local_path[0]); // Assign to traced path
    //         planner->GP.local_path.erase(planner->GP.local_path.begin()); // Remove the first element
    //     } else {
    //         break;
    //     }
    // }

    if (planner->GP.adjusted_path.empty()) return;

    const double dist_check_th = 2.0;

    for (int i=0; i<(int)planner->GP.adjusted_path.size(); ++i) {
        Viewpoint target = planner->GP.adjusted_path[i];
        double distance_to_drone = (target.position - position).norm();
        if (distance_to_drone < dist_check_th) {
            planner->GP.traced_path.push_back(planner->GP.adjusted_path[i]);
            planner->GP.adjusted_path.erase(planner->GP.adjusted_path.begin(), planner->GP.adjusted_path.begin() + (i+1));
            planner->GP.local_path.erase(planner->GP.local_path.begin(), planner->GP.local_path.begin() + (i+1));
            RCLCPP_INFO(this->get_logger(), "Reached Viewpoint: Deleting %d viewpoints from path", (i+1));
            break;
        }
    }
}

void PlannerNode::adjust_viewpoints(const nav_msgs::msg::Path::SharedPtr adjusted_vpts) {
    if (adjusted_vpts->poses.empty()) return;

    planner->GP.adjusted_path.clear();
    planner->GP.adjusted_path.reserve(adjusted_vpts->poses.size());
    for (const auto& ps : adjusted_vpts->poses) {
        Viewpoint vp;
        vp.position = Eigen::Vector3d(
                    ps.pose.position.x,
                    ps.pose.position.y,
                    ps.pose.position.z
        );
        vp.orientation = Eigen::Quaterniond(
                        ps.pose.orientation.w,
                        ps.pose.orientation.x,
                        ps.pose.orientation.y,
                        ps.pose.orientation.z
        );

        planner->GP.adjusted_path.push_back(vp);
    }
}

void PlannerNode::run() {
    if (!planner_flag) {
        /* Initial Flight to Structure (Predefined) */

        if (!path_init && planner->GP.local_path.empty()) {
            init_path(); // Set once
            path_init = true; 
            RCLCPP_INFO(this->get_logger(), "Initial Path Set - Following until structure detected!");     
        }

        drone_tracking();
        publish_path();
        
        if (!planner->local_vertices->empty()) {
            RCLCPP_INFO(this->get_logger(), "Recieved first vertices - Starting Planning!");
            planner->GP.local_path.clear();
            planner_flag = true;
        }

        return;
    }

    if (update_skeleton_flag) {
        planner->update_skeleton();
        update_skeleton_flag = false;
    }

    publish_gskel();
    publish_viewpoints();
    
    if (run_cnt >= 20) {
        planner->pose = {position, orientation};
        planner->plan_path();
        drone_tracking();
        publish_path();
    }
    else run_cnt++;

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