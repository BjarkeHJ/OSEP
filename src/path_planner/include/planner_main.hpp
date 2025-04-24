#ifndef PLANNER_MAIN_
#define PLANNER_MAIN_

#include <rclcpp/rclcpp.hpp>
#include <pcl/common/common.h>
#include <Eigen/Core>

struct GlobalSkeleton {
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_pts;
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_vertices;
};

class PathPlanner {
public:
    PathPlanner(rclcpp::Node::SharedPtr node);
    void init();
    void main();
    void update_skeleton();

    /* Data */
    GlobalSkeleton GS;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_pts;
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_vertices;

    Eigen::Matrix3d tf_rot;
    Eigen::Vector3d tf_trans;

private:
    rclcpp::Node::SharedPtr node_;

    /* Data */

    /* Params */
    double lkf_pn = 0.001; // LKF process noise
    double lkf_mn = 0.1; // LKF measurement noise
};

/* Linear Kalman Filter for Vertex Fusion*/
struct SkeletonVertex
{
    Eigen::Vector3d position;
    Eigen::Matrix3d covariance;
    int observation_count = 0;
    bool confidence_check = false;
    Eigen::Vector3d state;
    Eigen::Matrix3d P;
};

class VertexLKF {
public:
    VertexLKF(double process_noise = 0.1f, double measurement_noise = 0.5f) {
        Q = Eigen::Matrix3d::Identity() * process_noise;
        R = Eigen::Matrix3d::Identity() * measurement_noise;
    }
    void initialize(Eigen::Vector3d initial_position, Eigen::Matrix3d covariance) {
        x = initial_position;
        P = covariance;
    }
    void update(const Eigen::Vector3d &z) {
        // Prediction 
        Eigen::Vector3d x_pred = x;
        Eigen::Matrix3d P_pred = P + Q;
        // Kalman Gain
        Eigen::Matrix3d K = P_pred * (P_pred + R).inverse();
        // Correction
        x = x_pred + K * (z - x_pred);
        P = (Eigen::Matrix3d::Identity() - K) * P_pred;
    }
    Eigen::Vector3d getState() const {return x;}
    Eigen::Matrix3d getCovariance() const {return P;}
private:
    Eigen::Vector3d x;
    Eigen::Matrix3d P;
    Eigen::Matrix3d Q;
    Eigen::Matrix3d R;
};


#endif //PLANNER_MAIN_