/* 

Main skeleton extraction algorithm 

*/

#include <skel_main.hpp>

SkelEx::SkelEx(rclcpp::Node::SharedPtr node) : node_(node) 
{
}

void SkelEx::init() {
    RCLCPP_INFO(node_->get_logger(), "Initializing Module: Online Skeleton Extraction Planner");
    /* Params */
    // Stuff from launch file (Todo)...

    /* Modules */
    // Vis tools maybe?


    /* Data */
    SS.pts_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    SS.normals_.reset(new pcl::PointCloud<pcl::Normal>);
}

void SkelEx::main() {
    RCLCPP_INFO(node_->get_logger(), "Skeleton Extraction Main Algorithm...");
    pcd_size_ = SS.pts_->points.size();
    RCLCPP_INFO(node_->get_logger(), "Point Cloud Size: %d", pcd_size_);

}

void SkelEx::normal_estimation() {
    pcl::PointXYZ min, max;
    pcl::getMinMax3D(*SS.pts_, min, max);
    double x_scale, y_scale, z_scale;
    x_scale = max.x - min.x;
    y_scale = max.y - min.y;
    z_scale = max.z - min.z;
    norm_scale = std::max(x_scale, std::max(y_scale, z_scale));
    pcl::compute3DCentroid(*SS.pts_, centroid);

    for (int i=0; i<pcd_size_; ++i) {
        SS.pts_->points[i].x = (SS.pts_->points[i].x - centroid(0)) / norm_scale;
        SS.pts_->points[i].y = (SS.pts_->points[i].y - centroid(1)) / norm_scale;
        SS.pts_->points[i].z = (SS.pts_->points[i].z - centroid(2)) / norm_scale;

    }

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr ne_tree(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setInputCloud(SS.pts_);
    ne.setSearchMethod(ne_tree);
    ne.setKSearch(ne_KNN);
    ne.compute(*SS.normals_);

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_w_normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*SS.pts_, *SS.normals_, *cloud_w_normals);

    pcl::VoxelGrid<pcl::PointNormal> vgf;
    while (pcd_size_ > max_points) {
        vgf.setInputCloud(cloud_w_normals);
        vgf.setLeafSize(leaf_size_ds,leaf_size_ds,leaf_size_ds);
        vgf.filter(*cloud_w_normals);
        pcd_size_ = cloud_w_normals->points.size();
        if (pcd_size_ <= max_points) break;
        leaf_size_ds += 0.001;
    }

    pcd_size_ = cloud_w_normals->points.size();
    SS.pts_->clear();
    SS.normals_->clear();
    
}
