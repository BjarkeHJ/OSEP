#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_control_mode.hpp>

using namespace px4_msgs::msg;
using namespace std::chrono_literals;

class OffboardControl : public rclcpp::Node
{
public:
    OffboardControl() : Node("offboard_control") {
        /* ROS2 Publishers */
        offboard_control_mode_pub_ = this->create_publisher<OffboardControlMode>("/fmu/in/offboard_control_mode", 10);
        trajectory_setpoint_pub_ = this->create_publisher<TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 10);
        vehicle_command_pub_ = this->create_publisher<VehicleCommand>("/fmu/in/vehicle_command", 10);
        trigger_pub_ = this->create_publisher<std_msgs::msg::Bool>("/trigger", 10);

        /* ROS2 Subscribers */
        target_setpoint_sub_ = this->create_subscription<TrajectorySetpoint>("/target_setpoint", 10, 
                                                                            std::bind(&OffboardControl::setpoint_callback, this, std::placeholders::_1));

        /* ROS2 Timers */
        timer_ = this->create_wall_timer(100ms, std::bind(&OffboardControl::timer_callback, this));
    }

private:
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_pub_;
    rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_pub_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr trigger_pub_;
    rclcpp::Subscription<TrajectorySetpoint>::SharedPtr target_setpoint_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    void setpoint_callback(const TrajectorySetpoint &msg);
    void timer_callback();

    void arm();
    void disarm();
    void pub_offboard_control_mode();
    void pub_vehicle_command(uint16_t command, float param1 = 0.0, float param2 = 0.0);
    void pub_trajectory_setpoint();
    
    std::array<float, 3> transform_ned_enu(const std::array<float, 3> &input);
    float transform_yaw(const float &input);

    std::array<float, 3> position = {0.0f, 0.0f, 200.0f};
    float yaw;

    int offboard_control_counter = 0;
    // bool target_update_flag = false;
    bool target_update_flag = true;
};

void OffboardControl::setpoint_callback(const TrajectorySetpoint &msg) {
    target_update_flag = true;
    position = msg.position;
    yaw = msg.yaw;
}

void OffboardControl::timer_callback() {
    pub_offboard_control_mode();

    if (target_update_flag == true) {
        pub_trajectory_setpoint();
        // target_update_flag = false;
    }
    
    if (offboard_control_counter == 10) {
        arm();
        pub_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1, 6);
    }

    if (offboard_control_counter < 11) offboard_control_counter++;
}

void OffboardControl::arm() {
    // Publish trigger to initialize PathPlanning Algorithm
    std_msgs::msg::Bool trigger{};
    trigger.data = true;
    trigger_pub_->publish(trigger);

    // Arm Drone ... 
    pub_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0);
    RCLCPP_INFO(this->get_logger(), "Arm command send");
}

void OffboardControl::disarm() {
    pub_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0);
    RCLCPP_INFO(this->get_logger(), "Disarm command send");
}

void OffboardControl::pub_offboard_control_mode() {
    OffboardControlMode msg{};
    msg.position = true;
    msg.velocity = false;
    msg.acceleration = false;
    msg.attitude = false;
    msg.body_rate = false;
    msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
    offboard_control_mode_pub_->publish(msg);
}

void OffboardControl::pub_vehicle_command(uint16_t command, float param1, float param2) {
    VehicleCommand msg{};
    msg.param1 = param1;
    msg.param2 = param2;
    msg.command = command;
    msg.target_system = 1;
    msg.target_component = 1;
    msg.source_system = 1;
    msg.source_component = 1;
    msg.from_external = true;
    msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
    vehicle_command_pub_->publish(msg);
}

void OffboardControl::pub_trajectory_setpoint()
{
    TrajectorySetpoint msg{};
    msg.position = transform_ned_enu(position);
    msg.yaw = transform_yaw(yaw);
    msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
    trajectory_setpoint_pub_->publish(msg);
}

std::array<float, 3> OffboardControl::transform_ned_enu(const std::array<float, 3>& input)
{
    std::array<float, 3> result;
    result[0] = input[1];
    result[1] = input[0];
    result[2] = -input[2];
    return result;
}

float OffboardControl::transform_yaw(const float& input)
{   
    // Yaw should stay the same ... 
    // Previously we expected yaw to be iverted (-input)
    float new_yaw = -input + M_PI / 2;
    return new_yaw;
}


int main(int argc, char* argv[]) {
    std::cout << "Starting Node ... " << std::endl;
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OffboardControl>());
    rclcpp::shutdown();
    return 0;
}