/*
** This node is responsible for generating the command inputs u1 and u2 for the robot.
**
** Subscribers:
**   - geometry_msgs::PoseStamped waypoint  // the target state
**   - tiles_loc::State           state     // the current state of the robot
**
** Publishers:
**   - std_msgs::Float64 command_l  // the input u1 and u2 for the robot
**   - std_msgs::Float64 command_r  // the input u1 and u2 for the robot
 */

#include "ros/ros.h"
#include "std_msgs/Float64.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Quaternion.h"
#include "geometry_msgs/PoseStamped.h"
#include "tf/tf.h"
#include "tiles_loc/State.h"

// utils
double sawtooth(double x);
float max(float a, float b);
float min(float a, float b);
float sign(float a);

// callback functions
void waypoint_callback(const geometry_msgs::PoseStamped::ConstPtr& msg);
void state_callback(const tiles_loc::State::ConstPtr& msg);

// global variables and constants
ros::Publisher pub_cmd_l;
ros::Publisher pub_cmd_r;

double current_speed = 100.;
int max_speed = 300;

float w_x, w_y, w_th;
float x_x, x_y, x_th;

float last_L = 0;
float f = 0;

int main(int argc, char **argv)
{
  ros::init(argc, argv, "control_node");
  ros::NodeHandle n;
  ros::Rate loop_rate(25);  // 25Hz frequency

  // --- subscribers --- //
  // subscriber to waypoint from command
  ros::Subscriber sub_waypoint = n.subscribe("waypoint", 1000, waypoint_callback);
  // subscriber to curret state from robot
  ros::Subscriber sub_state = n.subscribe("state", 1000, state_callback);
  // ------------------ //

  // --- publishers --- //
  // publisher of command u for the robot
  pub_cmd_l = n.advertise<std_msgs::Float64>("cmd_l", 1000);
  pub_cmd_r = n.advertise<std_msgs::Float64>("cmd_r", 1000);
  // ------------------ //

  while (ros::ok()) {
    float L = cos(w_th)*(w_x - x_x) + sin(w_th)*(w_y - x_y);  // to control speed in curves
    float dist = sqrt(pow(w_x - x_x, 2) + pow(w_y - x_y, 2));

    //current_speed += sqrt(abs(L))*sign(L)/2.;

    if(dist > 1.) {
      current_speed = max_speed*0.90;

    } else {
      current_speed = 20*dist;
      current_speed = min(current_speed, 100);
      f += 1.*(L - last_L);
      current_speed += f;
    }

    last_L = L;

    current_speed = max(min(current_speed, max_speed), 0);

    float angle_state_waypoint = atan2(w_y - x_y, w_x - x_x);  // angle with the waypoint

    float c2 = (0.1*w_th + 0.9*angle_state_waypoint);
    //float c2 = atan2(Y-cons_y, X-cons_x);

    //float cmd_r = (0.5 - sawtooth(C-c2)/M_PI)*round(current_speed);
    //float cmd_l = (0.5 + sawtooth(C-c2)/M_PI)*round(current_speed);
    float max_diff = 9.*current_speed/10.*2.;
    float diff = sawtooth(c2 - x_th)/M_PI*max_diff;

    float cmd_l = current_speed - diff;
    float cmd_r = current_speed + diff;

    cmd_r = max(min(cmd_r, max_speed), -0);
    cmd_l = max(min(cmd_l, max_speed), -0);

//    ROS_INFO("[CONTROL] L : %f | current_speed : %f | C : %f", L, current_speed, x_th*180/M_PI);
//    ROS_INFO("[CONTROL] dist : %f", dist);
//    ROS_INFO("[CONTROL] cmd_l : %f | cmd_r : %f ", cmd_l, cmd_r);
//    ROS_INFO("[CONTROL] cons_cap : %f | angle : %f | c2 : %f ", w_th*180/M_PI, angle_state_waypoint*180/M_PI, c2*180/M_PI);
//    ROS_INFO("[CONTROL] sawtooth : %f \n", sawtooth(x_th - c2)/M_PI);

    cmd_l /= 100;
    cmd_r /= 100;

    std_msgs::Float64 cmd_msg_l;
    std_msgs::Float64 cmd_msg_r;

    cmd_msg_l.data = cmd_l;
    cmd_msg_r.data = cmd_r;

    pub_cmd_l.publish(cmd_msg_l);
    pub_cmd_r.publish(cmd_msg_r);

//    ROS_INFO("[CONTROL] Sent commands -> u1: [%f] | u2: [%f]", cmd_l, cmd_r);

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}

double sawtooth(double x) {
  return 2.*atan(tan(x/2.));
}

float max(float a, float b) {
  if(a > b) {
    return a;
  }
  return b;
}

float min(float a, float b) {
  return -max(-a, -b);
}

float sign(float a) {
  if(a < 0) {
    return -1;
  }
  return 1;
}

void waypoint_callback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
  w_x  = msg->pose.position.x;
  w_y  = msg->pose.position.y;
  w_th = tf::getYaw(msg->pose.orientation);
//  ROS_INFO("[CONTROL] Received waypoint -> w_x: [%f] | w_y: [%f] | w_th: [%f]", w_x, w_y, w_th);
}

void state_callback(const tiles_loc::State::ConstPtr& msg){
  x_x = (msg->x1_lb + msg->x1_ub)/2.;
  x_y = (msg->x2_lb + msg->x2_ub)/2.;
  x_th = (msg->x3_lb + msg->x3_ub)/2. * 180./M_PI;  // NOTE: check if radians or degrees should be user later
//  ROS_INFO("[CONTROL] Received state-> x1: [%f] | x2: [%f] | x3: [%f]", x_x, x_y, x_th);
}
