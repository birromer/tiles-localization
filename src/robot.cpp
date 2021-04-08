#include "ros/ros.h"

#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Quaternion.h"
#include "geometry_msgs/PoseStamped.h"

#include "tiles_loc/State.h"
#include "tiles_loc/Observation.h"
#include "tiles_loc/Cmd.h"

#include "tf/tf.h"

#include "std_msgs/Float32.h"
#include "std_msgs/Float32MultiArray.h"

#include <math.h>

float w_x=0, w_y=0, w_th=0;
float cmd_r, cmd_l;

double sawtooth(double x);

void waypoint_callback(const geometry_msgs::PoseStamped::ConstPtr& msg);
void cmd_callback(const tiles_loc::Cmd::ConstPtr& msg);

void integration_euler(double &x1, double &x2, double &x3, double &x4, double u1, double u2, double dt);

int main(int argc, char **argv) {
  ros::init(argc, argv, "robot_node");
  ros::NodeHandle n;

  ros::Rate loop_rate(25);  // 25Hz frequency
  double dt = 0.005;  // time step

  double u1 = cmd_r;
  double u2 = cmd_l;

  double x1, x2, x3, x4;

  // read initial values from the launcher
  n.param<double>("pos_x_init", x1, 0);
  n.param<double>("pos_y_init", x2, 0);
  n.param<double>("pos_th_init", x3, 0);

  // --- subscribers --- //
  // subscriber to z inputs from control
  ros::Subscriber sub_cmd = n.subscribe("cmd", 1000, cmd_callback);

  // subscriber to the image from the simulator
//  ros::Subscriber sub_img = n.subscribe("image", 1000, image_callback);

  // subscriber to the updated state from the localization subsystem
  ros::Subscriber sub_loc = n.subscribe("state_loc", 1000, x_loc_callback);
  // ------------------ //

  // --- publishers --- //
  // publisher of the robot's state for control and viewer
  ros::Publisher pub_state = n.advertise<tiles_loc::State>("state", 1000);

  // publisher of the robot's predicted state and observation for localization
  ros::Publisher pub_state_pred = n.advertise<tiles_loc::State>("state_pred", 1000);
  ros::Publisher pub_y = n.advertise<tiles_loc::Observation>("y", 1000);
  // ------------------ //

  // publi

  ros::Publisher pub_cmd_l;
  ros::Publisher pub_cmd_r;

  while (ros::ok()) {
    std::cout << "inicio do loop" << std::endl;
    u1 = cmd_r;
    u2 = cmd_l;

    if (DATA_FROM_SIM == false) { // evolve model
      std::cout << "entrei no caso com simulacao" << std::endl;
      integration_euler(x1, x2, x3, x4, u1, u2, dt);

    } else { // get simulation data
      std::cout << "entrei no caso sem simulacao" << std::endl;

      double vit = sqrt(pow(x1 - prev_x1, 2) + pow(x2-prev_x2,2));

      x1 = lx;
      x2 = ly;
      x3 = compass;
      x4 = vit;

      prev_x1 = x1;
      prev_x2 = x2;

    }

    // publish simulation
    ddboat_sim::State state_msg;

    state_msg.x1 =  x1;
    state_msg.x2 =  x2;
    state_msg.x3 =  x3;
    state_msg.x4 =  x4;

    state_pub.publish(state_msg);

    ros::spinOnce();

    loop_rate.sleep();
  }

  return 0;
}

double sawtooth(double x){
  return 2.*atan(tan(x/2.));
}

// callbacks for each subscriber
void waypoint_callback(const geometry_msgs::PoseStamped::ConstPtr& msg){
  w_x = msg->pose.position.x;
  w_y = msg->pose.position.y;
  w_th = tf::getYaw(msg->pose.orientation);
  ROS_INFO("received waypoint: [%f] [%f] [%f]", w_x, w_y, w_th);
}

void cmd_callback(const tiles_loc::Cmd::ConstPtr& msg) {
  cmd1 = msg->data.u1;
  cmd2 = msg->data.u2;
  ROS_INFO("received command: u1 [%f] u2 [%f]", cmd1, cmd2);
}

void integration_euler(double &x1, double &x2, double &x3, double &x4, double u1, double u2, double dt) {
  std::cout << "x1: " << x1 << "| x2: "  << x2 << " | x3: " << x3 << " | x4 " << x4 << std::endl;
  x1 = x1 + dt * (x4*cos(x3));
  x2 = x2 + dt * (x4*sin(x3));
  x3 = x3 + dt * (P1*(u1-u2));
  x4 = x4 + dt * (P2*(u1+u2) + P3*x4*abs(x4));
}
