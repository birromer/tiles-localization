#include "ros/ros.h"

#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Quaternion.h"
#include "geometry_msgs/PoseStamped.h"

#include "std_msgs/Float32.h"
#include "std_msgs/Float32MultiArray.h"

#include "tiles_loc/Observation.h"
#include "tiles_loc/Cmd.h"

#include "tf/tf.h"

#include <math.h>

double sawtooth(double x);

void cmd_callback(const tiles_loc::Cmd::ConstPtr& msg);

void image_callback(const sensor_msgs::ImageConstPtr& msg);

void state_loc_callback(const geometry_msgs::PoseStamped::ConstPtr& msg);

void integration_euler(double &x1, double &x2, double &x3, double u1, double u2, double dt);

float cmd_1, cmd_2;

int main(int argc, char **argv) {
  ros::init(argc, argv, "robot_node");
  ros::NodeHandle n;

  ros::Rate loop_rate(25);  // 25Hz frequency
  double dt = 0.005;  // time step

  double x1, x2, x3;  // current state of the robot
  double y1, y2, y3;  // current observation of the robot
  double u1, u2;      // current input received

  // read initial values from the launcher
  n.param<double>("pos_x_init", x1, 0);
  n.param<double>("pos_y_init", x2, 0);
  n.param<double>("pos_th_init", x3, 0);

  // --- subscribers --- //
  // subscriber to z inputs from control
  ros::Subscriber sub_cmd = n.subscribe("cmd", 1000, cmd_callback);

  // subscriber to the image from the simulator
  ros::Subscriber sub_img = n.subscribe("image", 1000, image_callback);

  // subscriber to the updated state from the localization subsystem
  ros::Subscriber sub_loc = n.subscribe("state_loc", 1000, state_loc_callback);
  // ------------------ //

  // --- publishers --- //
  // publisher of the state for control and viewer
  ros::Publisher pub_state = n.advertise<geometry_msgs::PoseStamped::ConstPtr>("state", 1000);

  // publisher of the predicted state and observation for localization
  ros::Publisher pub_state_pred = n.advertise<geometry_msgs::PoseStamped::ConstPtr>("state_pred", 1000);
  ros::Publisher pub_y = n.advertise<tiles_loc::Observation>("observation", 1000);
  // ------------------ //

  while (ros::ok()) {
    x1 = x_loc_1, x2 = x_loc_2, x3 = x_loc_3;  // start with the last state contracted from the localization
    y1 = obs_1, y2 = obs_2, y3 = obs3;  // use last observed parameters from the image
    u1 = cmd_1, u2 = cmd_2;  // use last input from command

    if (DATA_FROM_SIM == false) { // evolve model
      std::cout << "entrei no caso com simulacao" << std::endl;
      integration_euler(x1, x2, x3, x4, u1, u2, dt);

    } else { // get simulation data
      std::cout << "entrei no caso sem simulacao" << std::endl;

      double vit = sqrt(pow(x1 - prev_x1, 2) + pow(x2-prev_x2,2));

      x1 = lx;
      x2 = ly;
      x3 = compass;

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
  ROS_INFO("Received waypoint: [%f] [%f] [%f]", w_x, w_y, w_th);
}

void cmd_callback(const tiles_loc::Cmd::ConstPtr& msg) {
  cmd_1 = msg->data.u1;
  cmd_2 = msg->data.u2;
  ROS_INFO("Received command: u1 [%f] u2 [%f]", cmd_1, cmd_2);
}

void integration_euler(double &x1, double &x2, double &x3, double u1, double u2, double dt) {
  x1 = x1 + dt * (u1*cos(x3));
  x2 = x2 + dt * (u1*sin(x3));
  x3 = x3 + dt * (u2);
  ROS_INFO("Updated state -> x1: [%f] | x2: [%f] | x3: [%f] || u1: [%f] | u2: [%f]", x1, x2, x3, u1, u2);
}
