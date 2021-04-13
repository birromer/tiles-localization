/*
** This node is responsible for the visualization of the robot's state, wapoint and estimation.
**
** Subscribers:
**   - geometry_msgs::PoseStamped waypoint  // the target state, waypoint
**   - geometry_msgs::PoseStamped pose      // the current pose, ground truth
**   - geometry_msgs::PoseStamped state     // the estimated state
**
** Publishers:
**   - none
 */

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <vector>
#include <cmath>

#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Quaternion.h"
#include "geometry_msgs/PoseStamped.h"
#include "tf/tf.h"

#include <tf2/LinearMath/Quaternion.h>

#include <ibex.h>
#include <tubex.h>
#include <tubex-rob.h>

using namespace cv;
using namespace std;
using namespace ibex;
using namespace tubex;

void waypoint_callback(const geometry_msgs::PoseStamped::ConstPtr& msg){
  float w_x, w_y, w_th;
  w_x = msg->pose.position.x;
  w_y = msg->pose.position.y;
  w_th = tf::getYaw(msg->pose.orientation);

  vibes::drawVehicle(w_x, w_y, w_th*180./M_PI, 0.2, "green");
}

void state_callback(const geometry_msgs::PoseStamped::ConstPtr& msg){
  float x_x, x_y, x_th;
  x_x = msg->pose.position.x;
  x_y = msg->pose.position.y;
  x_th = tf::getYaw(msg->pose.orientation);

  vibes::drawVehicle(x_x, x_y, x_th*180./M_PI, 0.4, "blue");
}

void pose_callback(const geometry_msgs::Pose& msg){
  float x, y, c;
  x = msg.position.x;
  y = msg.position.y;
  c = tf::getYaw(msg.orientation);

  vibes::drawVehicle(x, y, c*180./M_PI, 0.3, "red");
}

int main(int argc, char **argv){
  vibes::beginDrawing();
  VIBesFigMap fig_map("Map");
  vibes::setFigureProperties("Map",vibesParams("x", 10, "y", -10, "width", 100, "height", 100));
  vibes::axisLimits(-10, 10, -10, 10, "Map");

  fig_map.show();

  ros::init(argc, argv, "viewer_node");

  ros::NodeHandle n;

  ros::Subscriber sub_waypoint = n.subscribe("waypoint", 1000, waypoint_callback);
  ros::Subscriber sub_state = n.subscribe("state", 1000, state_callback);
  ros::Subscriber sub_pose = n.subscribe("pose", 1000, pose_callback);

  ros::spin();

  vibes::endDrawing();
  return 0;
}
