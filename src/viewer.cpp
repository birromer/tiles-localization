/*
** This node is responsible for the visualization of the robot's state, wapoint and estimation.
**
** Subscribers:
**   - geometry_msgs::PoseStamped waypoint  // the target state, waypoint
**   - geometry_msgs::PoseStamped pose      // the current pose, ground truth
**   - tiles_loc::State state_loc           // the estimated state
**   - tiles_loc::State state_pred          // the predicted state, from the state equations
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

#include "tiles_loc/State.h"

#include <tf2/LinearMath/Quaternion.h>

#include <ibex.h>
#include <codac.h>
#include <codac-rob.h>

using namespace cv;
using namespace std;
using namespace ibex;
using namespace codac;

void waypoint_callback(const geometry_msgs::PoseStamped::ConstPtr& msg){
  float w_x, w_y, w_th;
  w_x = msg->pose.position.x;
  w_y = msg->pose.position.y;
  w_th = tf::getYaw(msg->pose.orientation);

  vibes::drawVehicle(w_x, w_y, w_th*180./M_PI, 0.3, "red");
}

void state_loc_callback(const tiles_loc::State::ConstPtr& msg){
  IntervalVector state_loc({
    {msg->x1_lb, msg->x1_ub},
    {msg->x2_lb, msg->x2_ub},
    {msg->x3_lb, msg->x3_ub}}
  );

  vibes::drawBox(state_loc.subvector(0, 1), "blue");
  vibes::drawVehicle(state_loc[0].mid(), state_loc[1].mid(), (state_loc[2].mid())*180./M_PI, 0.3, "blue");
}

void state_pred_callback(const tiles_loc::State::ConstPtr& msg) {
  IntervalVector state_pred({
    {msg->x1_lb, msg->x1_ub},
    {msg->x2_lb, msg->x2_ub},
    {msg->x3_lb, msg->x3_ub}}
  );

  vibes::drawBox(state_pred.subvector(0, 1), "green");
  vibes::drawVehicle(state_pred[0].mid(), state_pred[1].mid(), (state_pred[2].mid())*180./M_PI, 0.3, "green");
}

void pose_callback(const geometry_msgs::Pose& msg){
  float x, y, c;
  x = msg.position.x;
  y = msg.position.y;
  c = tf::getYaw(msg.orientation);

  vibes::drawVehicle(x, y, c*180./M_PI, 0.3, "pink");
}

int main(int argc, char **argv){
  vibes::beginDrawing();
  VIBesFigMap fig_map("Map");
  vibes::setFigureProperties("Map",vibesParams("x", 10, "y", -10, "width", 700, "height", 700));
  vibes::axisLimits(-10, 10, -10, 10, "Map");
  fig_map.show();

  ros::init(argc, argv, "viewer_node");

  ros::NodeHandle n;

  ros::Subscriber sub_waypoint = n.subscribe("waypoint", 1000, waypoint_callback);
  ros::Subscriber sub_state_loc = n.subscribe("state_loc", 1000, state_loc_callback);
  ros::Subscriber sub_state_pred = n.subscribe("state_pred", 1000, state_pred_callback);
  ros::Subscriber sub_pose = n.subscribe("pose", 1000, pose_callback);

  ros::spin();

  vibes::endDrawing();
  return 0;
}
