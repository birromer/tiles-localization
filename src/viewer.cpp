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
#include <iostream>
#include <fstream>

#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Quaternion.h"
#include "geometry_msgs/PoseStamped.h"
#include "tf/tf.h"
#include <tf2/LinearMath/Quaternion.h>

#include "tiles_loc/State.h"
#include "tiles_loc/Observation.h"

#include <ibex.h>
#include <codac.h>
#include <codac-rob.h>

using namespace cv;
using namespace std;
using namespace ibex;
using namespace codac;

ibex::IntervalVector state_loc(3, ibex::Interval::ALL_REALS);
ibex::IntervalVector state_pred(3, ibex::Interval::ALL_REALS);
ibex::IntervalVector observation(3, ibex::Interval::ALL_REALS);
double pose_1, pose_2, pose_3;
ofstream file_eq_yx;
ofstream file_eq_yp;
//ofstream file_gt;

void waypoint_callback(const geometry_msgs::PoseStamped::ConstPtr& msg){
  float w_x, w_y, w_th;
  w_x = msg->pose.position.x;
  w_y = msg->pose.position.y;
  w_th = tf::getYaw(msg->pose.orientation);

  vibes::drawVehicle(w_x, w_y, w_th*180./M_PI, 0.3, "red");
}

void state_loc_callback(const tiles_loc::State::ConstPtr& msg){
  state_loc[0] = ibex::Interval(msg->x1_lb, msg->x1_ub);
  state_loc[1] = ibex::Interval(msg->x2_lb, msg->x2_ub);
  state_loc[2] = ibex::Interval(msg->x3_lb, msg->x3_ub);

  vibes::drawBox(state_loc.subvector(0, 1), "blue");
  vibes::drawVehicle(state_loc[0].mid(), state_loc[1].mid(), (state_loc[2].mid())*180./M_PI, 0.3, "blue");
}

void state_pred_callback(const tiles_loc::State::ConstPtr& msg) {
  state_pred[0] = ibex::Interval(msg->x1_lb, msg->x1_ub);
  state_pred[1] = ibex::Interval(msg->x2_lb, msg->x2_ub);
  state_pred[2] = ibex::Interval(msg->x3_lb, msg->x3_ub);

  vibes::drawBox(state_pred.subvector(0, 1), "green");
  vibes::drawVehicle(state_pred[0].mid(), state_pred[1].mid(), (state_pred[2].mid())*180./M_PI, 0.3, "green");
}

void pose_callback(const geometry_msgs::Pose& msg){
  pose_1 = msg.position.x;
  pose_2 = msg.position.y;
  pose_3 = tf::getYaw(msg.orientation);

  vibes::drawVehicle(pose_1, pose_2, pose_3*180./M_PI, 0.3, "pink");
}

void observation_callback(const tiles_loc::Observation::ConstPtr& msg) {
  observation[0] = ibex::Interval(msg->y1_lb, msg->y1_ub);
  observation[1] = ibex::Interval(msg->y2_lb, msg->y2_ub);
  observation[2] = ibex::Interval(msg->y3_lb, msg->y3_ub);

  ibex::IntervalVector x = state_loc;
  ibex::IntervalVector y = observation;

  ROS_WARN("[VIEWER] Using state: x1 [%f] | x2 [%f] | x3 [%f]", x[0].mid(), x[1].mid(), x[2].mid());
  ROS_WARN("[VIEWER] Using parameters: y1 [%f] | y2 [%f] | y3 [%f]", y[0].mid(), y[1].mid(), y[2].mid());
  ROS_WARN("[VIEWER] Using truth: p1 [%f] | p2 [%f] | p3 [%f]", pose_1, pose_2, pose_3);

  // comparing Y with X
  double sim1_eq1 = sin(M_PI*(y[0].mid()-x[0].mid()));
  double sim1_eq2 = sin(M_PI*(y[1].mid()-x[1].mid()));
  double sim1_eq3 = sin(y[2].mid()-x[2].mid());

  double sim2_eq1 = sin(M_PI*(y[0].mid()-x[1].mid()));
  double sim2_eq2 = sin(M_PI*(y[1].mid()-x[0].mid()));
  double sim2_eq3 = cos(y[2].mid()-x[2].mid());

  file_eq_yx << sim1_eq1 << "," << sim1_eq2 << "," << sim1_eq3 << "," << sim2_eq1 << "," << sim2_eq2 << "," << sim2_eq3 << endl;

  ROS_INFO("[VIEWER] Equivalence equations 1:\nsin(pi*(y1-z1)) = [%f]\nsin(pi*(y2-z2)) = [%f]\nsin(y2-z2) = [%f]\n", sim1_eq1, sim1_eq2, sim1_eq3);
  ROS_INFO("[VIEWER] Equivalence equations 2:\nsin(pi*(y1-z2)) = [%f]\nsin(pi*(y2-z1)) = [%f]\ncos(y2-z1) = [%f]\n", sim2_eq1, sim2_eq2, sim2_eq3);

  // comparing Y with pose
  sim1_eq1 = sin(M_PI*(y[0].mid()-pose_1));
  sim1_eq2 = sin(M_PI*(y[1].mid()-pose_2));
  sim1_eq3 = sin(y[2].mid()-pose_3);

  sim2_eq1 = sin(M_PI*(y[0].mid()-pose_2));
  sim2_eq2 = sin(M_PI*(y[1].mid()-pose_1));
  sim2_eq3 = cos(y[2].mid()-pose_3);

  file_eq_yp << sim1_eq1 << "," << sim1_eq2 << "," << sim1_eq3 << "," << sim2_eq1 << "," << sim2_eq2 << "," << sim2_eq3 << endl;

  ROS_INFO("[VIEWER] Equivalence equations 1:\nsin(pi*(y1-z1)) = [%f]\nsin(pi*(y2-z2)) = [%f]\nsin(y2-z2) = [%f]\n", sim1_eq1, sim1_eq2, sim1_eq3);
  ROS_INFO("[VIEWER] Equivalence equations 2:\nsin(pi*(y1-z2)) = [%f]\nsin(pi*(y2-z1)) = [%f]\ncos(y2-z1) = [%f]\n", sim2_eq1, sim2_eq2, sim2_eq3);

//  file_gt << pose_1 << "," << pose_2 << "," << pose_3 << endl;

  // plot similarity equations
}

int main(int argc, char **argv){
  vibes::beginDrawing();
  VIBesFigMap fig_map("Map");
  vibes::setFigureProperties("Map",vibesParams("x", 10, "y", -10, "width", 700, "height", 700));
  vibes::axisLimits(-10, 10, -10, 10, "Map");
  fig_map.show();

  file_eq_yx.open("/home/birromer/ros/data_tiles/eq_yx.csv", fstream::in | fstream::out | fstream::trunc);
  file_eq_yp.open("/home/birromer/ros/data_tiles/eq_yp.csv", fstream::in | fstream::out | fstream::trunc);
//  file_gt.open("/home/birromer/ros/data_tiles/gt.csv", fstream::in | fstream::out | fstream::trunc);

  file_eq_yx << "sim1_eq1" << "," << "sim1_eq2" << "," << "sim1_eq3" << "," << "sim2_eq1" << "," << "sim2_eq2" << "," << "sim2_eq3" << endl;
  file_eq_yp << "sim1_eq1" << "," << "sim1_eq2" << "," << "sim1_eq3" << "," << "sim2_eq1" << "," << "sim2_eq2" << "," << "sim2_eq3" << endl;
//  file_gt << "x" << "," << "y" << "," << "theta" << endl;

  ros::init(argc, argv, "viewer_node");

  ros::NodeHandle n;

  ros::Subscriber sub_waypoint = n.subscribe("waypoint", 1000, waypoint_callback);
  ros::Subscriber sub_state_loc = n.subscribe("state_loc", 1000, state_loc_callback);
  ros::Subscriber sub_state_pred = n.subscribe("state_pred", 1000, state_pred_callback);
  ros::Subscriber sub_observation = n.subscribe("observation", 1000, observation_callback);
  ros::Subscriber sub_pose = n.subscribe("pose", 1000, pose_callback);

  ros::spin();

  vibes::endDrawing();
  file_eq_yx.close();
  file_eq_yp.close();
//  file_gt.close();

  return 0;
}
