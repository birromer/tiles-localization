/*
** This node is responsible for the localization method of the robot,
** estimating its state with contractors that use the evolved pose of the robot and
** the observed parameters from the input images
**
** Subscribers:
**   - tiles_loc::State state_pred_dt      // the change within dt of the predicted state from the state model
**   - tiles_loc::Observation observation  // the observation vector, processed from the input image
**
** Publishers:
**   - geometry_msgs::PoseStamped state_loc     // the estimation of the robot's state
*/

#include <ros/ros.h>
#include <vector>
#include <cmath>

#include "std_msgs/Int32.h"
#include "geometry_msgs/Quaternion.h"
#include "geometry_msgs/PoseStamped.h"

#include <tf2/LinearMath/Quaternion.h>
#include "tf/tf.h"

#include "tiles_loc/Observation.h"
#include "tiles_loc/State.h"

#include <ibex.h>
#include <codac.h>
#include <codac-rob.h>

ibex::IntervalVector state_pred_dt(3, ibex::Interval::ALL_REALS);  // change in the state within dt, from the state equations
ibex::IntervalVector observation(3, ibex::Interval::ALL_REALS);    // observed parameters, from the base node callback
double gt_1, gt_2, gt_3;  // ground truth //NOTE: only for debugging//
bool first_pose = true;
double offset_pose_1;
double offset_pose_2;

void state_pred_dt_callback(const tiles_loc::State::ConstPtr& msg);
void observation_callback(const tiles_loc::Observation::ConstPtr& msg);
void pose_callback(const geometry_msgs::Pose& msg);  //NOTE: used only for debugging
tiles_loc::State state_to_msg(ibex::IntervalVector state);

#define TILE_SIZE 0.166 // 1.0 //

int main(int argc, char **argv) {
  ros::init(argc, argv, "localization_node");
  ros::NodeHandle n;

  ros::Rate loop_rate(50);  // 50Hz frequency

  ibex::IntervalVector x(3, ibex::Interval::ALL_REALS);  // state of the robot
  ibex::IntervalVector y(3, ibex::Interval::ALL_REALS);  // observed parameters
  double pose_1, pose_2, pose_3;                         // NOTE: debugging only

  double x1, x2, x3;  // initial parameters for the state
  n.param<double>("pos_x_init", x1, 0);
  n.param<double>("pos_y_init", x2, 0);
  n.param<double>("pos_th_init", x3, 0);

  // starting state is known
  x[0] = ibex::Interval(x1, x1);
  x[1] = ibex::Interval(x2, x2);
  x[2] = ibex::Interval(x3, x3);

  // --- subscribers --- //
  // subscriber to predicted state and measured observation from base
  ros::Subscriber sub_state_pred_dt = n.subscribe("state_pred_dt", 1000, state_pred_dt_callback);
  ros::Subscriber sub_y = n.subscribe("observation", 1000, observation_callback);

  //NOTE: ground truth used only for debugging
  ros::Subscriber sub_pose = n.subscribe("pose", 1000, pose_callback);
  // ------------------ //

  // --- publishers --- //
  // publisher of the estimated states back to base node
  ros::Publisher pub_state_pred = n.advertise<tiles_loc::State>("state_pred", 1000);
  ros::Publisher pub_state_loc = n.advertise<tiles_loc::State>("state_loc", 1000);
  // ------------------ //

  while (ros::ok()) {
    //NOTE: ground truth used for debugging only
    pose_1 = gt_1;
    pose_2 = gt_2;
    pose_3 = gt_3;

    // predict the state according to the state equations
    x[0] = x[0] + state_pred_dt[0];
    x[1] = x[1] + state_pred_dt[1];
    x[2] = state_pred_dt[2];
//    x[0] = pose_1;
//    x[1] = pose_2;
//    x[2] = pose_3;

//    x.inflate(0.1);

    // publish predicted state
    tiles_loc::State state_pred_msg = state_to_msg(x);
    pub_state_pred.publish(state_pred_msg);

    x[0] = codac::Interval(pose_1, pose_1).inflate(0.04);
    x[1] = codac::Interval(pose_2, pose_2).inflate(0.04);

    // use last observed parameters from the image
    y[0] = observation[0];
    y[1] = observation[1];
    y[2] = observation[2];

    ibex::IntervalVector box0(6, ibex::Interval::ALL_REALS);
    ibex::IntervalVector box1(6, ibex::Interval::ALL_REALS);

    box0[0] = x[0], box0[1] = x[1], box0[2] = x[2], box0[3] = y[0], box0[4] = y[1], box0[5] = y[2];
    box1[0] = x[0], box1[1] = x[1], box1[2] = x[2], box1[3] = y[0], box1[4] = y[1], box1[5] = y[2];

    char f1_char[100];
    char f2_char[100];
    snprintf(f1_char, 100, "(sin(pi*(x[0]-y[0])/%.3f) ; sin(pi*(x[1]-y[1])/%.3f) ; sin(x[2]-y[2]))", TILE_SIZE, TILE_SIZE);
    snprintf(f2_char, 100, "(sin(pi*(x[0]-y[1])/%.3f) ; sin(pi*(x[1]-y[0])/%.3f) ; cos(x[2]-y[2]))", TILE_SIZE, TILE_SIZE);

    ibex::Function f1("x[3]", "y[3]", f1_char);
    ibex::Function f2("x[3]", "y[3]", f2_char);

    ibex::CtcFwdBwd c1(f1);
    ibex::CtcFwdBwd c2(f2);

    c1.contract(box0);
    c2.contract(box1);

    ibex::IntervalVector box(3, ibex::Interval::ALL_REALS);
    box[0] = box0[0] | box1[0];
    box[1] = box0[1] | box1[1];
    box[2] = box0[2] | box1[2];

    if(box[0].is_empty() or box[1].is_empty()) {
      ROS_WARN("[LOCALIZATION] Could not contract the state.");
    } else {
      x[0] = box[0];
      x[1] = box[1];
      x[2] = box[2];
    }

//    x[0] = ibex::Interval(pose_1, pose_1).inflate(0.1);
//    x[1] = ibex::Interval(pose_2, pose_2).inflate(0.1);
//    x[2] = ibex::Interval(pose_3, pose_3).inflate(0.1);

    // publish evolved state and observation, to be used only by the localization node
    tiles_loc::State state_loc_msg = state_to_msg(x);
    pub_state_loc.publish(state_loc_msg);

//    ROS_INFO("[LOCALIZATION] Sent estimated state: x1 ([%f],[%f]) | x2 ([%f],[%f]) | x3 ([%f],[%f])",
//             x[0].lb(), x[0].ub(), x[1].lb(), x[1].ub(), x[2].lb(), x[2].ub());

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}

tiles_loc::State state_to_msg(ibex::IntervalVector state) {
    tiles_loc::State msg;
    msg.x1_lb = state[0].lb();
    msg.x1_ub = state[0].ub();
    msg.x2_lb = state[1].lb();
    msg.x2_ub = state[1].ub();
    msg.x3_lb = state[2].lb();
    msg.x3_ub = state[2].ub();
    return msg;
}

void state_pred_dt_callback(const tiles_loc::State::ConstPtr& msg) {

  state_pred_dt[0] = ibex::Interval(msg->x1_lb, msg->x1_ub);
  state_pred_dt[1] = ibex::Interval(msg->x2_lb, msg->x2_ub);
  state_pred_dt[2] = ibex::Interval(msg->x3_lb, msg->x3_ub);

//  ROS_INFO("[LOCALIZATION] Received predicted state change-> x1: ([%f],[%f]) | x2: ([%f],[%f]) | x3: ([%f],[%f])",
//           msg->x1_lb, msg->x1_ub, msg->x2_lb, msg->x2_ub, msg->x3_lb, msg->x3_ub);
}

void observation_callback(const tiles_loc::Observation::ConstPtr& msg) {
  observation[0] = ibex::Interval(msg->y1_lb, msg->y1_ub);
  observation[1] = ibex::Interval(msg->y2_lb, msg->y2_ub);
  observation[2] = ibex::Interval(msg->y3_lb, msg->y3_ub);

//  ROS_INFO("[LOCALIZATION] Received observation -> y1: ([%f],[%f]) | y2: ([%f],[%f]) | y3: ([%f],[%f])",
//           msg->y1_lb, msg->y1_ub, msg->y2_lb, msg->y2_ub, msg->y3_lb, msg->y3_ub);

  if (observation[0].is_empty() && observation[1].is_empty()) {
    //ROS_WARN("[LOCALIZATION] Observation is empty.");
  }
}

//NOTE: Used for debugging only
void pose_callback(const geometry_msgs::Pose& msg){
  if (first_pose) {
    offset_pose_1 = -msg.position.x;
    offset_pose_2 = -msg.position.y;
    first_pose = false;
  }
  gt_1 = msg.position.x + offset_pose_1;
  gt_2 = msg.position.y + offset_pose_2;
  gt_3 = tf::getYaw(msg.orientation);
}
