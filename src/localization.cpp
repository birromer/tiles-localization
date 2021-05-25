/*
** This node is responsible for the localization method of the robot,
** estimating its state with contractors that use the evolved pose of the robot and
** the observed parameters from the input images
**
** Subscribers:
**   - tiles_loc::State dt_state_pred      // the change within dt of the predicted state from the state model
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

#include "tiles_loc/Observation.h"
#include "tiles_loc/State.h"

#include <ibex.h>
#include <codac.h>
#include <codac-rob.h>

ibex::IntervalVector d_state_pred(3, ibex::Interval::ALL_REALS);  // change in the state within dt, from the state equations
ibex::IntervalVector observation(3, ibex::Interval::ALL_REALS);    // observed parameters, from the base node callback
double gt_1, gt_2, gt_3;  // ground truth //NOTE: only for debugging//

void state_pred_callback(const tiles_loc::State::ConstPtr& msg);
void observation_callback(const tiles_loc::Observation::ConstPtr& msg);

ibex::IntervalVector integration_euler(ibex::IntervalVector state, ibex::IntervalVector d_state, double dt);
tiles_loc::State state_to_msg(ibex::IntervalVector state);

int main(int argc, char **argv) {
  ros::init(argc, argv, "localization_node");
  ros::NodeHandle n;

  ros::Rate loop_rate(25);  // 25Hz frequency

  ibex::IntervalVector x(3, ibex::Interval::ALL_REALS);  // state of the robot
  ibex::IntervalVector y(3, ibex::Interval::ALL_REALS);  // observed parameters

  // --- subscribers --- //
  // subscriber to predicted state and measured observation from base
  ros::Subscriber sub_state_pred = n.subscribe("state_pred", 1000, state_pred_callback);
  ros::Subscriber sub_y = n.subscribe("observation", 1000, observation_callback);
  // ------------------ //

  // --- publishers --- //
  // publisher of the estimated state back to base node
  ros::Publisher pub_state_loc = n.advertise<tiles_loc::State>("state_loc", 1000);
  // ------------------ //

  double pose_1, pose_2, pose_3;

  while (ros::ok()) {
    //NOTE: ground truth used for debugging only
    gt_1 = pose_1, gt_2 = pose_2, gt_3 = pose_3;

    // get last received derivative of x
    dx[0] = d_state_pred[0];
    dx[1] = d_state_pred[1];
    dx[2] = d_state_pred[2];

    // use last observed parameters from the image
    y[0] = observation[0];
    y[1] = observation[1];
    y[2] = observation[2];

    // predict the state according to state equations
    x = integration_euler(x, dx, dt);

    // start with the last state contracted from the localization
    x_pred[0] = state_pred[0];
    x_pred[1] = state_pred[1];
    x_pred[2] = state_pred[2];

    ibex::IntervalVector box0(6, ibex::Interval::ALL_REALS);
    ibex::IntervalVector box1(6, ibex::Interval::ALL_REALS);

    // TODO: test having angle from the state, such as if there were a compass
    box0[0] = x_pred[0], box0[1] = x_pred[1], box0[2] = x_pred[2], box0[3] = y[0], box0[4] = y[1], box0[5] = y[2]; //X[2];
    box1[0] = x_pred[0], box1[1] = x_pred[1], box1[2] = x_pred[2], box1[3] = y[0], box1[4] = y[1], box1[5] = y[2]; //X[2];

//    box0[0] = x_pred[0], box0[1] = x_pred[1], box0[2] = x_pred[2], box0[3] = x_pred[0], box0[4] = x_pred[1], box0[5] = x_pred[2];
//    box1[0] = x_pred[0], box1[1] = x_pred[1], box1[2] = x_pred[2], box1[3] = x_pred[0], box1[4] = x_pred[1], box1[5] = x_pred[2];

    ibex::Function f1("x[3]", "y[3]", "(sin(pi*(x[0]-y[0])) ; sin(pi*(x[1]-y[1])) ; sin(x[2]-y[2]))");
    ibex::Function f2("x[3]", "y[3]", "(sin(pi*(x[0]-y[1])) ; sin(pi*(x[1]-y[0])) ; cos(x[2]-y[2]))");

    ibex::CtcFwdBwd c1(f1);
    ibex::CtcFwdBwd c2(f2);

    c1.contract(box0);
    c2.contract(box1);

    ibex::IntervalVector box(3, ibex::Interval::ALL_REALS);
    box[0] = box0[0] | box1[0];
    box[1] = box0[1] | box1[1];
    box[2] = box0[2] | box1[2];

    if(box[0].is_empty() or box[1].is_empty()) {
      ROS_WARN("[LOCALIZATION] X is empty");

    } else {
      x_loc[0] = box[0];
      x_loc[1] = box[1];
      x_loc[2] = box[2];
    }

    ROS_WARN("Using parameters: y1 [%f] | y2 [%f] | y3 [%f]", y[0].mid(), y[1].mid(), y[2].mid());
    ROS_WARN("Using truth: p1 [%f] | p2 [%f] | p3 [%f]", pose_1, pose_2, pose_3);
    ROS_WARN("Using state: p1 [%f] | p2 [%f] | p3 [%f]", pose_1, pose_2, pose_3);

    // comparando Y com X
    ROS_INFO("Equivalence equations 1:\nsin(pi*(y1-z1)) = [%f]\nsin(pi*(y2-z2)) = [%f]\nsin(y2-z2) = [%f]\n", sin(M_PI*(y[0].mid()-state[0].mid())), sin(M_PI*(y[1].mid()-state[1].mid())), sin(y[2].mid()-state[2].mid()));
    ROS_INFO("Equivalence equations 2:\nsin(pi*(y1-z2)) = [%f]\nsin(pi*(y2-z1)) = [%f]\ncos(y2-z1) = [%f]\n", sin(M_PI*(y[0].mid()-state[1].mid())), sin(M_PI*(y[1].mid()-state[0].mid())), cos(y[2].mid()-state[2].mid()));

    // comparando Y com pose
//    ROS_INFO("Equivalence equations 1:\nsin(pi*(y1-z1)) = [%f]\nsin(pi*(y2-z2)) = [%f]\nsin(y2-z2) = [%f]\n", sin(M_PI*(y1-pose_1)), sin(M_PI*(y2-pose_2)), sin(y3-pose_3));
//    ROS_INFO("Equivalence equations 2:\nsin(pi*(y1-z2)) = [%f]\nsin(pi*(y2-z1)) = [%f]\ncos(y2-z1) = [%f]\n", sin(M_PI*(y1-pose_2)), sin(M_PI*(y2-pose_1)), cos(y3-pose_3));

//    x_loc[0] = x_pred[0];
//    x_loc[1] = x_pred[1];
//    x_loc[2] = x_pred[2];

    // publish evolved state and observation, to be used only by the localization node
    tiles_loc::State state_loc_msg = state_to_msg(x_loc);
    pub_state_loc.publish(state_loc_msg);

    ROS_INFO("[LOCALIZATION] Sent estimated state: x1 ([%f],[%f]) | x2 ([%f],[%f]) | x3 ([%f],[%f])",
             x_loc[0].lb(), x_loc[0].ub(), x_loc[1].lb(), x_loc[1].ub(), x_loc[2].lb(), x_loc[2].ub());

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

void state_pred_callback(const tiles_loc::State::ConstPtr& msg) {

  state_pred[0] = ibex::Interval(msg->x1_lb, msg->x1_ub);
  state_pred[1] = ibex::Interval(msg->x2_lb, msg->x2_ub);
  state_pred[2] = ibex::Interval(msg->x3_lb, msg->x3_ub);

  ROS_INFO("[LOCALIZATION] Received predicted state -> x1: ([%f],[%f]) | x2: ([%f],[%f]) | x3: ([%f],[%f])",
           msg->x1_lb, msg->x1_ub, msg->x2_lb, msg->x2_ub, msg->x3_lb, msg->x3_ub);
}

void observation_callback(const tiles_loc::Observation::ConstPtr& msg) {
  observation[0] = ibex::Interval(msg->y1_lb, msg->y1_ub);
  observation[1] = ibex::Interval(msg->y2_lb, msg->y2_ub);
  observation[2] = ibex::Interval(msg->y3_lb, msg->y3_ub);

  ROS_INFO("[LOCALIZATION] Received observation -> y1: ([%f],[%f]) | y2: ([%f],[%f]) | y3: ([%f],[%f])",
           msg->y1_lb, msg->y1_ub, msg->y2_lb, msg->y2_ub, msg->y3_lb, msg->y3_ub);

  if (observation[0].is_empty() && observation[1].is_empty()) {
    ROS_WARN("[LOCALIZATION] Observation is empty.");
  }
}

//NOTE: Used for debugging only
void pose_callback(const geometry_msgs::Pose& msg){
  gt_1 = msg.position.x;
  gt_2 = msg.position.y;
  gt_3 = tf::getYaw(msg.orientation);
}
