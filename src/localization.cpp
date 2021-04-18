/*
** This node is responsible for the localization method of the robot,
** estimating its state with contractors that use the evolved pose of the robot and
** the observed parameters from the input images
**
** Subscribers:
**   - geometry_msgs::PoseStamped state_pred  // the predicted state from the state model
**   - tiles_loc::Observation observation     // the observation vector, processed from the input image
**
** Publishers:
**   - geometry_msgs::PoseStamped state_loc     // the estimation of the robot's state
*/

#include <ros/ros.h>
#include <vector>
#include <cmath>

#include "geometry_msgs/Quaternion.h"
#include "geometry_msgs/PoseStamped.h"
#include <tf2/LinearMath/Quaternion.h>

#include "tiles_loc/Observation.h"
#include "tiles_loc/State.h"

#include <ibex.h>
#include <tubex.h>
#include <tubex-rob.h>


void state_pred_callback(const tiles_loc::State::ConstPtr& msg);
void observation_callback(const tiles_loc::Observation::ConstPtr& msg);

tiles_loc::State state_to_msg(ibex::IntervalVector state);


ibex::IntervalVector state_pred(3, ibex::Interval::ALL_REALS);  // predicted state of the robot, from the base node callback
double obs_1, obs_2, obs_3;  // observed parameters, from the base node callback


int main(int argc, char **argv) {
  ros::init(argc, argv, "localization_node");
  ros::NodeHandle n;

  ros::Rate loop_rate(25);  // 25Hz frequency

  IntervalVector x_loc(3, Interval::ALL_REALS);   // estimated state of the robot, from the contractors
  IntervalVector x_pred(3, Interval::ALL_REALS);  // predicted state of the robot, from the equations
  double y1, y2, y3;                              // observed parameters

  // --- subscribers --- //
  // subscriber to predicted state and measured observation from base
  ros::Subscriber sub_state_pred = n.subscribe("state_pred", 1000, state_pred_callback);
  ros::Subscriber sub_y = n.subscribe("observation", 1000, observation_callback);
  // ------------------ //

  // --- publishers --- //
  // publisher of the estimated state back to base node
  ros::Publisher pub_state_loc = n.advertise<tiles_loc::State>("state_loc", 1000);
  // ------------------ //

  while (ros::ok()) {
    // use last observed parameters from the image TODO: add intervals to observations
    y1 = obs_1, y2 = obs_2, y3 = obs_3;

    // start with the last state contracted from the localization
    x_pred[0] = state_pred[0];
    x_pred[1] = state_pred[1];
    x_pred[2] = state_pred[2];

    ibex::IntervalVector box0(6, Interval::ALL_REALS);
    ibex::IntervalVector box1(6, Interval::ALL_REALS);

    // TODO: test having angle from the state, such as if there were a compass
    box0[0] = x_pred[0], box0[1] = x_pred[1], box0[2] = x_pred[2], box0[3] = y1, box0[4] = y2, box0[5] = y3; //X[2];
    box1[0] = x_pred[0], box1[1] = x_pred[1], box1[2] = x_pred[2], box1[3] = y1, box1[4] = y2, box1[5] = y3; //X[2];

    ibex::Function f1("x[3]", "y[3]", "(sin(pi*(x[0]-y[0])) ; sin(pi*(x[1]-y[1])) ; sin(x[2]-y[2]))");
    ibex::Function f2("x[3]", "y[3]", "(sin(pi*(x[0]-y[1])) ; sin(pi*(x[1]-y[0])) ; cos(x[2]-y[2]))");

    ibex::CtcFwdBwd c1(f1);
    ibex::CtcFwdBwd c2(f2);

    c1.contract(box0);
    c2.contract(box1);

    IntervalVector box(3, Interval::ALL_REALS);
    box[0] = box0[0] | box1[0];
    box[1] = box0[1] | box1[1];
    box[2] = box0[2] | box1[2];

    if(box[0].is_empty() or box[1].is_empty()) {
      std::cout << "X empty" << std::endl;

    } else {
      x_loc[0] = box[0];
      x_loc[1] = box[1];
      x_loc[2] = box[2];

      float a = (x_loc[2].mid())*180./M_PI;
      std::cout << "angle robot: " << a << std::endl;
    }

    std::cout << "contracted state: " << x_loc << std::endl << std::endl;

    // publish evolved state and observation, to be used only by the localization node
    tiles_loc::State state_loc_msg = state_to_msg(x_loc);
    pub_state_loc.publish(state_loc_msg);

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

  ROS_WARN("[LOCALIZATION] Received predicted state -> x1: ([%f],[%f]) | x2: ([%f],[%f]) | x3: ([%f],[%f])",
           msg->x1_lb, msg->x1_ub, msg->x2_lb, msg->x2_ub, msg->x3_lb, msg->x3_ub);
}

void observation_callback(const tiles_loc::Observation::ConstPtr& msg) {
  obs_1 = msg->y1;
  obs_2 = msg->y2;
  obs_3 = msg->y3;

  ROS_WARN("[LOCALIZATION] Received observation -> y1: [%f] | y2: [%f] | y3: [%f]",
           msg->y1, msg->y2, msg->y3);
}
