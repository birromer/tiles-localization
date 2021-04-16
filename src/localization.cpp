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

#include <ibex.h>
#include <tubex.h>
#include <tubex-rob.h>


void state_pred_callback(const geometry_msgs::PoseStamped::ConstPtr& msg);
void observation_callback(const tiles_loc::Observation::ConstPtr& msg);


double x_x, x_y, x_th;  // predicted state of the robot, from the base node
double obs_1, obs_2, obs_3;


int main(int argc, char **argv) {
  ros::init(argc, argv, "localization_node");
  ros::NodeHandle n;

  ros::Rate loop_rate(25);  // 25Hz frequency

  IntervalVector state(3, Interval::ALL_REALS); // estimated state of the robot, from the contractors

  double x1_pred, x2_pred, x3_pred;  // predicted state of the robot, from the equations
  double x1_loc, x2_loc, x3_loc;     // estimated state of the robot, from the contracted intervals
  double y1, y2, y3;                 // observed parameters

  // --- subscribers --- //
  // subscriber to predicted state and measured observation from base
  ros::Subscriber sub_state_pred = n.subscribe("state_pred", 1000, state_pred_callback);
  ros::Subscriber sub_y = n.subscribe("observation", 1000, observation_callback);
  // ------------------ //

  // --- publishers --- //
  // publisher of the estimated state back to base node
  ros::Publisher pub_state_loc = n.advertise<geometry_msgs::PoseStamped::ConstPtr>("state_loc", 1000);
  // ------------------ //

  while (ros::ok()) {
    x1_pred = x_x, x2_pred = x_y, x3_pred = x_th;  // start with the last state contracted from the localization
    y1 = obs_1, y2 = obs_2, y3 = obs_3;            // use last observed parameters from the image

    IntervalVector X(3, Interval::ALL_REALS);
    X[0] = Interval(x1_pred).inflate(0.1);
    X[1] = Interval(x2_pred).inflate(0.1);
    X[2] = Interval(x3_pred).inflate(0.1);

    IntervalVector box0(6, Interval::ALL_REALS);
    IntervalVector box1(6, Interval::ALL_REALS);

    box0[0] = X[0], box0[1] = X[1], box0[2] = X[2], box0[3] = Interval(medx).inflate(0.03), box0[4] = Interval(medy).inflate(0.03), box0[5] = Interval(alpha_median).inflate(0.1); //X[2];
    box1[0] = X[0], box1[1] = X[1], box1[2] = X[2], box1[3] = Interval(medx).inflate(0.03), box1[4] = Interval(medy).inflate(0.03), box1[5] = Interval(alpha_median).inflate(0.1); //X[2];

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
      state[0] = box[0];
      state[1] = box[1];
      state[2] = box[2];

      float a = (state[2].mid())*180./M_PI;
      std::cout << "angle robot: " << a << std::endl;

//      vibes::drawBox(state.subvector(0, 1), "pink");
//      vibes::drawVehicle(state[0].mid(), state[1].mid(), a, 0.4, "blue");
    }

    std::cout << "contracted state: " << state << std::endl << std::endl;

    // publish evolved state and observation, to be used only by the localization node
    geometry_msgs::PoseStamped state_loc_msg = state_to_pose_stamped(x1_loc, x2_loc, x3_loc);
    pub_state_pred.publish(state_loc_msg);

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
