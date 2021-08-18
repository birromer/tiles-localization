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

#define TILE_SIZE 0.166
int num_imgs = 10000;

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

// headers for using root
#include <thread>
#include <chrono>
#include <stdexcept>
#include <memory>
#include "TCanvas.h"
#include "TGraph.h"
#include "TApplication.h"
#include "TAxis.h"

using namespace std::chrono_literals;

ibex::IntervalVector state_loc(3, ibex::Interval::ALL_REALS);
ibex::IntervalVector state_pred(3, ibex::Interval::ALL_REALS);
ibex::IntervalVector observation(3, ibex::Interval::ALL_REALS);
double pose_1, pose_2, pose_3;
ofstream file_eq_yx;
ofstream file_eq_yp;
//ofstream file_gt;

auto c1 = std::make_unique<TCanvas>("c1", "Equivalence equations");
auto f1 = std::make_unique<TGraph>(num_imgs);
auto f2 = std::make_unique<TGraph>(num_imgs);
auto f3 = std::make_unique<TGraph>(num_imgs);
auto f4 = std::make_unique<TGraph>(num_imgs);
auto f5 = std::make_unique<TGraph>(num_imgs);
auto f6 = std::make_unique<TGraph>(num_imgs);
auto f7 = std::make_unique<TGraph>(num_imgs);
vector<vector<double>> sim_data;
vector<vector<double>> sim_ground_truth;
int curr_img = 0;

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

  double expected_1 = (pose_1/TILE_SIZE - floor(pose_1/TILE_SIZE))*TILE_SIZE;
  double expected_2 = (pose_2/TILE_SIZE - floor(pose_2/TILE_SIZE))*TILE_SIZE;
  double expected_3 = ((pose_3-M_PI/4)/(M_PI/2) - floor((pose_3-M_PI/4)/(M_PI/2))-0.5) * 2*M_PI/4;  // modulo without function as it is in radians

  ROS_WARN("[VIEWER] Using state: x1 [%f] | x2 [%f] | x3 [%f]", x[0].mid(), x[1].mid(), x[2].mid());
  ROS_WARN("[VIEWER] Using parameters: y1 [%f] | y2 [%f] | y3 [%f]", y[0].mid(), y[1].mid(), y[2].mid());
  ROS_WARN("[VIEWER] Using truth: p1 [%f] | p2 [%f] | p3 [%f]", pose_1, pose_2, pose_3);

  // ground truth and parameters should have near 0 value in the equivalency equations
  double sim1_eq1 = sin(M_PI*(pose_1-y[0].mid())/TILE_SIZE);
  double sim1_eq2 = sin(M_PI*(pose_2-y[1].mid())/TILE_SIZE);
  double sim1_eq3 = sin(pose_3-y[2].mid());

  double sim2_eq1 = sin(M_PI*(pose_2-y[0].mid())/TILE_SIZE);
  double sim2_eq2 = sin(M_PI*(pose_1-y[1].mid())/TILE_SIZE);
  double sim2_eq3 = cos(pose_3-y[2].mid());

  file_eq_yp << sim1_eq1 << "," << sim1_eq2 << "," << sim1_eq3 << "," << sim2_eq1 << "," << sim2_eq2 << "," << sim2_eq3 << endl;

  ROS_INFO("[VIEWER] Equivalence equations 1:\nsin(pi*(y1-z1)) = [%f]\nsin(pi*(y2-z2)) = [%f]\nsin(y2-z2) = [%f]\n", sim1_eq1, sim1_eq2, sim1_eq3);
  ROS_INFO("[VIEWER] Equivalence equations 2:\nsin(pi*(y1-z2)) = [%f]\nsin(pi*(y2-z1)) = [%f]\ncos(y2-z1) = [%f]\n", sim2_eq1, sim2_eq2, sim2_eq3);

  // plot similarity equations
  vector<double> s{sim1_eq1, sim1_eq2, sim1_eq3, sim2_eq1, sim2_eq2, sim2_eq3};

  if (sim_data.size() == curr_img){
    sim_data.push_back(s);
  } else {
    sim_data.at(curr_img) = s;
  }

  // redraw graph
  for (int i=0; i < curr_img; i++) {
    if (i < curr_img) {
      f1->SetPoint(i, i, sim_data[i][0]);
      f2->SetPoint(i, i, sim_data[i][1]);
      f3->SetPoint(i, i, sim_data[i][2]);
      f4->SetPoint(i, i, sim_data[i][3]);
      f5->SetPoint(i, i, sim_data[i][4]);
      f6->SetPoint(i, i, sim_data[i][5]);
    } else {
      f1->SetPoint(i, i, 0);
      f2->SetPoint(i, i, 0);
      f3->SetPoint(i, i, 0);
      f4->SetPoint(i, i, 0);
      f5->SetPoint(i, i, 0);
      f6->SetPoint(i, i, 0);
    }
  }
  f1->RemovePoint(curr_img+1);
  f2->RemovePoint(curr_img+1);
  f3->RemovePoint(curr_img+1);
  f4->RemovePoint(curr_img+1);
  f5->RemovePoint(curr_img+1);
  f6->RemovePoint(curr_img+1);

  // notify ROOT that the plots have been modified and needs update
  c1->cd(1);
  c1->Update();
  c1->Pad()->Draw();
  c1->cd(2);
  c1->Update();
  c1->Pad()->Draw();
  c1->cd(3);
  c1->Update();
  c1->Pad()->Draw();
  c1->cd(4);
  c1->Update();
  c1->Pad()->Draw();
  c1->cd(5);
  c1->Update();
  c1->Pad()->Draw();
  c1->cd(6);
  c1->Update();
  c1->Pad()->Draw();

  curr_img += 1;
}

int main(int argc, char **argv){
  // ----------------- ROOT setup ----------------- //
  TApplication rootapp("viz", &argc, argv);

  // c1, f1, f2, f3, f4, f5, f6 declared globally
  c1->SetWindowSize(1550, 700);

  f1->SetTitle("sin(pi*(y1-z1)/tile_size)");
  f1->GetXaxis()->SetTitle("Iteration");
  f1->GetYaxis()->SetTitle("Similarity score");
  f1->SetMinimum(-1);
  f1->SetMaximum(1);

  f2->SetTitle("sin(pi*(y2-z2)/tile_size)");
  f2->GetXaxis()->SetTitle("Iteration");
  f2->GetYaxis()->SetTitle("Similarity score");
  f2->SetMinimum(-1);
  f2->SetMaximum(1);

  f3->SetTitle("sin(y3-z3)");
  f3->GetXaxis()->SetTitle("Iteration");
  f3->GetYaxis()->SetTitle("Similarity score");
  f3->SetMinimum(-1);
  f3->SetMaximum(1);

  f4->SetTitle("sin(pi*(y1-z2)/tile_size)");
  f4->GetXaxis()->SetTitle("Iteration");
  f4->GetYaxis()->SetTitle("Similarity score");
  f4->SetMinimum(-1);
  f4->SetMaximum(1);

  f5->SetTitle("sin(pi*(y2-z1)/tile_size)");
  f5->GetXaxis()->SetTitle("Iteration");
  f5->GetYaxis()->SetTitle("Similarity score");
  f5->SetMinimum(-1);
  f5->SetMaximum(1);

  f6->SetTitle("cos(y3-z3)");
  f6->GetXaxis()->SetTitle("Iteration");
  f6->GetYaxis()->SetTitle("Similarity score");
  f6->SetMinimum(-1);
  f6->SetMaximum(1);

  // divide the canvas into two vertical sub-canvas
  c1->Divide(2, 3);

  // "Register" the plots for each canvas slot
  c1->cd(1); // Set current canvas to canvas 1 (yes, 1 based indexing)
  f1->Draw();
  c1->cd(3);
  f2->Draw();
  c1->cd(5);
  f3->Draw();
  c1->cd(2);
  f4->Draw();
  c1->cd(4);
  f5->Draw();
  c1->cd(6);
  f6->Draw();
  // ------------------------------------------------ //

  // ----------------- VIBES setup ----------------- //
  vibes::beginDrawing();
  codac::VIBesFigMap fig_map("Map");
  vibes::setFigureProperties("Map",vibesParams("x", 10, "y", -10, "width", 700, "height", 700));
  vibes::axisLimits(-10, 10, -10, 10, "Map");
  fig_map.show();
  // ------------------------------------------------ //

  file_eq_yx.open("/home/birromer/ros/data_tiles/temp/eq_yx.csv", fstream::in | fstream::out | fstream::trunc);
//  file_eq_yp.open("/home/birromer/ros/data_tiles/temp/eq_yp.csv", fstream::in | fstream::out | fstream::trunc);
//  file_gt.open("/home/birromer/ros/data_tiles/temp/gt.csv", fstream::in | fstream::out | fstream::trunc);

  file_eq_yx << "sim1_eq1" << "," << "sim1_eq2" << "," << "sim1_eq3" << "," << "sim2_eq1" << "," << "sim2_eq2" << "," << "sim2_eq3" << endl;
//  file_eq_yp << "sim1_eq1" << "," << "sim1_eq2" << "," << "sim1_eq3" << "," << "sim2_eq1" << "," << "sim2_eq2" << "," << "sim2_eq3" << endl;
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
