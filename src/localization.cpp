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


int main(int argc, char **argv) {
  ros::init(argc, argv, "localization_node");
  ros::NodeHandle n;

  ros::Rate loop_rate(25);  // 25Hz frequency

  double xloc_1, xloc_2, xloc_3;  // estimated state of the robot

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
    x1 = xloc_1, x2 = xloc_2, x3 = xloc_3;  // start with the last state contracted from the localization
    y1 = obs_1, y2 = obs_2, y3 = obs_3;     // use last observed parameters from the image
    u1 = cmd_1, u2 = cmd_2;                 // use last input from command

    // publish current, unevolved state to be used by the control and viewer nodes
    geometry_msgs::PoseStamped state_msg = state_to_pose_stamped(x1, x2, x3);
    pub_state.publish(state_msg);

    // evolve state according to input and state equations
    integration_euler(x1, x2, x3, u1, u2, dt);

    // publish evolved state and observation, to be used only by the localization node
    geometry_msgs::PoseStamped state_pred_msg = state_to_pose_stamped(x1, x2, x3);
    pub_state_pred.publish(state_pred_msg);

    tiles_loc::Observation observation_msg = y_to_observation(y1, y2, y3);
    pub_y.publish(observation_msg);

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}

void calc_new_pos(std::vector<Vec4i> lines) {
  //cout << "state début : " << state << " | " << state[2]*180/M_PI << endl;

  //conversion from cartesian to polar form :
  float x1,x2,y1,y2;
  vector<float> Msn, Mew;

  alpha_median = alpha_median;
  cout << "alpha_median : " << alpha_median*180/M_PI << endl;

  //à la limite, on a le quart qui fluctue donc on veut éviter ça
  if(nn ==0) {
    if(abs(abs(alpha_median)-M_PI/4.)<0.1) {
      if(alpha_median >= 0 & last_alpha_median < 0) {
        quart -=1;
        nn = 1;
      } else if (alpha_median <= 0 & last_alpha_median > 0) {
        quart +=1;
        nn = 1;
      }
    }
  } else {
    nn+=1;
  }
  if(nn == 100) {
    nn =0;
  }
  last_alpha_median = alpha_median;

  cout << "________quart : " << quart << " n : " << nn << " " << abs(abs(alpha_median)-M_PI/4.) << endl;

  Mat rot = Mat::zeros(Size(frame_width, frame_height), CV_8UC3);

  for(int i=0; i<lines.size(); i++) {
    cv::Vec4i l = lines[i];
    x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
    //cout << x1 << " | " << y1<< " | " << x2 << " | " << y2 << endl;
    //cout << frame_height << " | " << frame_width << endl;

    //translation pour centrer les points autour de 0
    x1-=frame_width/2., y1-=frame_height/2., x2-=frame_width/2., y2-=frame_height/2.;

    //rotation autour de 0
    float alpha = atan2(y2-y1,x2-x1);

    float angle = modulo(alpha+M_PI/4., M_PI/2)-M_PI/4.;
    angle += quart*M_PI/2.;

    double s = sin(-angle);
    double c = cos(-angle);

    float x1b=x1, y1b=y1, x2b=x2, y2b=y2;

    x1 = x1b*c - y1b*s,
    y1 = +x1b*s + y1b*c;

    x2 = x2b*c - y2b*s,
    y2 = x2b*s + y2b*c;

    //translation pour l'affichage
    x1+=frame_width/2., y1+=frame_height/2., x2+=frame_width/2., y2+=frame_height/2.;

    float alpha2 = atan2(y2-y1,x2-x1);
    float x11=x1, y11=y1, x22=x2, y22=y2;

    //calcul pour medx et medy
    x1=((float)l[0]-frame_width/2.)*scale_pixel, y1=((float)l[1]-frame_height/2.)*scale_pixel;
    x2=((float)l[2]-frame_width/2.)*scale_pixel, y2=((float)l[3]-frame_height/2.)*scale_pixel;

    ROS_WARN("->> aqui x1 = [%f] | x2 = [%f]", x1, x2);
    ROS_WARN("->> aqui y1 = [%f] | y2 = [%f]", y1, y2);

    float d = ((x2-x1)*(y1)-(x1)*(y2-y1)) / sqrt(pow(x2-x1, 2)+pow(y2-y1, 2));

    float val;
    val = (d+0.5)-floor(d+0.5)-0.5;

    if (abs(cos(alpha2)) < 0.2) {
      line(rot, cv::Point(x11, y11), cv::Point(x22, y22), Scalar(255, 255, 255), 1, LINE_AA);
      Msn.push_back(val);

    } else if (abs(sin(alpha2)) < 0.2) {
      line(rot, cv::Point(x11, y11), cv::Point(x22, y22), Scalar(0, 0, 255), 1, LINE_AA);
      Mew.push_back(val);

    }
  }

  cv::imshow("rot", rot);

  alpha_median = alpha_median + quart*M_PI/2;
  alpha_median = modulo(alpha_median, 2*M_PI);

  cout << "alpha_median : " << alpha_median*180/M_PI << " " << quart%2<< endl;

  float medx = sign(cos(state[2].mid()))*median(Mew);
  float medy = sign(sin(state[2].mid()))*median(Msn);

  IntervalVector X(3, Interval::ALL_REALS);

  X[0] = Interval(state[0].mid()).inflate(percent*size_carrelage_x/2.);
  X[1] = Interval(state[1].mid()).inflate(percent*size_carrelage_y/2.);
  X[2] = Interval(modulo(alpha_median, 2*M_PI)).inflate(0.1);

  //normalisation :
  X[0] = X[0]/size_carrelage_x;
  X[1] = X[1]/size_carrelage_y;

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
  box[0] = box0[0]|box1[0];
  box[1] = box0[1]|box1[1];
  box[2] = box0[2]|box1[2];

  if(box[0].is_empty() or box[1].is_empty()){
    cout << "X empty" << endl;
  }else{

    state[0] = box[0];
    state[1] = box[1];

    state[0] = state[0]*size_carrelage_x;
    state[1] = state[1]*size_carrelage_y;
    state[2] = box[2];

    float a = (state[2].mid())*180./M_PI;
    cout << "angle robot : " << a << endl;
    vibes::drawBox(state.subvector(0, 1), "pink");
    vibes::drawVehicle(state[0].mid(), state[1].mid(), a, 0.4, "blue");
  }
  cout << "state fin : " << state << endl;
  cout << endl;
}
