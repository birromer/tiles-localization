/*
** This node is responsible for centering all other node's operations.
** It processes the image data in order to generate the observation vector and
** evolves the robot position according to the state equations
**
** Subscribers:
**   - tiles_loc::State state_loc        // the estimation of the robot's state
**   - sensor_msgs::ImageConstPtr image  // the image from the robot's camera
**   - geometry_msgs::Pose speed         //  the robot's velocity in translation and rotation in each dimension
**   - std_msgs::Float64 compass         //  the robot's orientation
**
** Publishers:
**   - tiles_loc::State state_pred_dt      // change within dt of the evolved state, predicted from the state equations
**   - tiles_loc::State state              // the current state of the robot, for synchronization
**   - tiles_loc::Observation observation  // the observation vector, processed from the incoming image
*/

#include "ros/ros.h"

#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/PoseStamped.h"
#include "tf/tf.h"

#include "std_msgs/Float64.h"
#include "std_msgs/Int32.h"

#include "tiles_loc/State.h"
#include "tiles_loc/Observation.h"

#include <image_transport/image_transport.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

#include <ibex.h>
#include <codac.h>
#include <codac-rob.h>
#include <math.h>
#include <stdarg.h>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>

using namespace cv;

#define MIN_GOOD_LINES 5
#define IMG_FOLDER "/home/birromer/ros/data_tiles/dataset_tiles/"

// TODO: test different errors
#define ERROR_PRED      0.2
#define ERROR_OBS       0.3
#define ERROR_OBS_ANGLE 0.1

typedef struct line_struct{
  Point2f p1;     // 1st point of the line
  Point2f p2;     // 2nd point of the line
  double angle;   // angle of the line
  double angle4;  // angle of the line compressed between [-pi/4, pi/4]
  double m_x;     // x coord of the point from the disambiguation function
  double m_y;     // y coord of the point from the disambiguation function
  double d;       // radius value in the polar coordinates of a line
  double dd;      // displacement between tiles
  int side;       // 0 if horizontal, 1 if vertical
} line_t;

// helper functions
double sawtooth(double x);
double modulo(double a, double b);
int sign(double x);
double median(std::vector<line_t> lines, int op);
double median(std::vector<double> scores);
ibex::IntervalVector compute_change_dt(ibex::IntervalVector state, double u1, double u2, double dt);
cv::Mat generate_grid_1(int dist_lines, ibex::IntervalVector obs);
cv::Mat generate_grid_2(int dist_lines, ibex::IntervalVector obs);
void ShowManyImages(string title, int nArgs, ...);

// message convertion functions
tiles_loc::State state_to_msg(ibex::IntervalVector state);
tiles_loc::Observation observation_to_msg(ibex::IntervalVector  y);

// callback functions
void state_loc_callback(const tiles_loc::State::ConstPtr& msg);
void image_callback(const sensor_msgs::ImageConstPtr& msg);
void speed_callback(const geometry_msgs::Pose::ConstPtr& msg);
void compass_callback(const std_msgs::Float64::ConstPtr& msg);
void time_callback(const std_msgs::Float64::ConstPtr& msg);
void pose_callback(const geometry_msgs::Pose& msg); // NOTE: temporary for dataset

// node communication related variables
ibex::IntervalVector state_loc(3, ibex::Interval::ALL_REALS);  // robot state from the localization method
ibex::IntervalVector obs(3, ibex::Interval::ALL_REALS);        // observed parameters from the image
double compass, speed_x, speed_y, speed_z, speed_rho, speed_tht, speed_psi;  // input from the sensors
double sim_time;       // simulation time from coppelia
double prev_sim_time;  // previosu simulation time from coppelia

int img_idx = 0;

std::vector<line_t> base_grid_lines;
bool base_grid_created = false;

int quart_state = 0;  // in which quarter of the plane is the current angle
double prev_a_hat;  // a_hat of the previous iteration

// image processing related variables
bool display_window;
double frame_width=0, frame_height=0;

const int dist_lines = 103.0;  //pixels between each pair of lines

// NOTE: TEMPORARY FOR CREATING DATASET
double pose_1, pose_2, pose_3;
ofstream file_gt;
// ------------------------------

int main(int argc, char **argv) {
  ros::init(argc, argv, "robot_node");
  ros::NodeHandle n;

  ros::Rate loop_rate(50);  // 50Hz frequency

  double dt;

  // NOTE: TEMPORARY FOR CREATING DATASET
  file_gt.open("/home/birromer/ros/data_tiles/gt.csv", fstream::in | fstream::out | fstream::trunc);
  file_gt << "x" << "," << "y" << "," << "theta" << endl;
  // ------------------------------

  // read initial values from the launcher
  ibex::IntervalVector state(3, ibex::Interval::ALL_REALS);          // current state of the robot
  ibex::IntervalVector state_pred_dt(3, ibex::Interval::ALL_REALS);  // change of state within time step
  ibex::IntervalVector y(3, ibex::Interval::ALL_REALS);              // current state of the robot

  double x1, x2, x3;  // initial parameters for the state
  n.param<double>("pos_x_init", x1, 0);
  n.param<double>("pos_y_init", x2, 0);
  n.param<double>("pos_th_init", x3, 0);
  display_window = n.param<bool>("display_window", true);

  // starting state is known
  state[0] = ibex::Interval(x1,x1);
  state[1] = ibex::Interval(x2,x2);
  state[2] = ibex::Interval(x3,x3);

  double u1, u2;      // current input received

  // start visualization windows windows
  if(display_window) {
//    cv::namedWindow("steps");
    cv::namedWindow("camera");
//    cv::namedWindow("grey");
//    cv::namedWindow("sobel");
//    cv::namedWindow("canny");
//    cv::namedWindow("morphology");
    cv::namedWindow("rotated");
    cv::namedWindow("lines");
    cv::namedWindow("view_param_1");
    cv::namedWindow("view_param_2");
    cv::startWindowThread();
  }

  // --- subscribers --- //
  // subscriber to the estimated movement of the robot (speed and heading)
  ros::Subscriber sub_time = n.subscribe("simulationTime", 1000, time_callback);
  ros::Subscriber sub_speed = n.subscribe("speed", 1000, speed_callback);
  ros::Subscriber sub_compass = n.subscribe("compass", 1000, compass_callback);

  // subscriber to the image from the simulator
  ros::Subscriber sub_img = n.subscribe("image", 1000, image_callback);

  // subscriber to the updated state from the localization subsystem
  ros::Subscriber sub_loc = n.subscribe("state_loc", 1000, state_loc_callback);

  // NOTE: Temporary for logging dataset
  ros::Subscriber sub_pose = n.subscribe("pose", 1000, pose_callback);
  // ------------------ //

  // --- publishers --- //
  // publisher of the state for control and viewer
  ros::Publisher pub_state = n.advertise<tiles_loc::State>("state", 1000);

  // publisher of the predicted state and observation for localization
  ros::Publisher pub_state_pred_dt = n.advertise<tiles_loc::State>("state_pred_dt", 1000);
  ros::Publisher pub_y = n.advertise<tiles_loc::Observation>("observation", 1000);
  // ------------------ //

  while (ros::ok()) {
    std::cout << " =================================================================== " << std::endl;
    std::cout << "--------------------- [ROBOT] beggining ros loop ------------------- " << std::endl;
    std::cout << " =================================================================== " << std::endl;

    // publish current, unevolved state to be used by the control and viewer nodes
    tiles_loc::State state_msg = state_to_msg(state);
    pub_state.publish(state_msg);

    dt = sim_time - prev_sim_time;
    prev_sim_time = sim_time;
    ROS_INFO("[ROBOT] Simulaton time step: [%f]", dt);

    state = state_loc;                             // start with the last state contracted from the localization
    y = obs;                                       // use last observed parameters from the image

    u1 = sqrt(speed_x*speed_x + speed_y*speed_y);  // u1 as the speed comes from the velocity in x and y
    u2 = compass;                                  // u2 as the heading comes from the compass

    // evolve state according to input and state equations
    state_pred_dt = compute_change_dt(state, u1, u2, dt);

    // publish evolved state and observation, to be used only by the localization node
    tiles_loc::State state_pred_dt_msg = state_to_msg(state_pred_dt);
    pub_state_pred_dt.publish(state_pred_dt_msg);

    ROS_INFO("[ROBOT] Sent change in dt: x1 ([%f],[%f]) | x2 ([%f],[%f]) | x3 ([%f],[%f])",
             state_pred_dt[0].lb(), state_pred_dt[0].ub(), state_pred_dt[1].lb(), state_pred_dt[1].ub(), state_pred_dt[2].lb(), state_pred_dt[2].ub());

    tiles_loc::Observation observation_msg = observation_to_msg(y);
    pub_y.publish(observation_msg);

    ros::spinOnce();
    loop_rate.sleep();
  }

  file_gt.close();
  return 0;
}

double sawtooth(double x){
  return 2.*atan(tan(x/2.));
}

double median(std::vector<double> scores) {
  //https://stackoverflow.com/questions/2114797/compute-median-of-values-stored-in-vector-c
  size_t size = scores.size();

  if (size == 0) {
    return 0;  // undefined
  } else {
    sort(scores.begin(), scores.end());  // sort elements and take middle one

    if (size % 2 == 0) {
      return (scores[size / 2 - 1] + scores[size / 2]) / 2;
    } else {
      return scores[size / 2];
    }
  }
}

/* use op to select which field to be used for comparisson:
**   1 = angle
**   2 = angle4
**   3 = m_x
**   4 = m_y
**   5 = d
**   6 = dd
*/
double median(std::vector<line_t> lines, int op) {
  size_t size = lines.size();

  if (size == 0) {
    return 0;  // undefined
  } else {
    sort(lines.begin(), lines.end(), [=](line_t l1, line_t l2) -> bool {
      if (op == 1)
        return l1.angle > l2.angle;
      else if (op == 2)
        return l1.angle4 > l2.angle4;
      else if (op == 3)
        return l1.m_x > l2.m_x;
      else if (op == 4)
        return l1.m_y > l2.m_y;
      else if (op == 5)
        return l1.d > l2.d;
      else if (op == 6)
        return l1.dd > l2.dd;
      return false;  // if not an option, leave as is
    });  // sort elements and take middle one

    if (size % 2 == 0) {
      if (op == 1)
        return (lines[size / 2 - 1].angle + lines[size / 2].angle) / 2;
      else if (op == 2)
        return (lines[size / 2 - 1].angle4 + lines[size / 2].angle4) / 2;
      else if (op == 3)
        return (lines[size / 2 - 1].m_x + lines[size / 2].m_x) / 2;
      else if (op == 4)
        return (lines[size / 2 - 1].m_y + lines[size / 2].m_y) / 2;
      else if (op == 5)
        return (lines[size / 2 - 1].d + lines[size / 2].d) / 2;
      else if (op == 6)
        return (lines[size / 2 - 1].dd + lines[size / 2].dd) / 2;

    } else {
      if (op == 1)
        return lines[size/2].angle;
      else if (op == 2)
        return lines[size/2].angle4;
      else if (op == 3)
        return lines[size/2].m_x;
      else if (op == 4)
        return lines[size/2].m_y;
      else if (op == 5)
        return lines[size/2].d;
      else if (op == 6)
        return lines[size/2].dd;
    }

    return 0;
  }
}

double modulo(double a, double b) {
  double r = a/b - floor(a/b);
  if(r<0) {
    r+=1;
  }
  return r*b;
}

int sign(double x) {
  if(x < 0) {
    return -1;
  }
  return 1;
}

ibex::IntervalVector compute_change_dt(ibex::IntervalVector state, double u1, double u2, double dt) {
  ROS_WARN("[ROBOT] STARTING STATE -> x1: ([%f],[%f]) | x2: ([%f],[%f]) | x3: ([%f],[%f])",
             state[0].lb(), state[0].ub(), state[1].lb(), state[1].ub(), state[2].lb(), state[2].ub());

  ibex::IntervalVector change_dt(3, ibex::Interval::ALL_REALS);
  change_dt[0] = (u1 * ibex::cos(u2)).inflate(ERROR_PRED) * dt;
  change_dt[1] = (u1 * ibex::sin(u2)).inflate(ERROR_PRED) * dt;
  change_dt[2] = ibex::Interval(u2).inflate(ERROR_PRED);// * dt;
//  change_dt[2] = ibex::Interval(u2).inflate(ERROR_PRED);// * dt;

  ROS_WARN("[ROBOT] Change on state with dt = [%f] -> dx1: ([%f],[%f]) | dx2: ([%f],[%f]) | dx3: ([%f],[%f])",
             dt, change_dt[0].lb(), change_dt[0].ub(), change_dt[1].lb(), change_dt[1].ub(), change_dt[2].lb(), change_dt[2].ub());

  return change_dt;
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

tiles_loc::Observation observation_to_msg(ibex::IntervalVector y) {
  tiles_loc::Observation msg;
  msg.y1_lb = y[0].lb();
  msg.y1_ub = y[0].ub();
  msg.y2_lb = y[1].lb();
  msg.y2_ub = y[1].ub();
  msg.y3_lb = y[2].lb();
  msg.y3_ub = y[2].ub();

  return msg;
}

// callbacks for each subscriber
void state_loc_callback(const tiles_loc::State::ConstPtr& msg) {
  state_loc[0] = ibex::Interval(msg->x1_lb, msg->x1_ub);
  state_loc[1] = ibex::Interval(msg->x2_lb, msg->x2_ub);
  state_loc[2] = ibex::Interval(msg->x3_lb, msg->x3_ub);
  ROS_INFO("[ROBOT] Received estimated state: x1 ([%f],[%f]) | x2 ([%f],[%f]) | x3 ([%f],[%f])",
           state_loc[0].lb(), state_loc[0].ub(), state_loc[1].lb(), state_loc[1].ub(), state_loc[2].lb(), state_loc[2].ub());
}

string to_filename(int idx, int n_digits) {
  ostringstream os;

  os << setfill('0') << setw(n_digits) << idx;

  return os.str();
}

void image_callback(const sensor_msgs::ImageConstPtr& msg) {
  Mat in;

  try {
    // convert message and flip as needed
    Mat in = cv_bridge::toCvShare(msg, "bgr8")->image;
    flip(in, in, 1);
    frame_height = in.size[0];
    frame_width = in.size[1];
    Mat src = Mat::zeros(Size(frame_width, frame_height), CV_8UC3);

    // NOTE: TEMPORARY FOR CREATING DATASET
    char cimg[1000];
    snprintf(cimg, 1000, "%s%06d.png", IMG_FOLDER, img_idx);
    imwrite(cimg, in);
    img_idx += 1;

    std::cout << "SAVED IMAGE " << cimg << endl;
    file_gt << pose_1 << "," << pose_2 << "," << pose_3 << endl;
    // ------------------------------------

    // convert to greyscale for later computing borders
    Mat grey;
    cvtColor(in, grey, CV_BGR2GRAY);

    // compute the gradient image in x and y with the laplacian for the borders
    Mat grad;
    Laplacian(grey, grad, CV_8U, 1, 1, 0, BORDER_DEFAULT);

    // detect edges, 50 and 255 as thresholds 1 and 2
    Mat edges;
    Canny(grad, edges, 50, 255, 3);

    // close and dilate lines for some noise removal
    Mat morph;
    int morph_elem = 0;
    int morph_size = 0;
    Mat element = getStructuringElement(morph_elem, Size(2*morph_size + 1, 2*morph_size+1), cv::Point(morph_size, morph_size));
    morphologyEx(edges, edges, MORPH_CLOSE, element);
    dilate(edges, morph, Mat(), cv::Point(-1,-1), 1, BORDER_CONSTANT, morphologyDefaultBorderValue());

    // detect lines using the hough transform
    std::vector<Vec4i> detected_lines;
    HoughLinesP(morph, detected_lines, 1, CV_PI/180., 60, 120, 50);

    // from the angles of the lines from the hough transform, as said in luc's paper
    // this is done for ase of computation
    std::vector<double> lines_m_x, lines_m_y, filtered_m_x, filtered_m_y;  // x and y components of the points in M

    double x_hat, y_hat, a_hat;

    // structures for storing the lines information
    std::vector<line_t> lines;  // stores the lines obtained from the hough transform
    std::vector<double> lines_angles;  // stores the angles of the lines from the hough transform with the image compressed between [-pi/4, pi/4]

    double line_angle, line_angle4, d, dd, m_x, m_y;
//    double median_angle;

    // extract the informations from the good detected lines
    for(int i=0; i<detected_lines.size(); i++) {
      Vec4i l = detected_lines[i];
      double p1_x = l[0], p1_y = l[1];
      double p2_x = l[2], p2_y = l[3];

      line_angle = atan2(p2_y - p1_y, p2_x - p1_x);               // get the angle of the line from the existing points
      line_angle4 = modulo(line_angle+M_PI/4., M_PI/2.)-M_PI/4.;  // compress image between [-pi/4, pi/4]

      m_x = cos(4*line_angle);
      m_y = sin(4*line_angle);

      // smallest radius of a circle with a point belonging to the line with origin in 0
      d = ((p2_x-p1_x)*(p1_y) - (p1_x)*(p2_y-p1_y)) / sqrt(pow(p2_x-p1_x,2) + pow(p2_y-p1_y,2));

      // decimal distance, displacement between the lines
      dd = (d/dist_lines - floor(d/dist_lines));

      line_t ln = {
        .p1     = cv::Point(p1_x, p1_y),
        .p2     = cv::Point(p2_x, p2_y),
        .angle  = line_angle,
        .angle4 = line_angle4,
        .m_x    = m_x,
        .m_y    = m_y,
        .d      = d,
        .dd     = dd,
        .side   = 0  // temporary value
      };

      // save the extracted information
      lines.push_back(ln);
    }

   // median of the components of the lines
    x_hat = median(lines, 3);
    y_hat = median(lines, 4);
//    median_angle = median(lines, 2);

    std::vector<line_t> lines_good;

    for (line_t l : lines) {
      if ((abs(x_hat - l.m_x) + abs(y_hat - l.m_y)) < 0.15) {
//      if (sawtooth(l.angle4 - median_angle) < 0.10) {
        filtered_m_x.push_back(l.m_x);
        filtered_m_y.push_back(l.m_y);

        line(src, l.p1, l.p2, Scalar(255, 0, 0), 3, LINE_AA);
        lines_good.push_back(l);

      } else {
        line(src, l.p1, l.p2, Scalar(0, 0, 255), 3, LINE_AA);

      }
    }

    x_hat = median(lines_good, 3);
    y_hat = median(lines_good, 4);

    prev_a_hat = a_hat;
    a_hat = atan2(y_hat, x_hat) * 1/4;
//    median_angle = median(lines_good, 2);

//    if ((a_hat - prev_a_hat) < (-M_PI/2 + 0.1))
//      quart_state += 1;
//    else if ((a_hat - prev_a_hat) > (M_PI/2 - 0.1))
//      quart_state -= 1;
//
//    if (quart_state > 3)
//      quart_state = 0;
//    else if (quart_state < 0)
//      quart_state = 3;
//
//    if (quart_state == 1)
//      a_hat -= M_PI/2;
//    else if (quart_state == 2)
//      a_hat += M_PI;
//    else if (quart_state == 3)
//      a_hat += M_PI/2;

    if(lines_good.size() > MIN_GOOD_LINES) {
      ROS_INFO("[ROBOT] Found [%ld] good lines", lines_good.size());
      Mat rot = Mat::zeros(Size(frame_width , frame_height), CV_8UC3);
      std::vector<line_t> bag_h, bag_v;
      double x1, y1, x2, y2;
      double angle_new;

      for (line_t l : lines_good) {
        //translation in order to center lines around 0
        x1 = l.p1.x - frame_width/2.0f;
        y1 = l.p1.y - frame_height/2.0f;
        x2 = l.p2.x - frame_width/2.0f;
        y2 = l.p2.y - frame_height/2.0f;

        // applies the 2d rotation to the line, making it either horizontal or vertical
//        if (l.angle > M_PI/2 && l.angle < M_PI || l.angle > 3*M_PI/2 && l.angle < 2*M_PI)
//          a_hat -= M_PI/2;

        double x1_temp = x1 * cos(-a_hat) - y1 * sin(-a_hat);
        double y1_temp = x1 * sin(-a_hat) + y1 * cos(-a_hat);

        double x2_temp = x2 * cos(-a_hat) - y2 * sin(-a_hat);
        double y2_temp = x2 * sin(-a_hat) + y2 * cos(-a_hat);

        // translates the image back
        x1 = x1_temp + frame_width/2.0f;
        x2 = x2_temp + frame_width/2.0f;

        y1 = y1_temp + frame_height/2.0f;
        y2 = y2_temp + frame_height/2.0f;

        // compute the new angle of the rotated lines
        angle_new = atan2(y2-y1, x2-x1);

        // determine if the lines are horizontal or vertical
        if (abs(cos(angle_new)) < 0.2) {  // vertical
          line(rot, cv::Point(x1, y1), cv::Point(x2, y2), Scalar(255, 255, 255), 1, LINE_AA);
          line(src, l.p1, l.p2, Scalar(255, 0, 0), 3, LINE_AA);
          bag_v.push_back(l);
//          ROS_WARN("V) D = [%f], D/l = [%f] | DD = [%f]", l.d, l.d/dist_lines, l.dd);

        } else if (abs(sin(angle_new)) < 0.2) {  // horizontal
          line(rot, cv::Point(x1, y1), cv::Point(x2, y2), Scalar(0, 0, 255), 1, LINE_AA);
          bag_h.push_back(l);
          line(src, l.p1, l.p2, Scalar(0, 255, 0), 3, LINE_AA);
//          ROS_WARN("H) D = [%f], D/l = [%f] | DD = [%f]", l.d, l.d/dist_lines, l.dd);
        }
      }

      double d_hat_h = dist_lines * median(bag_h, 6);
      double d_hat_v = dist_lines * median(bag_v, 6);

      ROS_WARN("PARAMETERS -> d_hat_h = [%f] | d_hat_v = [%f] | a_hat = [%f]", d_hat_h, d_hat_v, a_hat);

      obs = ibex::IntervalVector({
          {d_hat_h, d_hat_h},
          {d_hat_v, d_hat_v},
          {a_hat, a_hat}
//          {median_angle, median_angle}
      }).inflate(ERROR_OBS);

      obs[2] = ibex::Interval(a_hat, a_hat).inflate(ERROR_OBS_ANGLE);

      Mat view_param_1 = generate_grid_1(dist_lines, obs);
      Mat view_param_2 = generate_grid_2(dist_lines, obs);

      if(display_window){
//        cvtColor(grey, grey, CV_GRAY2BGR);
//        cvtColor(grad, grad, CV_GRAY2BGR);
//        cvtColor(edges, edges, CV_GRAY2BGR);
//        ShowManyImages("steps", 4, in, grey, grad, edges);//, morph, rot, src);
        cv::imshow("camera", in);
//        cv::imshow("grey", grey);
//        cv::imshow("sobel", grad);
//        cv::imshow("canny", edges);
//        cv::imshow("morphology", morph);
        cv::imshow("lines", src);
        cv::imshow("rotated", rot);
        cv::imshow("view_param_1", view_param_1);
        cv::imshow("view_param_2", view_param_2);
      }

    } else {
      ROS_WARN("Not enough good lines ([%ld])", lines_good.size());

    }

  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    return;
  }
}

cv::Mat generate_grid_1(int dist_lines, ibex::IntervalVector obs) {
  double d_hat_h = obs[0].mid();
  double d_hat_v = obs[1].mid();
  double a_hat   = obs[2].mid();

  int n_lines = 5;
  int max_dim = frame_height > frame_width? frame_height : frame_width;  // largest dimension so that always show something inside the picture

  if (!base_grid_created) {
    // center of the image, where tiles start with zero displacement
    int center_x = frame_width/2.;
    int center_y = frame_height/2.;

    // create a line every specified number of pixels
    // adds one before and one after because occluded areas may appear
//    int pos_x = (center_x % dist_lines) - 2*dist_lines;
    int pos_x = center_x - (n_lines/2)*dist_lines;
    while (pos_x < frame_width + (n_lines/2)*dist_lines) {
      line_t ln = {
        .p1     = cv::Point(pos_x, -max_dim),
        .p2     = cv::Point(pos_x, max_dim),
        .side   = 1  // 0 horizontal, 1 vertical
      };
      base_grid_lines.push_back(ln);
      pos_x += dist_lines;
    }

//    int pos_y = (center_y % dist_lines) - 2*dist_lines;
    int pos_y = center_y - (n_lines/2)*dist_lines;
    while (pos_y < frame_height + (n_lines/2)*dist_lines) {
      line_t ln = {
        .p1     = cv::Point(-max_dim, pos_y),
        .p2     = cv::Point(max_dim, pos_y),
        .side   = 0  // 0 horizontal, 1 vertical
      };
      base_grid_lines.push_back(ln);
      pos_y += dist_lines;
    }

    base_grid_created = true;
  }

  cv::Mat img_grid = cv::Mat::zeros(frame_height, frame_width, CV_8UC3);
  std::vector<line_t> grid_lines = base_grid_lines;

  for (line_t l : grid_lines) {
    //translation in order to center lines around 0
    int x1 = l.p1.x - frame_width/2.  + d_hat_h;
    int y1 = l.p1.y - frame_height/2. + d_hat_v;
    int x2 = l.p2.x - frame_width/2.  + d_hat_h;
    int y2 = l.p2.y - frame_height/2. + d_hat_v;

    // applies the 2d rotation to the line, making it either horizontal or vertical
    int x1_temp = x1;//x1 * cos(a_hat) - y1 * sin(a_hat);
    int y1_temp = y1;//x1 * sin(a_hat) + y1 * cos(a_hat);

    int x2_temp = x2;//x2 * cos(a_hat) - y2 * sin(a_hat);
    int y2_temp = y2;//x2 * sin(a_hat) + y2 * cos(a_hat);

    // translates the image back and adds displacement
    x1 = (x1_temp + frame_width/2. );//+ d_hat_h);
    y1 = (y1_temp + frame_height/2.);// + d_hat_v);
    x2 = (x2_temp + frame_width/2. );//+ d_hat_h);
    y2 = (y2_temp + frame_height/2.);// + d_hat_v);

    if (l.side == 1) {
      line(img_grid, cv::Point(x1, y1), cv::Point(x2, y2), Scalar(255, 0, 0), 3, LINE_AA);
    } else {
      line(img_grid, cv::Point(x1, y1), cv::Point(x2, y2), Scalar(0, 255, 0), 3, LINE_AA);
    }
  }

  return img_grid;
}

cv::Mat generate_grid_2(int dist_lines, ibex::IntervalVector obs) {
  double d_hat_h = obs[0].mid();
  double d_hat_v = obs[1].mid();
  double a_hat   = obs[2].mid();

  int max_dim = frame_height > frame_width? frame_height : frame_width;  // largest dimension so that always show something inside the picture

  if (!base_grid_created) {
    // center of the image, where tiles start with zero displacement
    int center_x = frame_width/2.;
    int center_y = frame_height/2.;

    // create a line every specified number of pixels
    // adds one before and one after because occluded areas may appear
    int pos_x = (center_x % dist_lines) - 2*dist_lines;
    while (pos_x < frame_width + 2*dist_lines) {
      line_t ln = {
        .p1     = cv::Point(pos_x, -max_dim),
        .p2     = cv::Point(pos_x, max_dim),
        .side   = 1  // 0 horizontal, 1 vertical
      };
      base_grid_lines.push_back(ln);
      pos_x += dist_lines;
    }

    int pos_y = (center_y % dist_lines) - 2*dist_lines;
    while (pos_y < frame_height + 2*dist_lines) {
      line_t ln = {
        .p1     = cv::Point(-max_dim, pos_y),
        .p2     = cv::Point(max_dim, pos_y),
        .side   = 0  // 0 horizontal, 1 vertical
      };
      base_grid_lines.push_back(ln);
      pos_y += dist_lines;
    }

    base_grid_created = true;
  }

  cv::Mat img_grid = cv::Mat::zeros(frame_height, frame_width, CV_8UC3);
  std::vector<line_t> grid_lines = base_grid_lines;

  for (line_t l : grid_lines) {
    //translation in order to center lines around 0
    int x1 = l.p1.x - frame_width/2.  + d_hat_v;
    int y1 = l.p1.y - frame_height/2. + d_hat_h;
    int x2 = l.p2.x - frame_width/2.  + d_hat_v;
    int y2 = l.p2.y - frame_height/2. + d_hat_h;

    // applies the 2d rotation to the line, making it either horizontal or vertical
    int x1_temp = x1;//x1 * cos(a_hat) - y1 * sin(a_hat);//+M_PI);
    int y1_temp = y1;//x1 * sin(a_hat) + y1 * cos(a_hat);//+M_PI);

    int x2_temp = x2;//x2 * cos(a_hat) - y2 * sin(a_hat);//+M_PI);
    int y2_temp = y2;//x2 * sin(a_hat) + y2 * cos(a_hat);//+M_PI);

    // translates the image back and adds displacement
    x1 = (x1_temp + frame_width/2. );//+ d_hat_h);
    y1 = (y1_temp + frame_height/2.);// + d_hat_v);
    x2 = (x2_temp + frame_width/2. );//+ d_hat_h);
    y2 = (y2_temp + frame_height/2.);// + d_hat_v);

    if (l.side == 1) {
      line(img_grid, cv::Point(x1, y1), cv::Point(x2, y2), Scalar(255, 0, 0), 3, LINE_AA);
    } else {
      line(img_grid, cv::Point(x1, y1), cv::Point(x2, y2), Scalar(0, 255, 0), 3, LINE_AA);
    }
  }

  return img_grid;
}

void ShowManyImages(string title, int nArgs, ...) {
// from https://github.com/opencv/opencv/wiki/DisplayManyImages

  int size;
  int i;
  int m, n;
  int x, y;

  // w - Maximum number of images in a row
  // h - Maximum number of images in a column
  int w, h;

  // scale - How much we have to resize the image
  float scale;
  int max;

  // If the number of arguments is lesser than 0 or greater than 12
  // return without displaying
  if(nArgs <= 0) {
      printf("Number of arguments too small....\n");
      return;
  }
  else if(nArgs > 14) {
      printf("Number of arguments too large, can only handle maximally 12 images at a time ...\n");
      return;
  }
  // Determine the size of the image,
  // and the number of rows/cols
  // from number of arguments
  else if (nArgs == 1) {
      w = h = 1;
      size = 300;
  }
  else if (nArgs == 2) {
    w = 2; h = 1;
      size = 300;
  }
  else if (nArgs == 3 || nArgs == 4) {
      w = 2; h = 2;
      size = 300;
  }
  else if (nArgs == 5 || nArgs == 6) {
      w = 3; h = 2;
      size = 200;
  }
  else if (nArgs == 7 || nArgs == 8) {
      w = 4; h = 2;
      size = 200;
  }
  else {
      w = 4; h = 3;
      size = 150;
  }

  // Create a new 3 channel image
  Mat DispImage = Mat::zeros(Size(100 + size*w, 60 + size*h), CV_8UC3);

  // Used to get the arguments passed
  va_list args;
  va_start(args, nArgs);

  // Loop for nArgs number of arguments
  for (i = 0, m = 20, n = 20; i < nArgs; i++, m += (20 + size)) {
      // Get the Pointer to the IplImage
      Mat img = va_arg(args, Mat);

      // Check whether it is NULL or not
      // If it is NULL, release the image, and return
      if(img.empty()) {
          printf("Invalid arguments");
          return;
      }

      // Find the width and height of the image
      x = img.cols;
      y = img.rows;

      // Find whether height or width is greater in order to resize the image
      max = (x > y)? x: y;

      // Find the scaling factor to resize the image
      scale = (float) ( (float) max / size );

      // Used to Align the images
      if( i % w == 0 && m!= 20) {
          m = 20;
          n+= 20 + size;
      }

      // Set the image ROI to display the current image
      // Resize the input image and copy the it to the Single Big Image
      Rect ROI(m, n, (int)( x/scale ), (int)( y/scale ));
      Mat temp; resize(img, temp, Size(ROI.width, ROI.height));
      temp.copyTo(DispImage(ROI));
  }

  // Create a new window, and show the Single Big Image
//  namedWindow( title, 1 );
  imshow(title, DispImage);
//  waitKey();

  // End the number of arguments
  va_end(args);
}

void compass_callback(const std_msgs::Float64::ConstPtr& msg) {
  compass = msg->data;
  ROS_INFO("[ROBOT] received compass data: [%f]", compass);
}

void time_callback(const std_msgs::Float64::ConstPtr& msg) {
  sim_time = msg->data;
  ROS_INFO("[ROBOT] received simulation time: [%f]", sim_time);
}

void speed_callback(const geometry_msgs::Pose::ConstPtr& msg) {
  speed_x = msg->position.x;
  speed_y = msg->position.y;
  speed_z = msg->position.z;

  speed_rho = msg->orientation.x;
  speed_tht = msg->orientation.y;
  speed_psi = msg->orientation.z;

  ROS_INFO("[ROBOT] Received current speed in x, y and z: [%f] [%f] [%f] | rho, theta and psi: [%f] [%f] [%f]", speed_x, speed_y, speed_z, speed_rho, speed_tht, speed_psi);
}

void pose_callback(const geometry_msgs::Pose& msg){
  pose_1 = msg.position.x;
  pose_2 = msg.position.y;
  pose_3 = tf::getYaw(msg.orientation);
}
