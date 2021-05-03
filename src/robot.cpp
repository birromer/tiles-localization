/*
** This node is responsible for centering all other node's operations.
** It processes the image data in order to generate the observation vector and
** evolves the robot position according to the state equations
**
** Subscribers:
**   - tiles_loc::State state_loc        // the estimation of the robot's state
**   - std_msgs::Float64 cmd_l           // the input u1 for the robot
**   - std_msgs::Float64 cmd_r           // the input u2 for the robot
**   - sensor_msgs::ImageConstPtr image  // the image from the robot's camera

**
** Publishers:
**   - tiles_loc::State state_pred    // the evolved state, predicted from the state equations
**   - tiles_loc::State state         // the current state of the robot
**   - tiles_loc::Observation observation  // the observation vector, processed from the incoming image
*/

#include "ros/ros.h"

#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Quaternion.h"
#include "geometry_msgs/PoseStamped.h"

#include "tf/tf.h"
#include <tf2/LinearMath/Quaternion.h>

#include "std_msgs/Float64.h"

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

using namespace cv;

#define MIN_GOOD_LINES 5

struct line_struct{
  Point2f p1;
  Point2f p2;
  double angle;
};


// helper functions
double sawtooth(double x);
double modulo(double a, double b);
int sign(double x);
double median(std::vector<double> scores);
ibex::IntervalVector integration_euler(ibex::IntervalVector state, double u1, double u2, double dt);
void ShowManyImages(string title, int nArgs, ...);

// message convertion functions
tiles_loc::State state_to_msg(double x1, double x2, double x3);
tiles_loc::State state_to_msg(ibex::IntervalVector state);
tiles_loc::Observation observation_to_msg(double y1, double y2, double y3);

// callback functions
void state_loc_callback(const tiles_loc::State::ConstPtr& msg);
void cmd_l_callback(const std_msgs::Float64::ConstPtr& msg);
void cmd_r_callback(const std_msgs::Float64::ConstPtr& msg);
void image_callback(const sensor_msgs::ImageConstPtr& msg);

void pose_callback(const geometry_msgs::Pose::ConstPtr& msg);

// node communication related variables
ibex::IntervalVector state(3, ibex::Interval::ALL_REALS);  // current state of the robot
ibex::IntervalVector state_loc(3, ibex::Interval::ALL_REALS);  // robot state from the localization method
double obs_1, obs_2, obs_3;      // observed parameters from the image
double cmd_1, cmd_2;             // commands from controller

double pose_1, pose_2, pose_3;      // true pose

// image processing related variables
bool display_window;
double frame_width, frame_height;

const double pix = 103;//99.3;//107.; //pixels entre chaque lignes
const double scale_pixel = 1./pix;
//taille du carrelage en mètre
double size_carrelage_x = 2.025/12.;
double size_carrelage_y = 2.025/12.;
double percent = 0.9; //de combien max on estime qu'on aura bougé (là, de moins d'un carreau donc le calcul est possible)
double alpha_median;
int quart;
double last_alpha_median = 0;
int nn = 0;
//variable sobel
int scale = 1;
int delta = 0;
int ddepth = CV_16S;
//traitement d'image pour détecter les lignes
Mat grad;


int main(int argc, char **argv) {
  ros::init(argc, argv, "robot_node");
  ros::NodeHandle n;

  ros::Rate loop_rate(25);  // 25Hz frequency
  double dt = 0.005;  // time step

  // read initial values from the launcher
  double x1, x2, x3;  // temporary values to get the initial parameters
  n.param<double>("pos_x_init", x1, 0);
  n.param<double>("pos_y_init", x2, 0);
  n.param<double>("pos_th_init", x3, 0);

//  ibex::IntervalVector state({{x1,x1}, {x2,x2}, {x3,x3}});  // current state of the robot
  state[0] = ibex::Interval(x1,x1);
  state[1] = ibex::Interval(x2,x2);
  state[2] = ibex::Interval(x3,x3);
  double y1, y2, y3;                                        // current observation of the robot
  double u1, u2;                                            // current input received

  display_window = n.param<bool>("display_window", true);

  // start visualization windows windows
  if(display_window) {
//    cv::namedWindow("steps");
    cv::namedWindow("lines");
    cv::namedWindow("camera");
    cv::namedWindow("grey");
    cv::namedWindow("sobel");
    cv::namedWindow("canny");
    cv::namedWindow("morphology");
    cv::namedWindow("rotated");
    cv::startWindowThread();
  }

  // --- subscribers --- //
  // subscriber to z inputs from control
  ros::Subscriber sub_cmd_l = n.subscribe("cmd_l", 1000, cmd_l_callback);
  ros::Subscriber sub_cmd_r = n.subscribe("cmd_r", 1000, cmd_r_callback);

  // subscriber to the image from the simulator
  ros::Subscriber sub_img = n.subscribe("image", 1000, image_callback);

  // subscriber to the updated state from the localization subsystem
  ros::Subscriber sub_loc = n.subscribe("state_loc", 1000, state_loc_callback);

  ros::Subscriber sub_pose= n.subscribe("pose", 1000, pose_callback);
  // ------------------ //

  // --- publishers --- //
  // publisher of the state for control and viewer
  ros::Publisher pub_state      = n.advertise<tiles_loc::State>("state", 1000);

  // publisher of the predicted state and observation for localization
  ros::Publisher pub_state_pred = n.advertise<tiles_loc::State>("state_pred", 1000);
  ros::Publisher pub_y = n.advertise<tiles_loc::Observation>("observation", 1000);
  // ------------------ //

  while (ros::ok()) {
    std::cout << " =================================================================== " << std::endl;
    std::cout << "--------------------- [ROBOT] beggining ros loop ------------------- " << std::endl;
    std::cout << " =================================================================== " << std::endl;

    state = state_loc;                   // start with the last state contracted from the localization
    y1 = obs_1, y2 = obs_2, y3 = obs_3;  // use last observed parameters from the image
    u1 = cmd_1, u2 = cmd_2;              // use last input from command

    // publish current, unevolved state to be used by the control and viewer nodes
    tiles_loc::State state_msg = state_to_msg(state);
    pub_state.publish(state_msg);

    // evolve state according to input and state equations
    state = integration_euler(state, u1, u2, dt);

    // publish evolved state and observation, to be used only by the localization node
    tiles_loc::State state_pred_msg = state_to_msg(state);
    pub_state_pred.publish(state_pred_msg);

    tiles_loc::Observation observation_msg = observation_to_msg(y1, y2, y3);
    pub_y.publish(observation_msg);

    ROS_WARN("Sent parameters: y1 [%f] | y2 [%f] | y3 [%f]", y1, y2, y3);
    ROS_WARN("Using truth: p1 [%f] | p2 [%f] | p3 [%f]", pose_1, pose_2, pose_3);

    ROS_INFO("Equivalence equations 1:\nsin(pi*(y1-z1)) = [%f]\nsin(pi*(y2-z2)) = [%f]\nsin(y2-z2) = [%f]\n", sin(M_PI*(y1-pose_1)), sin(M_PI*(y2-pose_2)), sin(y3-pose_3));
    ROS_INFO("Equivalence equations 2:\nsin(pi*(y1-z2)) = [%f]\nsin(pi*(y2-z1)) = [%f]\ncos(y2-z1) = [%f]\n", sin(M_PI*(y1-pose_2)), sin(M_PI*(y2-pose_1)), cos(y3-pose_3));

    ros::spinOnce();
    loop_rate.sleep();
  }

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

ibex::IntervalVector integration_euler(ibex::IntervalVector state, double u1, double u2, double dt) {
  ibex::IntervalVector state_new(3, ibex::Interval::ALL_REALS);
  state_new[0] = state[0] + dt * (u1*ibex::cos(state[0]));
  state_new[1] = state[1] + dt * (u1*ibex::sin(state[1]));
  state_new[2] = state[2] + dt * (u2);

  //ROS_INFO("[ROBOT] Updated state -> x1: ([%f],[%f]) | x2: ([%f],[%f]) | x3: ([%f],[%f]) || u1: [%f] | u2: [%f]",
//           state_new[0].lb(), state_new[0].ub(), state_new[1].lb(), state_new[1].ub(), state_new[2].lb(), state_new[2].ub(), u1, u2);

  return state_new;
}

void state_loc_callback(const tiles_loc::State::ConstPtr& msg) {
  state_loc[0] = ibex::Interval(msg->x1_lb, msg->x1_ub);
  state_loc[1] = ibex::Interval(msg->x2_lb, msg->x2_ub);
  state_loc[2] = ibex::Interval(msg->x3_lb, msg->x3_ub);
//  ROS_INFO("[ROBOT] Received estimated state: x1 ([%f],[%f]) | x2 ([%f],[%f]) | x3 ([%f],[%f])",
//           state_loc[0].lb(), state_loc[0].ub(), state_loc[1].lb(), state_loc[1].ub(), state_loc[2].lb(), state_loc[2].ub());
}

// callbacks for each subscriber
void cmd_l_callback(const std_msgs::Float64::ConstPtr& msg) {
  cmd_1 = msg->data;
//  ROS_INFO("[ROBOT] Received command: u1 [%f] ", cmd_1);
}

void cmd_r_callback(const std_msgs::Float64::ConstPtr& msg) {
  cmd_2 = msg->data;
//  ROS_INFO("[ROBOT] Received command: u2 [%f]", cmd_2);
}

tiles_loc::State state_to_msg(double x1, double x2, double x3) {
    tiles_loc::State msg;
    msg.x1_lb = x1;
    msg.x1_ub = x1;
    msg.x2_lb = x2;
    msg.x2_ub = x2;
    msg.x3_lb = x3;
    msg.x3_ub = x3;

    return msg;
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

tiles_loc::Observation observation_to_msg(double y1, double y2, double y3) {
  tiles_loc::Observation msg;
  msg.y1 = y1;
  msg.y2 = y2;
  msg.y3 = y3;

  return msg;
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
    std::vector<Vec4i> lines;
    HoughLinesP(morph, lines, 1, CV_PI/180., 60, 120, 50);

    // structures for storing the lines information
    std::vector<line_struct> lines_points;
    std::vector<double> lines_angles;
    double x1, x2, y1, y2;  // line goes from (x1,y1) to (x2,y2)

    // extract the informations from the good detected lines
    for(int i=0; i<lines.size(); i++) {
      // save the line info for quick access
      Vec4i l = lines[i];
      x1 = l[0];
      y1 = l[1];
      x2 = l[2];
      y2 = l[3];

      double angle_line = atan2(y2-y1, x2-x1);  // get the angle of the line from the existing points
      line(src, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), Scalar(255, 0, 0), 3, LINE_AA);

      line_struct ln = {
        .p1 = cv::Point(x1, y1),
        .p2 = cv::Point(x2, y2),
        .angle = angle_line
      };
      lines_points.push_back(ln);  // save the extracted information

      double angle4 = modulo(angle_line+M_PI/4., M_PI/2.)-M_PI/4.;  // compress image between [-pi/4, pi/4]
      lines_angles.push_back(angle4);
    }

    // get the median angle from the tiles being seen,
    // will be used to filder bad lines and get the orientation of the tiles
    double median_angle = median(lines_angles);
    alpha_median = median_angle;

    std::vector<Vec4i> lines_good;

    // filter lines with coherent orientation with the median into lines_good
    for (int i=0; i<lines_points.size(); i++) {
      if(sawtooth(lines_angles[i]-median_angle) < 0.1) {
        line(src, lines_points[i].p1, lines_points[i].p2, Scalar(255, 0, 0), 3, LINE_AA);
        lines_good.push_back(lines[i]);

      } else {
//         std::cout << "line with bad angle" << lines_angles[i] << " | " << sawtooth(lines_angles[i]-median_angle) << std::endl;
         line(src, lines_points[i].p1, lines_points[i].p2, Scalar(0, 255, 0), 3, LINE_AA);
      }
    }

    if(lines_good.size() > MIN_GOOD_LINES) {
//      ROS_INFO("[ROBOT] Found [%ld] good lines", lines_good.size());

      //conversion from cartesian to polar form :
      double x1, x2, y1, y2;
      std::vector<double> median_h, median_v;  // median for horizontal and vertical lines

//      std::cout << "alpha_median : " << alpha_median*180/M_PI << std::endl;

      // counts how many times 90 degrees it has turned to retorate that afterwards
      // think about the mabiguity of the 90° rotations on the tiles
      //à la limite, on a le quart qui fluctue donc on veut éviter ça  -> wat
      if(nn == 0) {
        // detects if angle between 45 and -45 (first abs takes both sides of zero
        // and second lets only compare to < 0.1)
        if(abs(abs(alpha_median)-M_PI/4.) < 0.1) {
          if(alpha_median >= 0 & last_alpha_median < 0) {
            quart -= 1;
            nn = 1;
          } else if (alpha_median <= 0 & last_alpha_median > 0) {
            quart += 1;
            nn = 1;
          }
        }
      } else {
        nn += 1;
      }

      if(nn == 100) {  // arbitraty number to verify 2 pi problem
        nn = 0;
      }

      last_alpha_median = alpha_median;

      std::cout << "________quart : " << quart << " n : " << nn << " " << abs(abs(alpha_median)-M_PI/4.) << std::endl;

      Mat rot = Mat::zeros(Size(frame_width, frame_height), CV_8UC3);

      for(int i=0; i<lines_good.size(); i++) {
        cv::Vec4i l = lines_good[i];
        x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
        //std::cout << x1 << " | " << y1<< " | " << x2 << " | " << y2 << std::endl;
        //std::cout << frame_height << " | " << frame_width << std::endl;

        //translation in order to center lines around 0
        x1 -= frame_width/2.0f;
        y1 -= frame_height/2.0f;
        x2 -= frame_width/2.0f;
        y2 -= frame_height/2.0f;

        // rotation around 0, in order to have only horizontal and vetical lines
        double alpha = atan2(y2-y1, x2-x1);

        double angle = modulo(alpha+M_PI/4., M_PI/2)-M_PI/4.;  // compress image between [-pi/4, pi/4]
        angle += quart*M_PI/2.;  // undoes the 90° extra rotations that were performed by the movement of the robot

        double s = sin(-angle);
        double c = cos(-angle);

        double x1b = x1;
        double y1b = y1;
        double x2b = x2;
        double y2b = y2;

        // applies the 2d rotation to the line, making it either horizontal or vertical
        x1 = x1b*c - y1b*s,
        y1 = +x1b*s + y1b*c;

        x2 = x2b*c - y2b*s,
        y2 = x2b*s + y2b*c;

        // translates the image back for the display
        x1 += frame_width/2.0f;
        y1 += frame_height/2.0f;
        x2 += frame_width/2.0f;
        y2 += frame_height/2.0f;

        double alpha2 = atan2(y2-y1,x2-x1);
        double x11 = x1;
        double y11 = y1;
        double x22 = x2;
        double y22 = y2;

        //calcul pour medx et medy
        x1 = ((double)l[0]-frame_width/2.0f)*scale_pixel;
        y1 = ((double)l[1]-frame_height/2.0f)*scale_pixel;

        x2 = ((double)l[2]-frame_width/2.0f)*scale_pixel;
        y2 = ((double)l[3]-frame_height/2.0f)*scale_pixel;

        double d = ((x2-x1)*(y1)-(x1)*(y2-y1)) / sqrt(pow(x2-x1, 2)+pow(y2-y1, 2));

        double val;
        val = (d+0.5)-floor(d+0.5)-0.5;

        if (abs(cos(alpha2)) < 0.2) {
          line(rot, cv::Point(x11, y11), cv::Point(x22, y22), Scalar(255, 255, 255), 1, LINE_AA);
          median_v.push_back(val);

        } else if (abs(sin(alpha2)) < 0.2) {
          line(rot, cv::Point(x11, y11), cv::Point(x22, y22), Scalar(0, 0, 255), 1, LINE_AA);
          median_h.push_back(val);

        }
      }

      alpha_median = alpha_median + quart*M_PI/2;
      alpha_median = modulo(alpha_median, 2*M_PI);

//      std::cout << "alpha_median : " << alpha_median*180/M_PI << " " << quart%2<< std::endl;

      double medx = sign(cos(state[2].mid()))*median(median_h);
      double medy = sign(sin(state[2].mid()))*median(median_v);

      obs_1 = medx;
      obs_2 = medy;
      obs_3 = alpha_median;

      if(display_window){
//        cvtColor(grey, grey, CV_GRAY2BGR);
//        cvtColor(grad, grad, CV_GRAY2BGR);
//        cvtColor(edges, edges, CV_GRAY2BGR);
//        ShowManyImages("steps", 4, in, grey, grad, edges);//, morph, rot, src);
        cv::imshow("camera", in);
        cv::imshow("grey", grey);
        cv::imshow("sobel", grad);
        cv::imshow("canny", edges);
        cv::imshow("morphology", morph);
        cv::imshow("lines", src);
        cv::imshow("rotated", rot);

      }
    } else {
      ROS_WARN("Not enought good lines ([%ld])", lines_good.size());

    }

  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    return;
  }
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
  std::cout << "REACHED HEEEEEEEEEEEEEEEEEEEEERE 1" << std::endl;
      Rect ROI(m, n, (int)( x/scale ), (int)( y/scale ));
  std::cout << "REACHED HEEEEEEEEEEEEEEEEEEEEERE 2" << std::endl;
      Mat temp; resize(img, temp, Size(ROI.width, ROI.height));
  std::cout << "REACHED HEEEEEEEEEEEEEEEEEEEEERE 3" << std::endl;
      temp.copyTo(DispImage(ROI));
  std::cout << "REACHED HEEEEEEEEEEEEEEEEEEEEERE 4" << std::endl;
  }

  // Create a new window, and show the Single Big Image
//  namedWindow( title, 1 );
  std::cout << "REACHED HEEEEEEEEEEEEEEEEEEEEERE 5" << std::endl;
  imshow(title, DispImage);
  std::cout << "REACHED HEEEEEEEEEEEEEEEEEEEEERE 6" << std::endl;
//  waitKey();

  // End the number of arguments
  va_end(args);
}

void pose_callback(const geometry_msgs::Pose::ConstPtr& msg) {
    geometry_msgs::Pose pose;
    pose_1 = msg->position.x;
    pose_2 = msg->position.y;
    pose_3 = tf::getYaw(msg->orientation);;

}
