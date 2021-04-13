#include "ros/ros.h"

#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Quaternion.h"
#include "geometry_msgs/PoseStamped.h"

#include "tf/tf.h"
#include <tf2/LinearMath/Quaternion.h>


#include "std_msgs/Float32.h"
#include "std_msgs/Float32MultiArray.h"

#include "tiles_loc/Observation.h"
#include "tiles_loc/Cmd.h"

#include <image_transport/image_transport.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

#include <math.h>

using namespace cv;

struct line_struct{
  Point2f p1;
  Point2f p2;
  float angle;
};


// helper functions
double sawtooth(double x);
float modulo(float a, float b);
int sign(float x);
double median(std::vector<double> scores);
void integration_euler(double &x1, double &x2, double &x3, double u1, double u2, double dt);
geometry_msgs::PoseStamped state_to_pose_stamped(double x1, double x2, double x3);
tiles_loc::Observation y_to_observation(double y1, double y2, double y3);


// callback functions
void cmd_callback(const tiles_loc::Cmd::ConstPtr& msg);
void image_callback(const sensor_msgs::ImageConstPtr& msg);
void state_loc_callback(const geometry_msgs::PoseStamped::ConstPtr& msg);


// node communication related variables
float xloc_1, xloc_2, xloc_3;  // robot state from the localization method
float obs_1, obs_2, obs_3;     // observed parameters from the image
float cmd_1, cmd_2;            // commands from controller

// image processing related variables
bool display_window;
double frame_width, frame_height;

const float pix = 103;//99.3;//107.; //pixels entre chaque lignes
//taille du carrelage en mètre
float size_carrelage_x = 2.025/12.;
float size_carrelage_y = 2.025/12.;
float percent = 0.9; //de combien max on estime qu'on aura bougé (là, de moins d'un carreau donc le calcul est possible)
float alpha_median;
int quart;
float last_alpha_median = 0;
int nn = 0;
//variable sobel
int scale = 1;
int delta = 0;
int ddepth = CV_16S;
//traitement d'image pour détecter les lignes
Mat src, src_, src_gray;
Mat grad;


int main(int argc, char **argv) {
  ros::init(argc, argv, "robot_node");
  ros::NodeHandle n;

  ros::Rate loop_rate(25);  // 25Hz frequency
  double dt = 0.005;  // time step

  double x1, x2, x3;  // current state of the robot
  double y1, y2, y3;  // current observation of the robot
  double u1, u2;      // current input received

  // read initial values from the launcher
  n.param<double>("pos_x_init", x1, 0);
  n.param<double>("pos_y_init", x2, 0);
  n.param<double>("pos_th_init", x3, 0);
  display_window =  n.param<bool>("display_window", true);

  // start visualization windows windows
  if(display_window) {
    cv::namedWindow("view");
    cv::namedWindow("view base");
    cv::namedWindow("grey");
    cv::namedWindow("sobel");
    cv::namedWindow("canny");
    cv::namedWindow("rotated");
    cv::startWindowThread();
  }

  // --- subscribers --- //
  // subscriber to z inputs from control
  ros::Subscriber sub_cmd = n.subscribe("cmd", 1000, cmd_callback);

  // subscriber to the image from the simulator
  ros::Subscriber sub_img = n.subscribe("image", 1000, image_callback);

  // subscriber to the updated state from the localization subsystem
  ros::Subscriber sub_loc = n.subscribe("state_loc", 1000, state_loc_callback);
  // ------------------ //

  // --- publishers --- //
  // publisher of the state for control and viewer
  ros::Publisher pub_state = n.advertise<geometry_msgs::PoseStamped::ConstPtr>("state", 1000);

  // publisher of the predicted state and observation for localization
  ros::Publisher pub_state_pred = n.advertise<geometry_msgs::PoseStamped::ConstPtr>("state_pred", 1000);
  ros::Publisher pub_y = n.advertise<tiles_loc::Observation>("observation", 1000);
  // ------------------ //

  while (ros::ok()) {
    x1 = xloc_1, x2 = xloc_2, x3 = xloc_3;  // start with the last state contracted from the localization
    y1 = obs_1, y2 = obs_2, y3 = obs_3;      // use last observed parameters from the image
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

float modulo(float a, float b) {
  float r = a/b - floor(a/b);
  if(r<0) {
    r+=1;
  }
  return r*b;
}

int sign(float x) {
  if(x < 0) {
    return -1;
  }
  return 1;
}

void integration_euler(double &x1, double &x2, double &x3, double u1, double u2, double dt) {
  x1 = x1 + dt * (u1*cos(x3));
  x2 = x2 + dt * (u1*sin(x3));
  x3 = x3 + dt * (u2);
  ROS_INFO("Updated state -> x1: [%f] | x2: [%f] | x3: [%f] || u1: [%f] | u2: [%f]", x1, x2, x3, u1, u2);
}

// callbacks for each subscriber
void cmd_callback(const tiles_loc::Cmd::ConstPtr& msg) {
  cmd_1 = msg->u1;
  cmd_2 = msg->u2;
  ROS_INFO("Received command: u1 [%f] u2 [%f]", cmd_1, cmd_2);
}

geometry_msgs::PoseStamped state_to_pose_stamped(double x1, double x2, double x3) {
    geometry_msgs::PoseStamped msg;
    msg.pose.position.x = x1;
    msg.pose.position.y = x2;
    msg.pose.position.z = 0;
    tf::Quaternion q;
    q.setRPY(0, 0, x3);  // roll, pitch, yaw
    tf::quaternionTFToMsg(q, msg.pose.orientation);

    return msg;
}

tiles_loc::Observation y_to_observation(double y1, double y2, double y3) {
  tiles_loc::Observation msg;
  msg.y1 = y1;
  msg.y2 = y2;
  msg.y3 = y3;

  return msg;
}

void image_callback(const sensor_msgs::ImageConstPtr& msg) {
  double obs_1, obs_2, obs_3;

  Mat in;

  try {
    // convert message and flip as needed
    Mat in = flip(cv_bridge::toCvShare(msg, "bgr8")->image, in, 1);
    frame_height = in.size[0];
    frame_width = in.size[1];
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    return;
  }

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
  int morph_elem = 0;
  int morph_size = 0;
  Mat element = getStructuringElement(morph_elem, Size(2*morph_size + 1, 2*morph_size+1), cv::Point(morph_size, morph_size));
  morphologyEx(edges, edges, MORPH_CLOSE, element);
  dilate(edges, edges, Mat(), cv::Point(-1,-1), 1, BORDER_CONSTANT, morphologyDefaultBorderValue());

  // detect lines using the hough transform
  std::vector<Vec4i> lines;
  HoughLinesP(edges, lines, 1, CV_PI/180., 60, 120, 50);

  // structures for storing the lines information
  std::vector<line_struct> lines_points;
  std::vector<double> lines_angles;
  double x1, x2, y1, y2;

  // extract the informations from the good detected lines
  for(int i=0; i<lines.size(); i++) {
    Vec4i l = lines[i];
    x1 = l[0];
    y1 = l[1];
    x2 = l[2];
    y2 = l[3];

    Mat src = Mat::zeros(Size(frame_width, frame_height), CV_8UC3);

    double angle_line = atan2(y2-y1, x2-x1);
    line(src, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), Scalar(255, 0, 0), 3, LINE_AA);

    line_struct ln = {
      .p1 = cv::Point(x1, y1),
      .p2 = cv::Point(x2, y2),
      .angle = angle_line
    };
    lines_points.push_back(ln);

    float angle4 = modulo(angle_line+M_PI/4., M_PI/2.)-M_PI/4.;
    lines_angles.push_back(angle4);
  }

  float median_angle = median(lines_angles);  // get the angle from the tiles being seen
  alpha_median = median_angle;
  std::vector<Vec4i> lines_good;

  // filter lines with coherent orientation
  for (int i=0; i<lines_points.size(); i++) {
    if(sawtooth(lines_angles[i]-median_angle) < 0.1) {
      line(src, lines_points[i].p1, lines_points[i].p2, Scalar(255, 0, 0), 3, LINE_AA);
      lines_good.push_back(lines[i]);

    } else {
       std::cout << "line with bad angle" << lines_angles[i] << " | " << sawtooth(lines_angles[i]-median_angle) << endl;
       line(src, lines_points[i].p1, lines_points[i].p2, Scalar(0, 255, 0), 3, LINE_AA);
    }
  }

  if(lines_good.size() > 5) {
    std::cout << "found " << lines_good.size() << " good lines" << std::endl;

    //conversion from cartesian to polar form :
    float x1, x2, y1, y2;
    vector<double> Msn, Mew;

    alpha_median = alpha_median;
    std::cout << "alpha_median : " << alpha_median*180/M_PI << std::endl;

    //à la limite, on a le quart qui fluctue donc on veut éviter ça
    if(nn == 0) {
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

    std::cout << "________quart : " << quart << " n : " << nn << " " << abs(abs(alpha_median)-M_PI/4.) << std::endl;

    Mat rot = Mat::zeros(Size(frame_width, frame_height), CV_8UC3);

    for(int i=0; i<lines.size(); i++) {
      cv::Vec4i l = lines[i];
      x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
      //std::cout << x1 << " | " << y1<< " | " << x2 << " | " << y2 << std::endl;
      //std::cout << frame_height << " | " << frame_width << std::endl;

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

    std::cout << "alpha_median : " << alpha_median*180/M_PI << " " << quart%2<< std::endl;

    float medx = sign(cos(state[2].mid()))*median(Mew);
    float medy = sign(sin(state[2].mid()))*median(Msn);


  } else {
    std::cout << "Pas assez de lignes (" << lines_good.size() << ")" << std::endl;
  }


  Mat src = Mat::zeros(Size(frame_width, frame_height), CV_8UC3);

  if(display_window){
    cv::imshow("View base", in);
    cv::imshow("grey",grey);
    cv::imshow("Sobel",grad);
    cv::imshow("Canny",edges);
    cv::imshow("view",src);
  }
}
