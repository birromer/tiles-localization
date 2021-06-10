#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <ibex.h>
#include <codac.h>
#include <codac-rob.h>
#include <math.h>
#include <stdarg.h>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

#define MIN_GOOD_LINES 5

#define ERROR_PRED      0.1
#define ERROR_OBS       0.3
#define ERROR_OBS_ANGLE 0.1

#define NUM_IMGS 10
#define IMG_FOLDER "/home/birromer/ros/dataset_tiles/"
#define GT_FILE "/home/birromer/ros/gt.csv"
#define SIM_FILE "/home/birromer/ros/test_sim.csv"

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

double sawtooth(double x);
double modulo(double a, double b);
int sign(double x);
double median(std::vector<line_t> lines, int op);
double median(std::vector<double> scores);

cv::Mat gen_grid(int dist_lines, ibex::IntervalVector obs);
ibex::IntervalVector get_obs(cv::Mat image);

std::vector<line_t> base_grid_lines;
bool base_grid_created = false;

bool display_window;
double frame_width=0, frame_height=0;

const int dist_lines = 103.0;  //pixels between each pair of lines

int main(int argc, char **argv) {
  int curr_img = 0;
  bool verbose = false;

  double pose_1, pose_2, pose_3;
  ibex::IntervalVector obs(3, ibex::Interval::ALL_REALS);  // observed parameters from the image
//  ibex::IntervalVector pose(3, ibex::Interval::ALL_REALS);  // observed parameters from the image

  ifstream file_gt(GT_FILE);
  ofstream file_sim(SIM_FILE);
  file_sim << "sim1_eq1" << "," << "sim1_eq2" << "," << "sim1_eq3" << "," << "sim2_eq1" << "," << "sim2_eq2" << "," << "sim2_eq3" << endl;

  // Make sure the file is open
  if(!myFile.is_open())
    throw std::runtime_error("Could not open file");

  std::string line, colname;

  std::getline(file_gt, line);  // skip line with column names

  while(getline(file_gt, line)) {
    cout << "Processing image " << curr_img << "/" << NUM_IMGS << endl;

    // 1. preprocessing
    // 1.1 read the image
    char cimg[1000];
    snprintf(cimg, 1000, "%s%06d.png", imgFold, curr_imag);
    string ref_filename(cimg);
    Mat in = imread(ref_filename);
    frame_height = in.size[0];
    frame_width = in.size[1];

    // 1.2 convert to greyscale for later computing borders
    Mat grey;
    cvtColor(in, grey, CV_BGR2GRAY);

    // 1.3 compute the gradient image in x and y with the laplacian for the borders
    Mat grad;
    Laplacian(grey, grad, CV_8U, 1, 1, 0, BORDER_DEFAULT);

    // 1.4 detect edges, 50 and 255 as thresholds 1 and 2
    Mat edges;
    Canny(grad, edges, 50, 255, 3);

    // 1.5 close and dilate lines for some noise removal
    Mat morph;
    int morph_elem = 0;
    int morph_size = 0;
    Mat element = getStructuringElement(morph_elem, Size(2*morph_size + 1, 2*morph_size+1), cv::Point(morph_size, morph_size));
    morphologyEx(edges, edges, MORPH_CLOSE, element);
    dilate(edges, morph, Mat(), cv::Point(-1,-1), 1, BORDER_CONSTANT, morphologyDefaultBorderValue());

    // 1.6 detect lines using the hough transform
    std::vector<Vec4i> detected_lines;
    HoughLinesP(morph, detected_lines, 1, CV_PI/180., 60, 120, 50);

    // 2 extract parameters
    // from the angles of the lines from the hough transform, as said in luc's paper
    // this is done for ease of computation
    std::vector<double> lines_m_x, lines_m_y, filtered_m_x, filtered_m_y;  // x and y components of the points in M
    double x_hat, y_hat, a_hat;

    // structures for storing the lines information
    std::vector<line_t> lines;  // stores the lines obtained from the hough transform
    std::vector<double> lines_angles;  // stores the angles of the lines from the hough transform with the image compressed between [-pi/4, pi/4]

    double line_angle, line_angle4, d, dd, m_x, m_y;

    // 2.1 extract the informations from the good detected lines
    for(int i=0; i<detected_lines.size(); i++) {
      Vec4i l = detected_lines[i];
      double p1_x = l[0], p1_y = l[1];
      double p2_x = l[2], p2_y = l[3];

      line_angle = atan2(p2_y - p1_y, p2_x - p1_x);               // get the angle of the line from the existing points
      line_angle4 = modulo(line_angle+M_PI/4., M_PI/2.)-M_PI/4.;  // compress image between [-pi/4, pi/4]

      m_x = cos(4*line_angle);
      m_y = sin(4*line_angle);

      // 2.1.1 smallest radius of a circle with a point belonging to the line with origin in 0
      d = ((p2_x-p1_x)*(p1_y)-(p1_x)*(p2_y-p1_y)) / sqrt(pow(p2_x-p1_x, 2)+pow(p2_y-p1_y, 2));

      // 2.1.2 decimal distance, displacement between the lines
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

    / 1.7.2 median of the components of the lines
    x_hat = median(lines, 3);
    y_hat = median(lines, 4);

    std::vector<line_t> lines_good;

    // 2.2 filter lines with bad orientation
    for (line_t l : lines) {
      if ((abs(x_hat - l.m_x) + abs(y_hat - l.m_y)) < 0.15) {
        filtered_m_x.push_back(l.m_x);
        filtered_m_y.push_back(l.m_y);

        line(src, l.p1, l.p2, Scalar(255, 0, 0), 3, LINE_AA);
        lines_good.push_back(l);

      } else {
        line(src, l.p1, l.p2, Scalar(0, 0, 255), 3, LINE_AA);

      }
    }

    x_hat = median(filtered_m_x);
    y_hat = median(filtered_m_y);
    a_hat = atan2(y_hat, x_hat) * 1/4;

    if(lines_good.size() > MIN_GOOD_LINES) {
      cout << "Found " << lines_good.size() << " good lines" << endl;
      Mat rot = Mat::zeros(Size(frame_width , frame_height), CV_8UC3);
      std::vector<line_t> bag_h, bag_v;
      double x1, y1, x2, y2;
      double angle_new;

      // 2.3 generate bags of horizontal and vertical lines
      for (line_t l : lines_good) {
        //translation in order to center lines around 0
        x1 = l.p1.x - frame_width/2.0f;
        y1 = l.p1.y - frame_height/2.0f;
        x2 = l.p2.x - frame_width/2.0f;
        y2 = l.p2.y - frame_height/2.0f;

        // 2.3.1 applies the 2d rotation to the line, making it either horizontal or vertical
//        if (l.angle > M_PI/2 && l.angle < M_PI || l.angle > 3*M_PI/2 && l.angle < 2*M_PI)
//          a_hat -= M_PI/2;

        double x1_temp = x1 * cos(-a_hat) - y1 * sin(-a_hat);
        double y1_temp = x1 * sin(-a_hat) + y1 * cos(-a_hat);

        double x2_temp = x2 * cos(-a_hat) - y2 * sin(-a_hat);
        double y2_temp = x2 * sin(-a_hat) + y2 * cos(-a_hat);

        // 2.3.2 translates the image back
        x1 = x1_temp + frame_width/2.0f;
        x2 = x2_temp + frame_width/2.0f;

        y1 = y1_temp + frame_height/2.0f;
        y2 = y2_temp + frame_height/2.0f;

        // 2.3.3 compute the new angle of the rotated lines
        angle_new = atan2(y2-y1, x2-x1);

        // 2.3.4 determine if the lines are horizontal or vertical
        if (abs(cos(angle_new)) < 0.2) {  // vertical
          line(rot, cv::Point(x1, y1), cv::Point(x2, y2), Scalar(255, 255, 255), 1, LINE_AA);
          line(src, l.p1, l.p2, Scalar(255, 0, 0), 3, LINE_AA);
          bag_v.push_back(l);

        } else if (abs(sin(angle_new)) < 0.2) {  // horizontal
          line(rot, cv::Point(x1, y1), cv::Point(x2, y2), Scalar(0, 0, 255), 1, LINE_AA);
          bag_h.push_back(l);
          line(src, l.p1, l.p2, Scalar(0, 255, 0), 3, LINE_AA);
        }
      }

      // 2.4 get displacements parameters
      double d_hat_h = dist_lines * median(bag_h, 6);
      double d_hat_v = dist_lines * median(bag_v, 6);

      cout << "PARAMETERS -> d_hat_h = " << d_hat_h << " | d_hat_v = " << d_hat_v << " | a_hat = " a_hat << endl;

      obs = ibex::IntervalVector({
          {d_hat_h, d_hat_h},
          {d_hat_v, d_hat_v},
          {a_hat, a_hat}
      }).inflate(ERROR_OBS);

      obs[2] = ibex::Interval(a_hat, a_hat).inflate(ERROR_OBS_ANGLE);

      return obs;
    } else {
      cout << "Not enough good lines (" << lines_good.size() << ")" << endl;
    }

    // 3 generate the representation of the observed parameters
    Mat view_param_1 = generate_grid_1(dist_lines, obs);
    Mat view_param_2 = generate_grid_2(dist_lines, obs);

    if(display_window) {
      cv::imshow("camera", in);
      cv::imshow("lines", src);
      cv::imshow("rotated", rot);
      cv::imshow("view_param_1", view_param_1);
      cv::imshow("view_param_2", view_param_2);
    }

    // 4 get the pose from the ground truth
    stringstream ss(line);  // stringstream of the current line
    vector<double> line_val;
    double val;

    // extract each value
    while(ss >> val){
        line_val.push_back(val);

        if(ss.peek() == ',')
          ss.ignore();  // ignore commas
    }

    pose_1 = line_vals[0];
    pose_2 = line_vals[1];
    pose_3 = line_vals[2];

    // 5 equivalency equations

    // ground truth and parameters should have near 0 value in the equivalency equations
    sim1_eq1 = sin(M_PI*(obs[0].mid()-pose_1));
    sim1_eq2 = sin(M_PI*(obs[1].mid()-pose_2));
    sim1_eq3 = sin(obs[2].mid()-pose_3);

    sim2_eq1 = sin(M_PI*(obs[0].mid()-pose_2));
    sim2_eq2 = sin(M_PI*(obs[1].mid()-pose_1));
    sim2_eq3 = cos(obs[2].mid()-pose_3);

    file_sim << sim1_eq1 << "," << sim1_eq2 << "," << sim1_eq3 << "," << sim2_eq1 << "," << sim2_eq2 << "," << sim2_eq3 << endl;

    cout << "Equivalence equations 1:\nsin(pi*(y1-z1)) = " << sim1_eq1 << "\nsin(pi*(y2-z2)) = " << sim1_eq2 << "\nsin(y2-z2) = " << sim1_eq3 << endl;
    cout << "Equivalence equations 2:\nsin(pi*(y1-z2)) = " << sim2_eq1 << "\nsin(pi*(y2-z1)) = " << sim2_eq2 << "\ncos(y2-z1) = " << sim2_eq3 << endl;

    curr_img += 1;
  }  // end of loop for each image
}
