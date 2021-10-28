#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <ibex.h>
#include <codac.h>
#include <codac-rob.h>
#include <math.h>
#include <stdarg.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include <ncurses.h>  // for interactivity
#include <boost/program_options.hpp>  // for command line options

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

using namespace cv;
using namespace std;
using namespace ibex;
using namespace codac;

#define MIN_GOOD_LINES 5

#define ERROR_PRED      0.04
#define ERROR_OBS       0.04
#define ERROR_OBS_ANGLE 0.02

#define DATASET "report_testing_environment" // "centered"
string path_test;

int num_imgs = 13758;

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

typedef struct robot_struct{
  Point2f p1;     // 1st point of the robot triangle
  Point2f p2;     // 2nd point of the robot triangle
  Point2f p3;     // 2nd point of the robot triangle
  double angle;
} robot_t;

double sawtooth(double x);
double modulo(double a, double b);
int sign(double x);
double median(std::vector<line_t> lines, int op);
cv::Point2f rotate_pt(cv::Point2f pt, double alpha, cv::Point2f c);

cv::Mat generate_grid_1(int dist_lines, ibex::IntervalVector obs);
cv::Mat generate_grid_2(int dist_lines, ibex::IntervalVector obs);
cv::Mat gen_img(std::vector<double> expected);
cv::Mat gen_img_rot(std::vector<double> expected);
/* colors:
 * pink       -> ground truth
 * dark green -> predicted state (state equations applied at previous state, before contraction)
 * light blue -> contracted state
 * yellow     -> oberved parameters of the image (first set)
 * orange     -> oberved parameters of the image (second set, x <-> y, 90 deg angle)
 */
cv::Mat generate_global_frame(ibex::IntervalVector state, ibex::IntervalVector obs, ibex::IntervalVector box, std::vector<double> pose);

robot_t rotate_robot(robot_t robot, double theta);
robot_t translate_robot(robot_t robot, double dx, double dy);

void ShowManyImages(string title, int nArgs, ...);

// variables for the images from the observation
std::vector<line_t> base_grid_lines;
bool base_grid_created = false;
std::vector<line_t> base_img_lines;
bool base_img_created = false;

// variables for the global frame from the state and the observation
std::vector<line_t> base_global_frame_lines;
bool base_global_frame_created = false;
cv::Mat base_global_frame;
robot_t base_robot;

int frame_width=256, frame_height=384;
int max_dim;

bool verbose = true;
bool interactive = false;
bool display_window = false;
bool intervals = false;

double tile_size = 0.166;    // size of the side of the tile, in meters, also seen as l
double px_per_m = 620.48;     // pixels per meter
int dist_lines = tile_size * px_per_m;  //pixels between each pair of lines

/* orientations are the quarters of the grid, 1 to 4, starting in the top right, counter-clockwise*/
double prev_a_hat, a_hat;
int curr_quart = 1;

namespace po = boost::program_options;  // for argument parsing

int main(int argc, char **argv) {
  // -------------- NCURSES setup ----------------- //
  char kb_key;
//  initscr();
  // ---------------------------------------------- //

  // -------------- BOOST options setup -------------- //
  // Declare the supported options.
  po::options_description desc("Execution options");

  desc.add_options()
    ("help", "produce help message")
    ("interactive", "let frame by frame view")
    ("intervals", "display intervals contraction")
    ("display", "display processed frames")
    ("ppm", po::value<double>(&px_per_m), "number of pixels per meter")
    ("tile-size", po::value<double>(&tile_size), "size of the side of the tile in meters")
    ("output-file", po::value<string>(), "output file");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  initscr();

  if (vm.count("interactive")) {
    interactive = true;
    printw("Interactive mode enabled\n");
  } else {
    interactive = false;
    printw("Interactive disabled\n");
  }

  if (vm.count("display")) {
    display_window = true;
    printw("Display enabled\n");
  } else {
    display_window = false;
    printw("Display disabled\n");
  }

  if (vm.count("intervals")) {
    intervals = true;
    printw("Display intervals enabled\n");
  } else {
    intervals = false;
    printw("Display intervals disabled\n");
  }

  if (vm.count("output-file")) {
    path_test = vm["output-file"].as<string>();
    printw("Output file specified by user: %c\n", path_test);
  } else {
    char temp[1000];
    snprintf(temp, 1000, "/home/birromer/robotics/data/tiles_loc/%s/test_sim.csv", DATASET);
    path_test = string(temp);
    printw("Output file default: %s\n", path_test);
  }

  printw("Number of pixels per meter: %f\n", px_per_m);
  printw("Size of the tle in meters: %f\n", tile_size);
  dist_lines = tile_size * px_per_m;  //pixels between each pair of lines
  printw("Distance between lines in pixels: %d\n", dist_lines);

  printw(" ================== Testing start ==================\n");
  // ---------------------------------------------- //

  // ----------------- VIBES setup ----------------- //
  if (intervals) {
    vibes::beginDrawing();
    VIBesFigMap fig_map("Map");
    vibes::setFigureProperties("Map",vibesParams("x", 10, "y", -10, "width", 700, "height", 700));
    vibes::axisLimits(-10, 10, -10, 10, "Map");
    fig_map.show();
  }
  // ---------------------------------------------- //

  // ----------------- ROOT setup ----------------- //
  if (DATASET == "centered") {
    num_imgs = 7496;
  } else if (DATASET == "not_centerd") {
    num_imgs = 9698;
  }

  TApplication rootapp("viz", &argc, argv);

  auto c1 = std::make_unique<TCanvas>("c1", "Equivalence equations");
  c1->SetWindowSize(1550, 700);

  auto f1 = std::make_unique<TGraph>(num_imgs);
  f1->SetTitle("sin(pi*(y1-z1))");
  f1->GetXaxis()->SetTitle("Iteration");
  f1->GetYaxis()->SetTitle("Similarity score");
  f1->SetMinimum(-1);
  f1->SetMaximum(1);

  auto f2 = std::make_unique<TGraph>(num_imgs);
  f2->SetTitle("sin(pi*(y2-z2))");
  f2->GetXaxis()->SetTitle("Iteration");
  f2->GetYaxis()->SetTitle("Similarity score");
  f2->SetMinimum(-1);
  f2->SetMaximum(1);

  auto f3 = std::make_unique<TGraph>(num_imgs);
  f3->SetTitle("sin(y3-z3)");
  f3->GetXaxis()->SetTitle("Iteration");
  f3->GetYaxis()->SetTitle("Similarity score");
  f3->SetMinimum(-1);
  f3->SetMaximum(1);

  auto f4 = std::make_unique<TGraph>(num_imgs);
  f4->SetTitle("sin(pi*(y1-z2))");
  f4->GetXaxis()->SetTitle("Iteration");
  f4->GetYaxis()->SetTitle("Similarity score");
  f4->SetMinimum(-1);
  f4->SetMaximum(1);

  auto f5 = std::make_unique<TGraph>(num_imgs);
  f5->SetTitle("sin(pi*(y2-z1))");
  f5->GetXaxis()->SetTitle("Iteration");
  f5->GetYaxis()->SetTitle("Similarity score");
  f5->SetMinimum(-1);
  f5->SetMaximum(1);

  auto f6 = std::make_unique<TGraph>(num_imgs);
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

  int curr_img = 0;

  vector<vector<double>> sim_test_data;
  vector<vector<double>> sim_ground_truth;

  if(display_window) {
    cv::namedWindow("steps");
    cv::namedWindow("global_frame");
    cv::startWindowThread();
  }

  double pose_1=0.0, pose_2=0.0, pose_3=0.0, timestamp=0.0;
  double expected_1, expected_2, expected_3;
  double offset_pose_1, offset_pose_2;
  bool first_pose = true;
  ibex::IntervalVector obs(3, ibex::Interval::ALL_REALS);    // observed parameters from the image
  ibex::IntervalVector state(3, ibex::Interval::ALL_REALS);  // working state from prediction (in this case gt)
  // --------- start file with testing data ----------- //

  ofstream file_sim(path_test);

  if(!file_sim.is_open())
    throw std::runtime_error("Could not open SIM file");

  file_sim << "sim1_eq1" << "," << "sim1_eq2" << "," << "sim1_eq3" << "," << "sim2_eq1" << "," << "sim2_eq2" << "," << "sim2_eq3" << endl;
  // ------------------------------------------------ //

  // ----------- get ground truth data -------------- //
  char path_gt[1000];
  snprintf(path_gt, 1000, "/home/birromer/robotics/data/tiles_loc/%s/gt.csv", DATASET);
  ifstream file_gt(path_gt);

  if(!file_gt.is_open())
    throw std::runtime_error("Could not open GT file");

  char path_error[1000];
  snprintf(path_error, 1000, "/home/birromer/robotics/data/tiles_loc/%s/error.csv", DATASET);
  ofstream file_error(path_error);
  file_error << "timestamp" << "," << "error_x" << "," << "error_y" << "," << "error_abs" << endl;

  if(!file_error.is_open())
    throw std::runtime_error("Could not open error output file");

  std::string line_content, colname;

  std::getline(file_gt, line_content);  // skip line with column names

  while(getline(file_gt, line_content)) {
    // get the pose from the ground truth
    stringstream ss(line_content);  // stringstream of the current line
    vector<double> line_vals;
    double val;

    // extract each value
    while(ss >> val){
        line_vals.push_back(val);  // will always have: timestamp, pose_x, pose_y, pose_th

        if(ss.peek() == ',')
          ss.ignore();  // ignore commas
    }

    if (first_pose) {
      offset_pose_1 = -line_vals[1];
      offset_pose_2 = -line_vals[2];
      first_pose = false;

      state = ibex::IntervalVector({
          {pose_1, pose_1},
          {pose_2, pose_2},
          {pose_3, pose_3}
      }).inflate(ERROR_PRED);
      // get first timestamp
      timestamp = line_vals[0];
      // initialize a_hat with the truth
      prev_a_hat = line_vals[3];
      a_hat = line_vals[3];
    }

    vector<double> pose{line_vals[0], line_vals[1] + offset_pose_1, line_vals[2] + offset_pose_2, line_vals[3]};  // timestamp, x, y, th
    sim_ground_truth.push_back(pose);
  }
  // ------------------------------------------------ //

  while(curr_img < num_imgs) {
    // 1. preprocessing
    // (generated) 1.1 generate imagem from gt parameters
    // 1.1.1 get the pose from the ground truth
    // access the vector where it is stored

    double prev_timestamp = timestamp;
    double prev_pose_1 = pose_1;
    double prev_pose_2 = pose_2;
    double prev_pose_3 = pose_3;

    timestamp = sim_ground_truth[curr_img][0];
    pose_1 = sim_ground_truth[curr_img][1];
    pose_2 = sim_ground_truth[curr_img][2];
    pose_3 = sim_ground_truth[curr_img][3];

    state = ibex::IntervalVector({
        {pose_1, pose_1},
        {pose_2, pose_2},
        {pose_3, pose_3}
    }).inflate(ERROR_PRED);

    double dt = (timestamp - prev_timestamp)*10;

    // predict position from state equations, considering that speed and angle are measured
    double u1 = sqrt(pow(pose_1 - prev_pose_1, 2) + pow(pose_2 - prev_pose_2, 2));  // u1 as the speed
    double u2 = pose_3 - prev_pose_3;  // u2 as the diff in heading

//    state[0] = state[0] + (u1 * ibex::cos(state[2])).inflate(ERROR_PRED) * dt;
//    state[1] = state[1] + (u1 * ibex::cos(state[2])).inflate(ERROR_PRED) * dt;
//    state[2] = state[2] + ibex::Interval(u2).inflate(ERROR_PRED) * dt;

    // 1.1.2 gerenate gt parameters
    expected_1 = (pose_1/tile_size - floor(pose_1/tile_size))*tile_size; // modulo(pose_1, tile_size);
    expected_2 = (pose_2/tile_size - floor(pose_2/tile_size))*tile_size; // modulo(pose_2, tile_size);
    expected_3 = ((pose_3-M_PI/4)/(M_PI/2) - floor((pose_3-M_PI/4)/(M_PI/2))-0.5) * 2*M_PI/4;  // modulo without function as it is in radians
//    expected_3 = modulo(pose_3+M_PI/4., M_PI/2.)-M_PI/4.;

    std::vector<double> expected{expected_1, expected_2, expected_3};

    // (dataset) 1.1 read the image
    char path_img[1000];
    snprintf(path_img, 1000, "/home/birromer/robotics/data/tiles_loc/%s/dataset_tiles/%06d.png", DATASET, curr_img);
    string ref_filename(path_img);
    Mat in_dataset = imread(ref_filename);
//    frame_height = in.size[0];
//    frame_width = in.size[1];
    circle(in_dataset, Point2i(frame_width/2, frame_height/2), 3, Scalar(0, 255, 0), 3);

    // here we can alternate the methods being compared
//    Mat in = gen_img_rot(expected);
//    Mat in = gen_img(expected);
    Mat in = in_dataset;

    Mat in_alt = gen_img_rot(expected);
//    Mat in_alt = gen_img(expected);
//    Mat in_alt = in_dataset;

    // 1.2 convert to greyscale for later computing borders
    Mat grey, grey_rot;
    cvtColor(in, grey, COLOR_BGR2GRAY);
    cvtColor(in_alt, grey_rot, COLOR_BGR2GRAY);

//    // create a skeleton representation, trying to diminish number of detected lines
//    cv::Mat img = grey;
//    cv::threshold(img, img, 127, 255, cv::THRESH_BINARY);
//
//    cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
//    cv::Mat temp(img.size(), CV_8UC1);
//
//    cv::Mat element_skel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
//
//    bool done;
//    do
//    {
//      cv::morphologyEx(img, temp, cv::MORPH_OPEN, element_skel);
//      cv::bitwise_not(temp, temp);
//      cv::bitwise_and(img, temp, temp);
//      cv::bitwise_or(skel, temp, skel);
//      cv::erode(img, img, element_skel);
//
//      double max;
//      cv::minMaxLoc(img, 0, &max);
//      done = (max == 0);
//    } while (!done);

    // 1.3 compute the gradient image in x and y with the laplacian for the borders
    Mat grad, grad_rot;
//    Laplacian(skel, grad, CV_8U, 1, 1, 0, BORDER_DEFAULT);
    Laplacian(grey, grad, CV_8U, 1, 1, 0, BORDER_DEFAULT);
    Laplacian(grey_rot, grad_rot, CV_8U, 1, 1, 0, BORDER_DEFAULT);

    // 1.4 detect edges, 50 and 255 as thresholds 1 and 2
    Mat edges, edges_rot;
    Canny(grad, edges, 50, 255, 3);
    Canny(grad_rot, edges_rot, 50, 255, 3);

    // 1.5 close and dilate lines for some noise removal
    Mat morph, morph_rot;
    int morph_elem = 0;
    int morph_size = 0;
    Mat element = getStructuringElement(morph_elem, Size(2*morph_size + 1, 2*morph_size+1), cv::Point(morph_size, morph_size));
    morphologyEx(edges, edges, MORPH_CLOSE, element);
    dilate(edges, morph, Mat(), cv::Point(-1,-1), 1, BORDER_CONSTANT, morphologyDefaultBorderValue());

    morphologyEx(edges_rot, edges_rot, MORPH_CLOSE, element);
    dilate(edges_rot, morph_rot, Mat(), cv::Point(-1,-1), 1, BORDER_CONSTANT, morphologyDefaultBorderValue());

    // 1.6 detect lines using the hough transform
    std::vector<Vec4i> detected_lines, detected_lines_rot;
    HoughLinesP(edges, detected_lines, 1, CV_PI/180., 60, 100, 50); // rho, theta, threshold, minLineLength, maxLineGap
    HoughLinesP(edges_rot, detected_lines_rot, 1, CV_PI/180., 60, 100, 50); // rho, theta, threshold, minLineLength, maxLineGap
//    HoughLinesP(edges, detected_lines, 1, CV_PI/180., 60, 120, 50); // rho, theta, threshold, minLineLength, maxLineGap

    // 1.7 filter lines and create single representation of multiple similar ones
    std::vector<Vec4i> limit_lines, limit_lines_rot;
    for(int i=0; i<detected_lines.size(); i++) {
      double p1_x = detected_lines[i][0], p1_y = detected_lines[i][1];
      double p2_x = detected_lines[i][2], p2_y = detected_lines[i][3];

      double a = (p2_y - p1_y) / (p2_x - p1_x);
      double b = p1_y - a * p1_x;

      double new_p1_y = 1;
      double new_p1_x = (new_p1_y - b)/a;

      double new_p2_y = frame_height-1;
      double new_p2_x = (new_p2_y - b)/a;

      if (new_p1_x > frame_width){
        new_p1_x = frame_width-1;
      } else if (new_p1_x < 0){
        new_p1_x = 1;
      }
      new_p1_y = a * new_p1_x + b;

      if (new_p2_x > frame_width){
        new_p2_x = frame_width-1;
      } else if (new_p2_x < 0){
        new_p2_x = 1;
      }
      new_p2_y = a * new_p2_x + b;

      Vec4i p(new_p1_x, new_p1_y, new_p2_x, new_p2_y);
      limit_lines.push_back(p);
    }
//    limit_lines = detected_lines;  // uncomment if want to ignore process above
    limit_lines_rot = detected_lines_rot;  // uncomment if want to ignore process above

    // 2.0 extract parameters from the angles of the lines from the hough transform, as said in luc's paper
    // this is done for ease of computation
    double x_hat, y_hat;
    double x_hat_rot, y_hat_rot, a_hat_rot;

    // structures for storing the lines information
    std::vector<line_t> lines, lines_rot;  // stores the lines obtained from the hough transform

    double line_angle, line_angle4, d, dd, m_x, m_y;
    double line_angle_rot, line_angle4_rot, d_rot, dd_rot, m_x_rot, m_y_rot;

    // 2.1 extract the informations from the good detected lines
    for(int i=0; i<limit_lines.size(); i++) {
      Vec4i l = limit_lines[i];
      double p1_x = l[0], p1_y = l[1];
      double p2_x = l[2], p2_y = l[3];

      line_angle = atan2(p2_y - p1_y, p2_x - p1_x);  // get the angle of the line from the existing points
      line_angle4 = ((line_angle-M_PI/4)/(M_PI/2) - floor((line_angle-M_PI/4)/(M_PI/2))-0.5) * 2*M_PI/4;;  // compress image between [-pi/4, pi/4]

      m_x = cos(4*line_angle);
      m_y = sin(4*line_angle);

      // 2.1.1 smallest radius of a circle with a point belonging to the line with origin in 0, being 0 corrected to the center of the image
      d = abs((p2_x-p1_x)*(p1_y-frame_height/2) - (p1_x-frame_width/2)*(p2_y-p1_y)) / sqrt(pow(p2_x-p1_x,2) + pow(p2_y-p1_y,2));  // value in pixels
      d = d / px_per_m;  // convert from pixels to meters

      // 2.1.2 decimal distance, displacement between the lines
      dd = (d/tile_size) - (floor(d/tile_size));  // this compresses image to [0, 1]

      line_t ln = {
        .p1     = cv::Point(p1_x, p1_y),
        .p2     = cv::Point(p2_x, p2_y),
        .angle  = line_angle,
        .angle4 = line_angle4,  // this is the one to be used as the angle of the line
        .m_x    = m_x,
        .m_y    = m_y,
        .d      = d,
        .dd     = dd,
        .side   = 0  // temporary value
      };

      // save the extracted information
      lines.push_back(ln);

      Vec4i l_rot = limit_lines_rot[i];
      double p1_x_rot = l_rot[0], p1_y_rot = l_rot[1];
      double p2_x_rot = l_rot[2], p2_y_rot = l_rot[3];

      line_angle_rot = atan2(p2_y_rot - p1_y_rot, p2_x_rot - p1_x_rot);  // get the angle of the line from the existing points
      line_angle4_rot = ((line_angle_rot-M_PI/4)/(M_PI/2) - floor((line_angle_rot-M_PI/4)/(M_PI/2))-0.5) * 2*M_PI/4;;  // compress image between [-pi/4, pi/4]

      m_x_rot = cos(4*line_angle_rot);
      m_y_rot = sin(4*line_angle_rot);

      // 2.1.1 smallest radius of a circle with a point belonging to the line with origin in 0, being 0 corrected to the center of the image
      d_rot = abs((p2_x_rot-p1_x_rot)*(p1_y_rot-frame_height/2.) - (p1_x_rot-frame_width/2.)*(p2_y_rot-p1_y_rot)) / sqrt(pow(p2_x_rot-p1_x_rot,2) + pow(p2_y_rot-p1_y_rot,2));  // value in pixels
      d_rot = d_rot / px_per_m;  // convert from pixels to meters

      // 2.1.2 decimal distance, displacement between the lines
      dd_rot = (d_rot/tile_size) - (floor(d_rot/tile_size));  // this compresses image to [0, 1]

//      printw("dd -> normal: %.3f <=> rot: %.3f  ||  \n", dd, dd_rot);
//
      line_t ln_rot = {
        .p1     = cv::Point(p1_x_rot, p1_y_rot),
        .p2     = cv::Point(p2_x_rot, p2_y_rot),
        .angle  = line_angle_rot,
        .angle4 = line_angle4_rot,  // this is the one to be used as the angle of the line
        .m_x    = m_x_rot,
        .m_y    = m_y_rot,
        .d      = d_rot,
        .dd     = dd_rot,
        .side   = 0  // temporary value
      };

      lines_rot.push_back(ln_rot);

    }

    sort(lines.begin(), lines.end(), [=](line_t l1, line_t l2) -> bool { return l1.dd < l2.dd; });
    sort(lines_rot.begin(), lines_rot.end(), [=](line_t l1, line_t l2) -> bool { return l1.dd < l2.dd; });

//    printw("\ndd normal: ");
//    for (line_t l : lines) {
//      printw("%.4f, ", l.dd);
//    }
//
//    printw("\ndd rot:    ");
//    for (line_t l : lines_rot) {
//      printw("%.4f, ", l.dd);
//    }

    // 2.1.3 median of the components of the lines
    x_hat = median(lines, 3);
    y_hat = median(lines, 4);

    std::vector<line_t> lines_good;

    Mat src = Mat::zeros(Size(frame_width, frame_height), CV_8UC3);
    // 2.2 filter lines with bad orientation
    for (line_t l : lines) {
      if ((abs(x_hat - l.m_x) + abs(y_hat - l.m_y)) < 0.05) {
        line(src, l.p1, l.p2, Scalar(255, 0, 0), 3, LINE_AA);
        lines_good.push_back(l);
      } else {
        line(src, l.p1, l.p2, Scalar(0, 0, 255), 3, LINE_AA);
      }
    }

    x_hat = median(lines_good, 3);
    y_hat = median(lines_good, 4);

    prev_a_hat = a_hat;
    a_hat = atan2(y_hat, x_hat) * 1./4.;
//    a_hat = median(lines_good, 2);

    if (a_hat > 0 and prev_a_hat < 0)
      curr_quart -= 1;
    else if (a_hat < 0 and prev_a_hat > 0)
      curr_quart += 1;

    if (curr_quart > 4)
      curr_quart = 1;
    else if (curr_quart < 1)
      curr_quart = 4;

    printw("\nCurrent quarter is: %d\n", curr_quart);

    Mat rot = Mat::zeros(Size(frame_width , frame_height), CV_8UC3);

    if(lines_good.size() > MIN_GOOD_LINES) {
      printw("\nFound %d good lines\n", lines_good.size());

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
        if (abs(cos(angle_new)) < 0.2) {  // vertical lines, may be used to measure the horizontal displacement
          line(rot, cv::Point(x1, y1), cv::Point(x2, y2), Scalar(255, 255, 255), 1, LINE_AA);
          line(src, l.p1, l.p2, Scalar(255, 0, 0), 3, LINE_AA);

          if (curr_quart == 4 or curr_quart == 3) {
            if (l.p1.x < frame_width/2.) {
              l.dd = 1 - l.dd;
            }
          } else if (curr_quart == 1 or curr_quart == 2) {
            if (l.p1.x >= frame_width/2.) {
              l.dd = 1 - l.dd;
            }
          }

          bag_v.push_back(l);

        } else if (abs(sin(angle_new)) < 0.2) {  // horizontal lines, may be used to measure the vertical displacement
          line(rot, cv::Point(x1, y1), cv::Point(x2, y2), Scalar(0, 0, 255), 1, LINE_AA);

          if (curr_quart == 1 or curr_quart == 4) {
            if (l.p1.y < frame_height/2.) {
              l.dd = 1 - l.dd;
            }
          } else if (curr_quart == 2 or curr_quart == 3) {
            if (l.p1.y >= frame_height/2.) {
              l.dd = 1 - l.dd;
            }
          }

          bag_h.push_back(l);
          line(src, l.p1, l.p2, Scalar(0, 255, 0), 3, LINE_AA);
        }
      }

      // 2.4 get displacements parameters
      // for the horizontal displacement, it should consider the offset in the x axis (between vertical lines), and the opposite for vertical
      // displacement however there is no distinction between horizontal and vertical with the robot's knowledge, and the ambiguity is taken
      // into consideration in the equivalence
      double d_hat_h = tile_size * median(bag_v, 6);  // the median gives a value from 0 to 1 of displacement, multiplying by the tile size positions it in the world
      double d_hat_v = tile_size * median(bag_h, 6);

      obs = ibex::IntervalVector({
          {d_hat_h, d_hat_h},
          {d_hat_v, d_hat_v},
          {a_hat, a_hat}
      }).inflate(ERROR_OBS);

      obs[2] = ibex::Interval(a_hat, a_hat).inflate(ERROR_OBS_ANGLE);

    } else {
      printw("Not enough good lines (%d)\n", lines_good.size());
    }

    ibex::IntervalVector expected_i(3, ibex::Interval::ALL_REALS);
    expected_i = ibex::IntervalVector({
        {expected_1, expected_1},
        {expected_2, expected_2},
        {expected_3, expected_3}
    }).inflate(ERROR_OBS);

    // 3 generate the representation of the observed parameters
    Mat view_param_1 = generate_grid_1(dist_lines, obs);
    Mat view_param_2 = generate_grid_2(dist_lines, obs);
//    Mat view_param_1 = generate_grid_1(dist_lines, expected_i);
//    Mat view_param_2 = generate_grid_2(dist_lines, expected_i);


    printw("\nTIMESTAMP: %f | dt: %f\n", timestamp, dt);
    printw("INPUT      ->      u1 = %f |      u2 = %f\n", u1, u2);
    printw("POSE diff  ->      x1 = %f |      x2 = %f |    x3 = %f\n", pose_1-prev_pose_1, pose_2-prev_pose_2, pose_3-prev_pose_3);
    printw("------------------------------------------------------\n");
    printw("POSE       ->      x1 = %f |      x2 = %f |    x3 = %f\n", pose_1, pose_2, pose_3);
    printw("EXPECTED   -> d_hat_h = %f | d_hat_v = %f | a_hat = %f\n", expected_1, expected_2, expected_3);
    printw("PARAMETERS -> d_hat_h = %f | d_hat_v = %f | a_hat = %f\n", obs[0].mid(), obs[1].mid(), obs[2].mid());;
    printw("STATE      ->      x1 = %f |      x2 = %f |    x3 = %f\n", state[0].mid(), state[1].mid(), state[2].mid());

//    double tsim1_eq1 = sin(M_PI*(expected_1-pose_1));
//    double tsim1_eq2 = sin(M_PI*(expected_2-pose_2));
//    double tsim1_eq3 = sin(expected_3-pose_3);
//    double tsim2_eq1 = sin(M_PI*(expected_1-pose_2));
//    double tsim2_eq2 = sin(M_PI*(expected_2-pose_1));
//    double tsim2_eq3 = cos(expected_3-pose_3);
//    printw("\nEquivalence equations 1:\n  sin(pi*(y1-z1)) = %f\n  sin(pi*(y2-z2)) = %f\n  sin(y3-z3) = %f\n", tsim1_eq1, tsim1_eq2, tsim1_eq3);
//    printw("\nEquivalence equations 2:\n  sin(pi*(y1-z2)) = %f\n  sin(pi*(y2-z1)) = %f\n  cos(y3-z3) = %f\n", tsim2_eq1, tsim2_eq2, tsim2_eq3);

    // 5 equivalency equations
    ibex::IntervalVector box0(6, ibex::Interval::ALL_REALS);
    ibex::IntervalVector box1(6, ibex::Interval::ALL_REALS);

    box0[0] = state[0], box0[1] = state[1], box0[2] = state[2], box0[3] = obs[0], box0[4] = obs[1], box0[5] = obs[2];
    box1[0] = state[0], box1[1] = state[1], box1[2] = state[2], box1[3] = obs[0], box1[4] = obs[1], box1[5] = obs[2];

//    ibex::Function fun1("x[3]", "y[3]", "(sin(pi*(x[0]-y[0])/1.0) ; sin(pi*(x[1]-y[1])/1.0) ; sin(x[2]-y[2]))");
//    ibex::Function fun2("x[3]", "y[3]", "(sin(pi*(x[0]-y[1])/1.0) ; sin(pi*(x[1]-y[0])/1.0) ; cos(x[2]-y[2]))");

    char fun1_char[100];
    char fun2_char[100];
    snprintf(fun1_char, 100, "(sin(pi*(x[0]-y[0])/%.3f) ; sin(pi*(x[1]-y[1])/%.3f) ; sin(x[2]-y[2]))", tile_size, tile_size); // 1.0, 1.0); //
    snprintf(fun2_char, 100, "(sin(pi*(x[0]-y[1])/%.3f) ; sin(pi*(x[1]-y[0])/%.3f) ; cos(x[2]-y[2]))", tile_size, tile_size); // 1.0, 1.0); //
//    snprintf(fun1_char, 100, "(sin(pi*(x[0]-y[0])/%.3f) ; sin(pi*(x[1]-y[1])/%.3f))", tile_size, tile_size); // 1.0, 1.0); //
//    snprintf(fun2_char, 100, "(sin(pi*(x[0]-y[1])/%.3f) ; sin(pi*(x[1]-y[0])/%.3f))", tile_size, tile_size); // 1.0, 1.0); //

    ibex::Function fun1("x[3]", "y[3]", fun1_char);
    ibex::Function fun2("x[3]", "y[3]", fun2_char);

    ibex::CtcFwdBwd ctc1(fun1);
    ibex::CtcFwdBwd ctc2(fun2);

    ctc1.contract(box0);
    ctc2.contract(box1);

    ibex::IntervalVector box(3, ibex::Interval::ALL_REALS);
    box[0] = box0[0] | box1[0];
    box[1] = box0[1] | box1[1];
    box[2] = box0[2] | box1[2];

    if(box[0].is_empty() or box[1].is_empty()) {
      printw("Could not contract the state (!!!).\n");
    } else {
      printw("CONTRACTION->      x1 = %f |      x2 = %f |    x3 = %f\n", box[0].mid(), box[1].mid(), box[2].mid());
      printw("------------------------------------------------------------------------\n");
      printw("STATE      ->      x1 = [%f, %f] |      x2 = [%f, %f] |    x3 = [%f, %f]\n", state[0].lb(), state[0].ub(), state[1].lb(), state[1].ub(), state[2].lb(), state[2].ub());
      printw("CONTRACTION->      x1 = [%f, %f] |      x2 = [%f, %f] |    x3 = [%f, %f]\n", box[0].lb(), box[0].ub(), box[1].lb(), box[1].ub(), box[2].lb(), box[2].ub());
      printw(" Distance from center to truth: %.2f cm\n", sqrt(pow(pose_1 - box[0].mid(), 2) + pow(pose_2 - box[1].mid(), 2))*100);
      file_error << timestamp << "," << (pose_1 - box[0].mid())*100 << "," << (pose_2 - box[1].mid())*100 << "," << pose_3 - box[2].mid() << "," << sqrt(pow(pose_1 - box[0].mid(), 2) + pow(pose_2 - box[1].mid(), 2))*100 << endl;
    }

    if (intervals) {
      // draw ground truth
      vibes::drawVehicle(pose_1, pose_2, pose_3/M_PI*180., 0.3, "pink");  // draw ground truth

      // draw the predicted state (when testing only parameters, it's the ground truth)
      vibes::drawBox(state.subvector(0, 1), "green");
      vibes::drawVehicle(state[0].mid(), state[1].mid(), (state[2].mid())*180./M_PI, 0.3, "green");

      // draw the contracted state
      vibes::drawBox(box.subvector(0, 1), "blue");
      vibes::drawVehicle(box[0].mid(), box[1].mid(), (box[2].mid())*180./M_PI, 0.3, "blue");
    }

    // EXTRA visual stuff

    // global frame with observed parameter, pose and contraction
    // state is the prediction without contraction, obs are the parameters, box is the contraction and pose is the ground truth
    Mat view_global_frame = generate_global_frame(state, obs, box, sim_ground_truth[curr_img]);

    // display steps and global frame
    if(display_window) {
//      cvtColor(grad, grad, COLOR_GRAY2BGR);
//      cvtColor(edges, edges, COLOR_GRAY2BGR);
//      ShowManyImages("steps", 6, in, src, grad, edges, view_param_1, view_param_2);//
      ShowManyImages("steps", 4, in, view_param_1, view_param_2, in_alt);
      cv::imshow("global_frame", view_global_frame);
    }

    // ground truth and parameters should have near 0 value in the equivalency equations
//    double sim1_eq1 = sin(M_PI*(pose_1-obs[0].mid())/tile_size);
//    double sim1_eq2 = sin(M_PI*(pose_2-obs[1].mid())/tile_size);
//    double sim1_eq3 = sin(pose_3-obs[2].mid());
//
//    double sim2_eq1 = sin(M_PI*(pose_2-obs[0].mid())/tile_size);
//    double sim2_eq2 = sin(M_PI*(pose_1-obs[1].mid())/tile_size);
//    double sim2_eq3 = cos(pose_3-obs[2].mid());

    // comparisson between prediction and parameters, that's what is considered for the contraction
    double sim1_eq1 = sin(M_PI*(state[0].mid()-obs[0].mid())/tile_size);
    double sim1_eq2 = sin(M_PI*(state[1].mid()-obs[1].mid())/tile_size);
    double sim1_eq3 = sin(state[2].mid()-obs[2].mid());

    double sim2_eq1 = sin(M_PI*(state[1].mid()-obs[0].mid())/tile_size);
    double sim2_eq2 = sin(M_PI*(state[0].mid()-obs[1].mid())/tile_size);
    double sim2_eq3 = cos(state[2].mid()-obs[2].mid());

    file_sim << sim1_eq1 << "," << sim1_eq2 << "," << sim1_eq3 << "," << sim2_eq1 << "," << sim2_eq2 << "," << sim2_eq3 << endl;

    printw("\nEquivalence equations 1:\n  sin(pi*(z1-y1)/ts) = %f\n  sin(pi*(z2-y2/ts)) = %f\n  sin(z3-y3) = %f\n", sim1_eq1, sim1_eq2, sim1_eq3);
    printw("Equivalence equations 2:\n  sin(pi*(z1-y2)/ts) = %f\n  sin(pi*(z2-y1)/ts) = %f\n  cos(z3-y3) = %f\n", sim2_eq1, sim2_eq2, sim2_eq3);

    vector<double> s{sim1_eq1, sim1_eq2, sim1_eq3, sim2_eq1, sim2_eq2, sim2_eq3};

    if (sim_test_data.size() == curr_img){
      sim_test_data.push_back(s);
    } else {
      sim_test_data.at(curr_img) = s;
    }

    // redraw graph
    for (int i=0; i < curr_img; i++) {
      if (i < curr_img) {
        f1->SetPoint(i, i, sim_test_data[i][0]);
        f2->SetPoint(i, i, sim_test_data[i][1]);
        f3->SetPoint(i, i, sim_test_data[i][2]);
        f4->SetPoint(i, i, sim_test_data[i][3]);
        f5->SetPoint(i, i, sim_test_data[i][4]);
        f6->SetPoint(i, i, sim_test_data[i][5]);
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

    if (interactive) {
      kb_key = getch();

      if (kb_key == 104) {
        if (curr_img > 0) {
          curr_img -= 1;
          sim_test_data.pop_back();
        }
      } else if (kb_key == 108) {
        if (curr_img < num_imgs)
          curr_img += 1;
      }
    } else {
      curr_img += 1;
    }

    // NOTE : this update has to be after the visualization stuff
    if(!box[0].is_empty() and !box[1].is_empty()) {
      state[0] = box[0];
      state[1] = box[1];
      state[2] = box[2];
    }

    clear();
  }  // end of loop for each image

  endwin();  // finish ncurses mode
  if (intervals)
    vibes::endDrawing();  // close vibes drawing
  printw("Exited with success\n");
  file_error.close();
//  file_gt.close();
}

double sawtooth(double x){
  return 2.*atan(tan(x/2.));
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

cv::Mat gen_img(std::vector<double> expected) {
  double d_hat_h = expected[0] * px_per_m;  // parameters have to be scaled for being shown in pixels
  double d_hat_v = expected[1] * px_per_m;
  double a_hat   = expected[2];

  int n_lines = 10;
  int max_dim = frame_height > frame_width? frame_height : frame_width;  // largest dimension so that always show something inside the picture

  // center of the image, where tiles start with zero displacement
  cv::Point2f center(frame_width/2, frame_height/2);

  if (!base_img_created) {
    // create a line every specified number of pixels
    // adds one before and one after because occluded areas may appear
    int pos_x = center.x - (n_lines/2)*dist_lines;
    while (pos_x <= frame_width + (n_lines/2)*dist_lines) {
      line_t ln = {
        .p1     = cv::Point(pos_x, -max_dim),
        .p2     = cv::Point(pos_x, max_dim),
        .side   = 1  // 0 horizontal, 1 vertical
      };
      base_img_lines.push_back(ln);
      pos_x += dist_lines;
    }

    int pos_y = center.y - (n_lines/2)*dist_lines;
    while (pos_y <= frame_height + (n_lines/2)*dist_lines) {
      line_t ln = {
        .p1     = cv::Point(-max_dim, pos_y),
        .p2     = cv::Point(max_dim, pos_y),
        .side   = 0  // 0 horizontal, 1 vertical
      };
      base_img_lines.push_back(ln);
      pos_y += dist_lines;
    }

    base_img_created = true;
  }

  cv::Mat img_grid = cv::Mat::zeros(frame_height, frame_width, CV_8UC3);
  img_grid.setTo(cv::Scalar(255, 255, 255));
  std::vector<line_t> grid_lines = base_img_lines;

  for (line_t l : grid_lines) {
    // adds displacement
    l.p1.x += d_hat_v;
    l.p1.y += d_hat_h;

    l.p2.x += d_hat_v;
    l.p2.y += d_hat_h;

//    // applies the 2d rotation to the line
//    cv::Point2f p1_rot = rotate_pt(l.p1, -a_hat, center);
//    cv::Point2f p2_rot = rotate_pt(l.p2, -a_hat, center);

//    p1_rot.x += d_hat_v;
//    p1_rot.y += d_hat_h;
//
//    p2_rot.x += d_hat_v;
//    p2_rot.y += d_hat_h;

//    line(img_grid, cv::Point(p1_rot.x, p1_rot.y), cv::Point(p2_rot.x, p2_rot.y), Scalar(0, 0, 0), 2, LINE_AA);
    line(img_grid, cv::Point(l.p1.x, l.p1.y), cv::Point(l.p2.x, l.p2.y), Scalar(0, 0, 0), 2, LINE_AA);
  }

  // using point at tile at position (1,1) as reference
  cv::Point2f ref(center.x + dist_lines, center.y - dist_lines);
//  ref = rotate_pt(ref, -a_hat, center);
  ref.x += d_hat_v;
  ref.y += d_hat_h;

  circle(img_grid, Point2i(ref.x, ref.y), 3, Scalar(0, 0, 255), 2);
  circle(img_grid, Point2i(center.x, center.y), 3, Scalar(0, 255, 0), 3);

  return img_grid;
}

cv::Mat gen_img_rot(std::vector<double> expected) {
  double d_hat_h = expected[0] * px_per_m;  // parameters have to be scaled for being shown in pixels
  double d_hat_v = expected[1] * px_per_m;
  double a_hat   = expected[2];

  int n_lines = 10;
  int max_dim = frame_height > frame_width? frame_height : frame_width;  // largest dimension so that always show something inside the picture

  // center of the image, where tiles start with zero displacement
  cv::Point2f center(frame_width/2, frame_height/2);

  if (!base_img_created) {
    // create a line every specified number of pixels
    // adds one before and one after because occluded areas may appear
    int pos_x = center.x - (n_lines/2)*dist_lines;
    while (pos_x <= frame_width + (n_lines/2)*dist_lines) {
      line_t ln = {
        .p1     = cv::Point(pos_x, -max_dim),
        .p2     = cv::Point(pos_x, max_dim),
        .side   = 1  // 0 horizontal, 1 vertical
      };
      base_img_lines.push_back(ln);
      pos_x += dist_lines;
    }

    int pos_y = center.y - (n_lines/2)*dist_lines;
    while (pos_y <= frame_height + (n_lines/2)*dist_lines) {
      line_t ln = {
        .p1     = cv::Point(-max_dim, pos_y),
        .p2     = cv::Point(max_dim, pos_y),
        .side   = 0  // 0 horizontal, 1 vertical
      };
      base_img_lines.push_back(ln);
      pos_y += dist_lines;
    }

    base_img_created = true;
  }

  cv::Mat img_grid = cv::Mat::zeros(frame_height, frame_width, CV_8UC3);
  img_grid.setTo(cv::Scalar(255, 255, 255));
  std::vector<line_t> grid_lines = base_img_lines;

  for (line_t l : grid_lines) {
    // adds displacement
    l.p1.x += d_hat_v;
    l.p1.y += d_hat_h;

    l.p2.x += d_hat_v;
    l.p2.y += d_hat_h;

    // applies the 2d rotation to the line
    cv::Point2f p1_rot = rotate_pt(l.p1, -a_hat, center);
    cv::Point2f p2_rot = rotate_pt(l.p2, -a_hat, center);

    line(img_grid, cv::Point(p1_rot.x, p1_rot.y), cv::Point(p2_rot.x, p2_rot.y), Scalar(0, 0, 0), 2, LINE_AA);
  }

  // using point at tile at position (1,1) as reference
  cv::Point2f ref(center.x + dist_lines, center.y - dist_lines);
  ref.x += d_hat_v;
  ref.y += d_hat_h;
  ref = rotate_pt(ref, -a_hat, center);

  circle(img_grid, Point2i(ref.x, ref.y), 3, Scalar(0, 0, 255), 2);
  circle(img_grid, Point2i(center.x, center.y), 3, Scalar(0, 255, 0), 3);

  return img_grid;
}

cv::Point2f rotate_pt(cv::Point2f pt, double alpha, cv::Point2f c) {
    double c_a = cos(alpha);
    double s_a = sin(alpha);

    double x1_tmp =  pt.x*c_a + pt.y*s_a + (1-c_a)*c.x -     s_a*c.y;
    double y1_tmp = -pt.x*s_a + pt.y*c_a +     s_a*c.x + (1-c_a)*c.y;

    return cv::Point2f(x1_tmp, y1_tmp);
}

cv::Mat generate_grid_1(int dist_lines, ibex::IntervalVector obs) {
  double d_hat_h = obs[0].mid() * px_per_m;  // parameters have to be scaled for being shown in pixels
  double d_hat_v = obs[1].mid() * px_per_m;
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
    while (pos_x <= frame_width + (n_lines/2)*dist_lines) {
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
    while (pos_y <= frame_height + (n_lines/2)*dist_lines) {
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
    double x1 = l.p1.x - frame_width/2. ;// + d_hat_h;
    double y1 = l.p1.y - frame_height/2.;// + d_hat_v;
    double x2 = l.p2.x - frame_width/2. ;// + d_hat_h;
    double y2 = l.p2.y - frame_height/2.;// + d_hat_v;

    // applies the 2d rotation to the line, making it either horizontal or vertical
    double x1_temp = x1 * cos(a_hat) - y1 * sin(a_hat);//x1;//
    double y1_temp = x1 * sin(a_hat) + y1 * cos(a_hat);//y1;//

    double x2_temp = x2 * cos(a_hat) - y2 * sin(a_hat);//x2;//
    double y2_temp = x2 * sin(a_hat) + y2 * cos(a_hat);//y2;//

    // translates the image back and adds displacement
    x1 = (x1_temp + frame_width/2. + d_hat_h);
    y1 = (y1_temp + frame_height/2. + d_hat_v);
    x2 = (x2_temp + frame_width/2. + d_hat_h);
    y2 = (y2_temp + frame_height/2. + d_hat_v);

    if (l.side == 1) {
      line(img_grid, cv::Point(x1, y1), cv::Point(x2, y2), Scalar(255, 0, 0), 3, LINE_AA);
    } else {
      line(img_grid, cv::Point(x1, y1), cv::Point(x2, y2), Scalar(0, 255, 0), 3, LINE_AA);
    }
  }

  return img_grid;
}

cv::Mat generate_grid_2(int dist_lines, ibex::IntervalVector obs) {
  double d_hat_h = obs[1].mid() * px_per_m;  // parameters have to be scaled to be shown in pixels
  double d_hat_v = obs[0].mid() * px_per_m;
  double a_hat   = obs[2].mid() + M_PI/2.;

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
    while (pos_x <= frame_width + (n_lines/2)*dist_lines) {
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
    while (pos_y <= frame_height + (n_lines/2)*dist_lines) {
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
    double x1 = l.p1.x - frame_width/2. ;
    double y1 = l.p1.y - frame_height/2.;
    double x2 = l.p2.x - frame_width/2. ;
    double y2 = l.p2.y - frame_height/2.;

    // applies the 2d rotation to the line, making it either horizontal or vertical
    double x1_temp = x1 * cos(a_hat) - y1 * sin(a_hat);//x1;//
    double y1_temp = x1 * sin(a_hat) + y1 * cos(a_hat);//y1;//
                                                       //
    double x2_temp = x2 * cos(a_hat) - y2 * sin(a_hat);//x2;//
    double y2_temp = x2 * sin(a_hat) + y2 * cos(a_hat);//y2;//

    // translates the image back and adds displacement
    x1 = (x1_temp + frame_width/2. + d_hat_h);
    y1 = (y1_temp + frame_height/2. + d_hat_v);
    x2 = (x2_temp + frame_width/2. + d_hat_h);
    y2 = (y2_temp + frame_height/2. + d_hat_v);

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
      size = 300;
  }
  else if (nArgs == 7 || nArgs == 8) {
      w = 4; h = 2;
      size = 300;
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

robot_t rotate_robot(robot_t robot, double theta) {
    double x1 = robot.p1.x - max_dim/2.;
    double y1 = robot.p1.y - max_dim/2.;

    double x2 = robot.p2.x - max_dim/2.;
    double y2 = robot.p2.y - max_dim/2.;

    double x3 = robot.p3.x - max_dim/2.;
    double y3 = robot.p3.y - max_dim/2.;

    // applies the 2d rotation to the line, making it either horizontal or vertical
    double x1_temp = x1 * cos(-theta) - y1 * sin(-theta);
    double y1_temp = x1 * sin(-theta) + y1 * cos(-theta);

    double x2_temp = x2 * cos(-theta) - y2 * sin(-theta);
    double y2_temp = x2 * sin(-theta) + y2 * cos(-theta);

    double x3_temp = x3 * cos(-theta) - y3 * sin(-theta);
    double y3_temp = x3 * sin(-theta) + y3 * cos(-theta);

    // translates the image back and adds displacement
    x1 = x1_temp + max_dim/2.;
    y1 = y1_temp + max_dim/2.;

    x2 = x2_temp + max_dim/2.;
    y2 = y2_temp + max_dim/2.;

    x3 = x3_temp + max_dim/2.;
    y3 = y3_temp + max_dim/2.;

  robot_t robot_rot = {
    .p1     = cv::Point(x1, y1),
    .p2     = cv::Point(x2, y2),
    .p3     = cv::Point(x3, y3),
    .angle  = theta,
  };

  return robot_rot;
}

robot_t translate_robot(robot_t robot, double dx, double dy) {
  // applies the 2d rotation to the lines
  // dy negative as image progresses downwards
  robot_t robot_trans = {
    .p1     = cv::Point(robot.p1.x + dx, robot.p1.y - dy ),
    .p2     = cv::Point(robot.p2.x + dx, robot.p2.y - dy ),
    .p3     = cv::Point(robot.p3.x + dx, robot.p3.y - dy ),
    .angle  = robot.angle,
  };

  return robot_trans;
}

cv::Mat generate_global_frame(ibex::IntervalVector state, ibex::IntervalVector obs, ibex::IntervalVector box, std::vector<double> pose) {
  double state_1 = state[0].mid();
  double state_2 = state[1].mid();
  double state_3 = state[2].mid();

  // pose[0] is the timestamp
  double pose_1 = pose[1];
  double pose_2 = pose[2];
  double pose_3 = pose[3];

  double d_hat_h = obs[0].mid();
  double d_hat_v = obs[1].mid();
  double a_hat   = obs[2].mid();

//    d_hat_h = pose_1/tile_size - floor(pose_1/tile_size); //modulo(pose_1, tile_size);
//    d_hat_v = pose_2/tile_size - floor(pose_2/tile_size); //modulo(pose_2, tile_size);
//    a_hat = ((pose_3-M_PI/4)/(M_PI/2) - floor((pose_3-M_PI/4)/(M_PI/2))-0.5) * 2*M_PI/4;  // modulo without function as it is in radians

  double box_1 = box[0].mid();
  double box_2 = box[1].mid();
  double box_3 = box[2].mid();

  double b1_ub = box[0].ub();  // already added here dist_lines because only used for display directly
  double b1_lb = box[0].lb();
  double b2_ub = box[1].ub();
  double b2_lb = box[1].lb();

  double view_px_per_m = 350.0;     // pixels per meter for the visualization
  double view_dist_lines = tile_size * view_px_per_m;

  if (!base_global_frame_created) {
    int n_lines = 21;
    max_dim = view_dist_lines * (n_lines) + view_dist_lines/2.;  // largest dimension so that always show something inside the picture

    // center of the image, where tiles start with zero displacement
    double center_x = max_dim/2.;
    double center_y = max_dim/2.;

    // create a line every specified number of pixels, starting in a multiple from 0
    int pos_x = center_x - floor(n_lines/2.)*view_dist_lines;
    while (pos_x <= max_dim) {
      line_t ln = {
        .p1     = cv::Point(pos_x, -max_dim),
        .p2     = cv::Point(pos_x, max_dim),
        .side   = 1  // 0 horizontal, 1 vertical
      };
      base_global_frame_lines.push_back(ln);
      pos_x += view_dist_lines;
    }

    int pos_y = center_y - floor(n_lines/2.)*view_dist_lines;
    while (pos_y <= max_dim) {
      line_t ln = {
        .p1     = cv::Point(-max_dim, pos_y),
        .p2     = cv::Point(max_dim, pos_y),
        .side   = 0  // 0 horizontal, 1 vertical
      };
      base_global_frame_lines.push_back(ln);
      pos_y += view_dist_lines;
    }

    base_global_frame = cv::Mat::zeros(max_dim+10, max_dim+10, CV_8UC3);

    int count_lin = 1;
    for (line_t l : base_global_frame_lines) {
      count_lin += 1;

      Scalar color;
      if (l.side == 1) {
        if (abs(l.p1.x - center_x) <= 2)
          color = Scalar(0, 0, 255);
        else
          color = Scalar(0, 255, 0);
      } else {
        if (abs(l.p1.y - center_y) <= 2)
          color = Scalar(0, 0, 255);
        else
          color = Scalar(255, 0, 0);
      }

      line(base_global_frame, cv::Point(l.p1.x, l.p1.y), cv::Point(l.p2.x, l.p2.y), color, 1, LINE_AA);
    }

    base_robot = {
      .p1     = cv::Point(center_x - 20*0.7, center_y - 25*0.7),  // cv::Point(center_x - 20, center_y - 25),
      .p2     = cv::Point(center_x - 20*0.7, center_y + 25*0.7),  // cv::Point(center_x - 20, center_y + 25),
      .p3     = cv::Point(center_x + 40*0.7, center_y + 0),   // cv::Point(center_x + 40, center_y + 0),
      .angle  = 0,
    };

    base_global_frame_created = true;
  }

  cv::Mat global_frame = base_global_frame.clone();

  // draw observation representations
  robot_t robot_obs_1 = {
    .p1     = base_robot.p1,
    .p2     = base_robot.p2,
    .p3     = base_robot.p3,
    .angle  = base_robot.angle,
  };
  robot_obs_1 = rotate_robot(robot_obs_1, a_hat);  // rotate already centered at the origin
  robot_obs_1 = translate_robot(robot_obs_1, d_hat_h*view_px_per_m, d_hat_v*view_px_per_m);

  // yellow
  line(global_frame, robot_obs_1.p1, robot_obs_1.p2, Scalar(0, 255, 255), 1, LINE_AA);
  line(global_frame, robot_obs_1.p2, robot_obs_1.p3, Scalar(0, 255, 255), 1, LINE_AA);
  line(global_frame, robot_obs_1.p3, robot_obs_1.p1, Scalar(0, 255, 255), 1, LINE_AA);
  circle(global_frame, robot_obs_1.p1/3 + robot_obs_1.p2/3 + robot_obs_1.p3/3, 2, Scalar(0, 255, 255), 1);

  robot_t robot_obs_2 = {
    .p1     = base_robot.p1,
    .p2     = base_robot.p2,
    .p3     = base_robot.p3,
    .angle  = base_robot.angle,
  };
  robot_obs_2 = rotate_robot(robot_obs_2, a_hat);//+M_PI/2.);  // rotate already centered at the origin
  robot_obs_2 = translate_robot(robot_obs_2, d_hat_v*view_px_per_m, d_hat_h*view_px_per_m);

  // orange
  line(global_frame, robot_obs_2.p1, robot_obs_2.p2, Scalar(0, 69, 255), 1, LINE_AA);
  line(global_frame, robot_obs_2.p2, robot_obs_2.p3, Scalar(0, 69, 255), 1, LINE_AA);
  line(global_frame, robot_obs_2.p3, robot_obs_2.p1, Scalar(0, 69, 255), 1, LINE_AA);
  circle(global_frame, robot_obs_2.p1/3 + robot_obs_2.p2/3 + robot_obs_2.p3/3, 2, Scalar(0, 69, 255), 1);

  // draw state prediction
  robot_t robot_prediction = {
    .p1     = base_robot.p1,
    .p2     = base_robot.p2,
    .p3     = base_robot.p3,
    .angle  = base_robot.angle,
  };
  robot_prediction = rotate_robot(robot_prediction, state_3);  // rotate already centered at the origin
  robot_prediction = translate_robot(robot_prediction, state_1*view_px_per_m, state_2*view_px_per_m);  // translate according to state

  // dark green
  line(global_frame, robot_prediction.p1, robot_prediction.p2, Scalar(130, 200, 0), 1, LINE_AA);
  line(global_frame, robot_prediction.p2, robot_prediction.p3, Scalar(130, 200, 0), 1, LINE_AA);
  line(global_frame, robot_prediction.p3, robot_prediction.p1, Scalar(130, 200, 0), 1, LINE_AA);
  circle(global_frame, robot_prediction.p1/3 + robot_prediction.p2/3 + robot_prediction.p3/3, 2, Scalar(130, 200, 0), 1);

  // draw ground truth
  robot_t robot_gt= {
    .p1     = base_robot.p1,
    .p2     = base_robot.p2,
    .p3     = base_robot.p3,
    .angle  = base_robot.angle,
  };
  robot_gt = rotate_robot(robot_gt, pose_3);  // rotate already centered at the origin
  robot_gt = translate_robot(robot_gt, pose_1*view_px_per_m, pose_2*view_px_per_m);  // translate according to state

  // pink
  line(global_frame, robot_gt.p1, robot_gt.p2, Scalar(203, 192, 255), 1, LINE_AA);
  line(global_frame, robot_gt.p2, robot_gt.p3, Scalar(203, 192, 255), 1, LINE_AA);
  line(global_frame, robot_gt.p3, robot_gt.p1, Scalar(203, 192, 255), 1, LINE_AA);
  circle(global_frame, robot_gt.p1/3 + robot_gt.p2/3 + robot_gt.p3/3, 2, Scalar(203, 192, 255), 1);

  // draw contracted state
  robot_t robot_box = {
    .p1     = base_robot.p1,
    .p2     = base_robot.p2,
    .p3     = base_robot.p3,
    .angle  = base_robot.angle,
  };
  robot_box = rotate_robot(robot_box, box_3);  // rotate already centered at the origin
  robot_box = translate_robot(robot_box, box_1*view_px_per_m, box_2*view_px_per_m);  // translate according to state

  // light blue
  line(global_frame, robot_box.p1, robot_box.p2, Scalar(255, 255, 0), 1, LINE_AA);
  line(global_frame, robot_box.p2, robot_box.p3, Scalar(255, 255, 0), 1, LINE_AA);
  line(global_frame, robot_box.p3, robot_box.p1, Scalar(255, 255, 0), 1, LINE_AA);
  circle(global_frame, robot_box.p1/3 + robot_box.p2/3 + robot_box.p3/3, 2, Scalar(255, 255, 0), 1);

  line(global_frame, Point2f(max_dim/2.+b1_lb*view_px_per_m, max_dim/2.-b2_lb*view_px_per_m), Point2f(max_dim/2.+b1_ub*view_px_per_m, max_dim/2.-b2_lb*view_px_per_m), Scalar(255, 255, 0), 1, LINE_AA);
  line(global_frame, Point2f(max_dim/2.+b1_lb*view_px_per_m, max_dim/2.-b2_ub*view_px_per_m), Point2f(max_dim/2.+b1_ub*view_px_per_m, max_dim/2.-b2_ub*view_px_per_m), Scalar(255, 255, 0), 1, LINE_AA);
  line(global_frame, Point2f(max_dim/2.+b1_lb*view_px_per_m, max_dim/2.-b2_lb*view_px_per_m), Point2f(max_dim/2.+b1_lb*view_px_per_m, max_dim/2.-b2_ub*view_px_per_m), Scalar(255, 255, 0), 1, LINE_AA);
  line(global_frame, Point2f(max_dim/2.+b1_ub*view_px_per_m, max_dim/2.-b2_lb*view_px_per_m), Point2f(max_dim/2.+b1_ub*view_px_per_m, max_dim/2.-b2_ub*view_px_per_m), Scalar(255, 255, 0), 1, LINE_AA);

//  circle(global_frame, Point2f(max_dim, max_dim), 15, Scalar(255, 255, 255), 3);
  return global_frame;
}
