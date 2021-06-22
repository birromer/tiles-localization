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

#define ERROR_PRED      0.3
#define ERROR_OBS       0.3
#define ERROR_OBS_ANGLE 0.03

#define NUM_IMGS 13758

#define IMG_FOLDER "/home/birromer/ros/data_tiles/dataset_tiles/"
#define GT_FILE "/home/birromer/ros/data_tiles/gt.csv"
string sim_file = "/home/birromer/ros/data_tiles/test_sim.csv";
//#define IMG_FOLDER "/home/birromer/data_tiles_backup/dataset_tiles/"
//#define GT_FILE "/home/birromer/data_tiles_backup/gt.csv"
//string sim_file = "/home/birromer/data_tiles_backup/test_sim.csv";

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

cv::Mat generate_grid_1(int dist_lines, ibex::IntervalVector obs);
cv::Mat generate_grid_2(int dist_lines, ibex::IntervalVector obs);
cv::Mat generate_global_frame(int dist_lines, ibex::IntervalVector state, ibex::IntervalVector obs);

robot_t rotate_robot(robot_t robot, double theta);
robot_t translate_robot(robot_t robot, double dx, double dy);

void ShowManyImages(string title, int nArgs, ...);

// variables for the images from the observation
std::vector<line_t> base_grid_lines;
bool base_grid_created = false;

// variables for the global frame from the state and the observation
std::vector<line_t> base_global_frame_lines;
bool base_global_frame_created = false;
cv::Mat base_global_frame;
robot_t base_robot;

double prev_a_hat;  // a_hat of the previous iteration
int quart_state = 0;  // in which quarter of the plane is the current angle

double frame_width=0, frame_height=0;
int max_dim;

bool verbose = true;
bool interactive = false;
bool display_window = false;
bool intervals = false;

double dist_lines = 103.0;  //pixels between each pair of lines

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
    ("dist-lines", po::value<double>(&dist_lines), "distance between lines")
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
    sim_file = vm["output-file"].as<string>();
    printw("Output file specified by user: %c\n", sim_file);
  } else {
    printw("Output file default: %s\n", sim_file);
  }

  printw("Distance between lines is %f\n", dist_lines);

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
  TApplication rootapp("viz", &argc, argv);

  auto c1 = std::make_unique<TCanvas>("c1", "Equivalence equations");
  c1->SetWindowSize(1550, 700);

  auto f1 = std::make_unique<TGraph>(NUM_IMGS);
  f1->SetTitle("sin(pi*(y1-z1))");
  f1->GetXaxis()->SetTitle("Iteration");
  f1->GetYaxis()->SetTitle("Similarity score");
  f1->SetMinimum(-1);
  f1->SetMaximum(1);

  auto f2 = std::make_unique<TGraph>(NUM_IMGS);
  f2->SetTitle("sin(pi*(y2-z2))");
  f2->GetXaxis()->SetTitle("Iteration");
  f2->GetYaxis()->SetTitle("Similarity score");
  f2->SetMinimum(-1);
  f2->SetMaximum(1);

  auto f3 = std::make_unique<TGraph>(NUM_IMGS);
  f3->SetTitle("sin(y3-z3)");
  f3->GetXaxis()->SetTitle("Iteration");
  f3->GetYaxis()->SetTitle("Similarity score");
  f3->SetMinimum(-1);
  f3->SetMaximum(1);

  auto f4 = std::make_unique<TGraph>(NUM_IMGS);
  f4->SetTitle("sin(pi*(y1-z2))");
  f4->GetXaxis()->SetTitle("Iteration");
  f4->GetYaxis()->SetTitle("Similarity score");
  f4->SetMinimum(-1);
  f4->SetMaximum(1);

  auto f5 = std::make_unique<TGraph>(NUM_IMGS);
  f5->SetTitle("sin(pi*(y2-z1))");
  f5->GetXaxis()->SetTitle("Iteration");
  f5->GetYaxis()->SetTitle("Similarity score");
  f5->SetMinimum(-1);
  f5->SetMaximum(1);

  auto f6 = std::make_unique<TGraph>(NUM_IMGS);
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

  double pose_1, pose_2, pose_3;
  ibex::IntervalVector obs(3, ibex::Interval::ALL_REALS);    // observed parameters from the image
  ibex::IntervalVector state(3, ibex::Interval::ALL_REALS);  // working state from prediction (in this case gt)

  // --------- start file with testing data ----------- //
  ofstream file_sim(sim_file);

  if(!file_sim.is_open())
    throw std::runtime_error("Could not open SIM file");

  file_sim << "sim1_eq1" << "," << "sim1_eq2" << "," << "sim1_eq3" << "," << "sim2_eq1" << "," << "sim2_eq2" << "," << "sim2_eq3" << endl;
  // ------------------------------------------------ //

  // ----------- get ground truth data -------------- //
  ifstream file_gt(GT_FILE);

  if(!file_gt.is_open())
    throw std::runtime_error("Could not open GT file");

  std::string line_content, colname;

  std::getline(file_gt, line_content);  // skip line with column names

  while(getline(file_gt, line_content)) {
    // get the pose from the ground truth
    stringstream ss(line_content);  // stringstream of the current line
    vector<double> line_vals;
    double val;

    // extract each value
    while(ss >> val){
        line_vals.push_back(val);

        if(ss.peek() == ',')
          ss.ignore();  // ignore commas
    }

    vector<double> pose{line_vals[0], line_vals[1], line_vals[2]};
    sim_ground_truth.push_back(pose);
  }
  // ------------------------------------------------ //

  while(curr_img < NUM_IMGS) {
    printw("Processing image %d/%d\n", curr_img, NUM_IMGS);

    // 1. preprocessing
    // 1.1 read the image
    char cimg[1000];
    snprintf(cimg, 1000, "%s%06d.png", IMG_FOLDER, curr_img);
    string ref_filename(cimg);
    Mat in = imread(ref_filename);
    frame_height = in.size[0];
    frame_width = in.size[1];

    // 1.2 convert to greyscale for later computing borders
    Mat grey;
    cvtColor(in, grey, COLOR_BGR2GRAY);

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
    Mat grad;
//    Laplacian(skel, grad, CV_8U, 1, 1, 0, BORDER_DEFAULT);
    Laplacian(grey, grad, CV_8U, 1, 1, 0, BORDER_DEFAULT);

    // 1.4 detect edges, 50 and 255 as thresholds 1 and 2
    Mat edges;
    Canny(grad, edges, 50, 255, 3);

//    // 1.5 close and dilate lines for some noise removal
//    Mat morph;
//    int morph_elem = 0;
//    int morph_size = 0;
//    Mat element = getStructuringElement(morph_elem, Size(2*morph_size + 1, 2*morph_size+1), cv::Point(morph_size, morph_size));
//    morphologyEx(edges, edges, MORPH_CLOSE, element);
//    dilate(edges, morph, Mat(), cv::Point(-1,-1), 1, BORDER_CONSTANT, morphologyDefaultBorderValue());

    // 1.6 detect lines using the hough transform
    std::vector<Vec4i> detected_lines;
    HoughLinesP(edges, detected_lines, 1, CV_PI/180., 60, 120, 50);

    // 2.0 extract parameters from the angles of the lines from the hough transform, as said in luc's paper
    // this is done for ease of computation
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

      // 2.1.1 smallest radius of a circle with a point belonging to the line with origin in 0, being 0 corrected to the center of the image
      d = ((p2_x-p1_x)*(p1_y-frame_height/2) - (p1_x-frame_width/2)*(p2_y-p1_y)) / sqrt(pow(p2_x-p1_x,2) + pow(p2_y-p1_y,2));

      // 2.1.2 decimal distance, displacement between the lines
      dd = ((d/dist_lines + 0.5) - (floor(d/dist_lines) + 0.5)) - 0.5;
//      dd = (d/dist_lines - floor(d/dist_lines));

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
    }

    // 1.7.2 median of the components of the lines
    x_hat = median(lines, 3);
    y_hat = median(lines, 4);

    std::vector<line_t> lines_good;

    Mat src = Mat::zeros(Size(frame_width, frame_height), CV_8UC3);
    // 2.2 filter lines with bad orientation
    for (line_t l : lines) {
      if ((abs(x_hat - l.m_x) + abs(y_hat - l.m_y)) < 0.15) {
        line(src, l.p1, l.p2, Scalar(255, 0, 0), 3, LINE_AA);
        lines_good.push_back(l);

      } else {
        line(src, l.p1, l.p2, Scalar(0, 0, 255), 3, LINE_AA);

      }
    }

    x_hat = median(lines_good, 3);
    y_hat = median(lines_good, 4);

//    prev_a_hat = a_hat;
    a_hat = atan2(y_hat, x_hat) * 1/4;

//    if (a_hat > 0 && prev_a_hat < 0)
//      quart_state -= 1;
//    else if (a_hat < 0 && prev_a_hat > 0)
//      quart_state += 1;
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
//
//    printw("CURRENT QUART STATE = %d\n", quart_state);

    Mat rot = Mat::zeros(Size(frame_width , frame_height), CV_8UC3);
    if(lines_good.size() > MIN_GOOD_LINES) {
      printw("Found %d good lines\n", lines_good.size());
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
      double d_hat_h = median(bag_h, 6);
      double d_hat_v = median(bag_v, 6);

      obs = ibex::IntervalVector({
          {d_hat_h, d_hat_h},
          {d_hat_v, d_hat_v},
          {a_hat, a_hat}
      }).inflate(ERROR_OBS);

      obs[2] = ibex::Interval(a_hat, a_hat).inflate(ERROR_OBS_ANGLE);

    } else {
      printw("Not enough good lines (%d)\n", lines_good.size());
    }

    // 3 generate the representation of the observed parameters
    Mat view_param_1 = generate_grid_1(dist_lines, obs);
    Mat view_param_2 = generate_grid_2(dist_lines, obs);

    // 4 get the pose from the ground truth
    // access the vector where it is stored
    // TODO: test if really that pose and doesnt have to be scaled according to pixel distance for the comparisson to be true
    double pose_1 = sim_ground_truth[curr_img][0];// + 0.300971;
    double pose_2 = sim_ground_truth[curr_img][1];// - 0.26699;
    double pose_3 = sim_ground_truth[curr_img][2];

    state = ibex::IntervalVector({
        {pose_1, pose_1},
        {pose_2, pose_2},
        {pose_3, pose_3}
    }).inflate(ERROR_PRED);

    printw("\nPARAMETERS -> d_hat_h = %f | d_hat_v = %f | a_hat = %f\n", obs[0].mid(), obs[1].mid(), obs[2].mid());;
    printw("POSE       ->      x1 = %f |      x2 = %f |    x3 = %f\n", pose_1, pose_2, pose_3);

    // global frame with observed parameter and pose
    Mat view_global_frame = generate_global_frame(dist_lines, state, obs);

    // 5 equivalency equations
    // ground truth and parameters should have near 0 value in the equivalency equations
    double sim1_eq1 = sin(M_PI*(obs[0].mid()-pose_1));
    double sim1_eq2 = sin(M_PI*(obs[1].mid()-pose_2));
    double sim1_eq3 = sin(obs[2].mid()-pose_3);

    double sim2_eq1 = sin(M_PI*(obs[0].mid()-pose_2));
    double sim2_eq2 = sin(M_PI*(obs[1].mid()-pose_1));
    double sim2_eq3 = cos(obs[2].mid()-pose_3);

    file_sim << sim1_eq1 << "," << sim1_eq2 << "," << sim1_eq3 << "," << sim2_eq1 << "," << sim2_eq2 << "," << sim2_eq3 << endl;

    printw("\nEquivalence equations 1:\n  sin(pi*(y1-z1)) = %f\n  sin(pi*(y2-z2)) = %f\n  sin(y3-z3) = %f\n", sim1_eq1, sim1_eq2, sim1_eq3);
    printw("\nEquivalence equations 2:\n  sin(pi*(y1-z2)) = %f\n  sin(pi*(y2-z1)) = %f\n  cos(y3-z3) = %f\n", sim2_eq1, sim2_eq2, sim2_eq3);

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

    // display steps and global frame
    if(display_window) {
      ShowManyImages("steps", 5, in, src, rot, view_param_1, view_param_2);
      cv::imshow("global_frame", view_global_frame);
    }

    if (intervals) {
      ibex::IntervalVector box0(6, ibex::Interval::ALL_REALS);
      ibex::IntervalVector box1(6, ibex::Interval::ALL_REALS);

      box0[0] = state[0], box0[1] = state[1], box0[2] = state[2], box0[3] = obs[0], box0[4] = obs[1], box0[5] = obs[2];
      box1[0] = state[0], box1[1] = state[1], box1[2] = state[2], box1[3] = obs[0], box1[4] = obs[1], box1[5] = obs[2];

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
        printw("Could not contract the state (!!!).\n");
      }

      // draw ground truth
      vibes::drawVehicle(pose_1, pose_2, pose_3*180./M_PI, 0.3, "pink");  // draw ground truth

      // draw the predicted state (in this case is the ground truth, as we're testing only the parameters)
      vibes::drawBox(state.subvector(0, 1), "green");
      vibes::drawVehicle(state[0].mid(), state[1].mid(), (state[2].mid())*180./M_PI, 0.3, "green");

      // draw the contracted state
      vibes::drawBox(box.subvector(0, 1), "blue");
      vibes::drawVehicle(box[0].mid(), box[1].mid(), (box[2].mid())*180./M_PI, 0.3, "blue");
    }

    if (interactive) {
      kb_key = getch();

      if (kb_key == 104) {
        if (curr_img > 0) {
          curr_img -= 1;
          sim_test_data.pop_back();
        }
      } else if (kb_key == 108) {
        if (curr_img < NUM_IMGS)
          curr_img += 1;
      }
    } else {
      curr_img += 1;
    }

    clear();
  }  // end of loop for each image

  endwin();  // finish ncurses mode
  if (intervals)
    vibes::endDrawing();  // close vibes drawing
  printw("Exited with success\n");
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

cv::Mat generate_grid_1(int dist_lines, ibex::IntervalVector obs) {
  double d_hat_h = obs[0].mid() * dist_lines;  // parameters have to be scaled for being shown in pixels
  double d_hat_v = obs[1].mid() * dist_lines;
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
    double x1_temp = x1 * cos(a_hat) - y1 * sin(a_hat);// x1;//
    double y1_temp = x1 * sin(a_hat) + y1 * cos(a_hat);// y1;//

    double x2_temp = x2 * cos(a_hat) - y2 * sin(a_hat);// x2;//
    double y2_temp = x2 * sin(a_hat) + y2 * cos(a_hat);// y2;//

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
  double d_hat_h = obs[1].mid() * dist_lines;  // parameters have to be scaled to be shown in pixels
  double d_hat_v = obs[0].mid() * dist_lines;
  double a_hat   = obs[2].mid() + M_PI;

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
    double x1_temp = x1 * cos(a_hat) - y1 * sin(a_hat);
    double y1_temp = x1 * sin(a_hat) + y1 * cos(a_hat);

    double x2_temp = x2 * cos(a_hat) - y2 * sin(a_hat);
    double y2_temp = x2 * sin(a_hat) + y2 * cos(a_hat);

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
  robot_t robot_trans = {
    .p1     = cv::Point(robot.p1.x + dx, robot.p1.y - dy ),
    .p2     = cv::Point(robot.p2.x + dx, robot.p2.y - dy ),
    .p3     = cv::Point(robot.p3.x + dx, robot.p3.y - dy ),
    .angle  = robot.angle,
  };

  return robot_trans;
}

cv::Mat generate_global_frame(int dist_lines, ibex::IntervalVector state, ibex::IntervalVector obs) {
  double d_hat_h = obs[0].mid();
  double d_hat_v = obs[1].mid();
  double a_hat   = obs[2].mid();

  double state_1 = state[0].mid();
  double state_2 = state[1].mid();
  double state_3 = state[2].mid();

  if (!base_global_frame_created) {
    int n_lines = 11;
    max_dim = dist_lines * (n_lines) + dist_lines/2.;  // largest dimension so that always show something inside the picture

    // center of the image, where tiles start with zero displacement
    double center_x = max_dim/2.;
    double center_y = max_dim/2.;

    // create a line every specified number of pixels, starting in a multiple from 0
    int pos_x = center_x - (n_lines/2)*dist_lines;
    while (pos_x <= max_dim) {
      line_t ln = {
        .p1     = cv::Point(pos_x, -max_dim),
        .p2     = cv::Point(pos_x, max_dim),
        .side   = 1  // 0 horizontal, 1 vertical
      };
      base_global_frame_lines.push_back(ln);
      pos_x += dist_lines;
    }

    int pos_y = center_y - (n_lines/2)*dist_lines;
    while (pos_y <= max_dim) {
      line_t ln = {
        .p1     = cv::Point(-max_dim, pos_y),
        .p2     = cv::Point(max_dim, pos_y),
        .side   = 0  // 0 horizontal, 1 vertical
      };
      base_global_frame_lines.push_back(ln);
      pos_y += dist_lines;
    }

    base_global_frame = cv::Mat::zeros(max_dim+10, max_dim+10, CV_8UC3);

    int count_lin = 1;
    for (line_t l : base_global_frame_lines) {
      count_lin += 1;

      Scalar color;
      if (l.side == 1) {
        if (l.p1.x == center_x)
          color = Scalar(0, 0, 255);
        else
          color = Scalar(255, 0, 0);
      } else {
        if (l.p1.y == center_y)
          color = Scalar(0, 0, 255);
        else
          color = Scalar(0, 255, 0);
      }

      line(base_global_frame, cv::Point(l.p1.x, l.p1.y), cv::Point(l.p2.x, l.p2.y), color, 3, LINE_AA);
    }

    base_robot = {
      .p1     = cv::Point(center_x - 20, center_y - 25),
      .p2     = cv::Point(center_x - 20, center_y + 25),
      .p3     = cv::Point(center_x + 40, center_y + 0),
      .angle  = 0,
    };

    base_global_frame_created = true;
  }

  cv::Mat global_frame = base_global_frame.clone();

  robot_t robot_obs_1 = {
    .p1     = base_robot.p1,
    .p2     = base_robot.p2,
    .p3     = base_robot.p3,
    .angle  = base_robot.angle,
  };
  robot_obs_1 = rotate_robot(robot_obs_1, a_hat);  // rotate already centered at the origin
  robot_obs_1 = translate_robot(robot_obs_1, d_hat_h*dist_lines, d_hat_v*dist_lines);

  // yellow
  line(global_frame, robot_obs_1.p1, robot_obs_1.p2, Scalar(0, 255, 255), 1, LINE_AA);
  line(global_frame, robot_obs_1.p2, robot_obs_1.p3, Scalar(0, 255, 255), 1, LINE_AA);
  line(global_frame, robot_obs_1.p3, robot_obs_1.p1, Scalar(0, 255, 255), 1, LINE_AA);
  circle(global_frame, robot_obs_1.p1/3 + robot_obs_1.p2/3 + robot_obs_1.p3/3, 4, Scalar(0, 255, 255), 1);

  robot_t robot_obs_2 = {
    .p1     = base_robot.p1,
    .p2     = base_robot.p2,
    .p3     = base_robot.p3,
    .angle  = base_robot.angle,
  };
  robot_obs_2 = rotate_robot(robot_obs_2, a_hat+M_PI/2.);  // rotate already centered at the origin
  robot_obs_2 = translate_robot(robot_obs_2, d_hat_v*dist_lines, d_hat_h*dist_lines);

  // orange
  line(global_frame, robot_obs_2.p1, robot_obs_2.p2, Scalar(0, 69, 255), 1, LINE_AA);
  line(global_frame, robot_obs_2.p2, robot_obs_2.p3, Scalar(0, 69, 255), 1, LINE_AA);
  line(global_frame, robot_obs_2.p3, robot_obs_2.p1, Scalar(0, 69, 255), 1, LINE_AA);
  circle(global_frame, robot_obs_2.p1/3 + robot_obs_2.p2/3 + robot_obs_2.p3/3, 4, Scalar(0, 69, 255), 1);

  robot_t robot_state = {
    .p1     = base_robot.p1,
    .p2     = base_robot.p2,
    .p3     = base_robot.p3,
    .angle  = base_robot.angle,
  };
  robot_state = rotate_robot(robot_state, state_3);  // rotate already centered at the origin
  robot_state = translate_robot(robot_state, state_1*dist_lines, state_2*dist_lines);  // translate according to state

  // light blue
  line(global_frame, robot_state.p1, robot_state.p2, Scalar(255, 255, 0), 1, LINE_AA);
  line(global_frame, robot_state.p2, robot_state.p3, Scalar(255, 255, 0), 1, LINE_AA);
  line(global_frame, robot_state.p3, robot_state.p1, Scalar(255, 255, 0), 1, LINE_AA);
  circle(global_frame, robot_state.p1/3 + robot_state.p2/3 + robot_state.p3/3, 4, Scalar(255, 255, 0), 1);

  circle(global_frame, Point2f(max_dim, max_dim), 15, Scalar(255, 255, 255), 3);

  return global_frame;
}
