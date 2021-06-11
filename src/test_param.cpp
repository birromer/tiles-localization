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

#define MIN_GOOD_LINES 5

#define ERROR_PRED      0.1
#define ERROR_OBS       0.3
#define ERROR_OBS_ANGLE 0.1

#define NUM_IMGS 6697
#define IMG_FOLDER "/home/birromer/ros/data_tiles/dataset_tiles/"
#define GT_FILE "/home/birromer/ros/data_tiles/gt.csv"
#define SIM_FILE "/home/birromer/ros/data_tiles/test_sim.csv"

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

cv::Mat generate_grid_1(int dist_lines, ibex::IntervalVector obs);
cv::Mat generate_grid_2(int dist_lines, ibex::IntervalVector obs);
ibex::IntervalVector get_obs(cv::Mat image);

std::vector<line_t> base_grid_lines;
bool base_grid_created = false;

double frame_width=0, frame_height=0;

bool verbose = true;
bool display_window = true;

const int dist_lines = 103.0;  //pixels between each pair of lines

int main(int argc, char **argv) {
  // ------------------ ROOT setup ------------------ //
  TApplication rootapp("viz", &argc, argv);

  auto c1 = std::make_unique<TCanvas>("c1", "Equivalence equations");
  c1->SetWindowSize(1550, 700);

  // create the spectrogram
  auto f1 = std::make_unique<TGraph>(NUM_IMGS);
  f1->SetTitle("Set of equations 1");
  f1->GetXaxis()->SetTitle("Iteration");
  f1->GetYaxis()->SetTitle("Similarity score");
  f1->SetMinimum(-1);
  f1->SetMaximum(1);

  // create the waveform plot
  auto f2 = std::make_unique<TGraph>(NUM_IMGS);
  f2->SetTitle("Set of equations 2");
  f2->GetXaxis()->SetTitle("Iteration");
  f2->GetYaxis()->SetTitle("Similarity score");
  f2->SetMinimum(-1);
  f2->SetMaximum(1);

  // divide the canvas into two vertical sub-canvas
  c1->Divide(1, 2);

  // "Register" the plots for each canvas slot
  c1->cd(1); // Set current canvas to canvas 1 (yes, 1 based indexing)
  f1->Draw();
  c1->cd(2); // Set current canvas to canvas 2
  f2->Draw();
  // ------------------------------------------------ //

  int curr_img = 0;

  vector<vector<double>> sim_test_data;

  if(display_window) {
    cv::namedWindow("camera");
    cv::namedWindow("rotated");
    cv::namedWindow("lines");
    cv::namedWindow("view_param_1");
    cv::namedWindow("view_param_2");
    cv::startWindowThread();
  }

  double pose_1, pose_2, pose_3;
  ibex::IntervalVector obs(3, ibex::Interval::ALL_REALS);  // observed parameters from the image
//  ibex::IntervalVector pose(3, ibex::Interval::ALL_REALS);  // observed parameters from the image

  ifstream file_gt(GT_FILE);
  ofstream file_sim(SIM_FILE);
  file_sim << "sim1_eq1" << "," << "sim1_eq2" << "," << "sim1_eq3" << "," << "sim2_eq1" << "," << "sim2_eq2" << "," << "sim2_eq3" << endl;

  if(!file_gt.is_open())
    throw std::runtime_error("Could not open GT file");

  if(!file_sim.is_open())
    throw std::runtime_error("Could not open SIM file");

  std::string line_content, colname;

  std::getline(file_gt, line_content);  // skip line with column names

  while(getline(file_gt, line_content)) {
    cout << "Processing image " << curr_img << "/" << NUM_IMGS << endl;

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

    // 1.7.2 median of the components of the lines
    x_hat = median(lines, 3);
    y_hat = median(lines, 4);

    std::vector<line_t> lines_good;

    Mat src = Mat::zeros(Size(frame_width, frame_height), CV_8UC3);
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

    Mat rot = Mat::zeros(Size(frame_width , frame_height), CV_8UC3);
    if(lines_good.size() > MIN_GOOD_LINES) {
      cout << "Found " << lines_good.size() << " good lines" << endl;
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

      cout << "PARAMETERS -> d_hat_h = " << d_hat_h << " | d_hat_v = " << d_hat_v << " | a_hat = " << a_hat << endl;

      obs = ibex::IntervalVector({
          {d_hat_h, d_hat_h},
          {d_hat_v, d_hat_v},
          {a_hat, a_hat}
      }).inflate(ERROR_OBS);

      obs[2] = ibex::Interval(a_hat, a_hat).inflate(ERROR_OBS_ANGLE);

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
    stringstream ss(line_content);  // stringstream of the current line
    vector<double> line_vals;
    double val;

    // extract each value
    while(ss >> val){
        line_vals.push_back(val);

        if(ss.peek() == ',')
          ss.ignore();  // ignore commas
    }

    pose_1 = line_vals[0];
    pose_2 = line_vals[1];
    pose_3 = line_vals[2];

    // 5 equivalency equations

    // ground truth and parameters should have near 0 value in the equivalency equations
    double sim1_eq1 = sin(M_PI*(obs[0].mid()-pose_1));
    double sim1_eq2 = sin(M_PI*(obs[1].mid()-pose_2));
    double sim1_eq3 = sin(obs[2].mid()-pose_3);

    double sim2_eq1 = sin(M_PI*(obs[0].mid()-pose_2));
    double sim2_eq2 = sin(M_PI*(obs[1].mid()-pose_1));
    double sim2_eq3 = cos(obs[2].mid()-pose_3);

    file_sim << sim1_eq1 << "," << sim1_eq2 << "," << sim1_eq3 << "," << sim2_eq1 << "," << sim2_eq2 << "," << sim2_eq3 << endl;

    cout << "Equivalence equations 1:\nsin(pi*(y1-z1)) = " << sim1_eq1 << "\nsin(pi*(y2-z2)) = " << sim1_eq2 << "\nsin(y2-z2) = " << sim1_eq3 << endl;
    cout << "Equivalence equations 2:\nsin(pi*(y1-z2)) = " << sim2_eq1 << "\nsin(pi*(y2-z1)) = " << sim2_eq2 << "\ncos(y2-z1) = " << sim2_eq3 << endl;

    vector<double> s{sim1_eq1, sim1_eq2, sim1_eq3, sim2_eq1, sim2_eq2, sim2_eq3}
    sim_test_data.push_back(s);

    curr_img += 1;

    // redraw graph
    for (int i=0; i < sim_test_data.size(); i++) {
      f1->SetPoint(i, sim_test_data[i][0]);
    }

    // notify ROOT that the plots have been modified and needs update
    c1->cd(1);
    c1->Update();
    c1->Pad()->Draw();
    c1->cd(2);
    c1->Update();
    c1->Pad()->Draw();

  }  // end of loop for each image
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

cv::Mat generate_grid_1(int dist_lines, ibex::IntervalVector obs) {
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
