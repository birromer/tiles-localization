#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <ibex.h>
#include <codac.h>
#include <codac-rob.h>
#include <math.h>
#include <stdarg.h>

using namespace cv;
using namespace std;

#define MIN_GOOD_LINES 5

#define ERROR_PRED      0.1
#define ERROR_OBS       0.3
#define ERROR_OBS_ANGLE 0.1

#define NUM_IMGS 10
#define IMG_FOLDER "/home/birromer/ros/dataset_tiles/"

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

ibex::IntervalVector obs(3, ibex::Interval::ALL_REALS);  // observed parameters from the image

std::vector<line_t> base_grid_lines;
bool base_grid_created = false;

bool display_window;
double frame_width=0, frame_height=0;

const int dist_lines = 103.0;  //pixels between each pair of lines

int main(int argc, char **argv) {

  int curr_img = 0;
  bool verbose = false;

  for(int curr_img = 0; curr_img <= n_img; curr_img++) {
    cout << "Processing image " << curr_img << "/" << NUM_IMGS << endl;

    // 1. Preprocessing
    // 1.1 Read the image
    char cimg[1000];
    snprintf(cimg, 1000, "%s%06d.png", imgFold, curr_imag);
    string ref_filename(cimg);
    Mat ref_img = imread(ref_filename);
    Size size(ref_img.cols, ref_img.rows);

  }


//  Mat view_param_1 = generate_grid_1(dist_lines, obs);
}
