#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <vector>
#include <cmath>

#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Quaternion.h"
#include "geometry_msgs/PoseStamped.h"
#include "tf/tf.h"

#include <tf2/LinearMath/Quaternion.h>

#include <ibex.h>
#include <tubex.h>
#include <tubex-rob.h>
//#include <tubex-3rd.h>

using namespace cv;
using namespace std;
using namespace ibex;
using namespace tubex;


ros::Publisher chatter_state;
bool display_window;

//fonctions utiles
struct line_struct{
  Point2f p1;
  Point2f p2;
  float angle;
};

double sawtooth(double x){
  return 2.*atan(tan(x/2.));
}

float median(std::vector<float> scores){
  //https://stackoverflow.com/questions/2114797/compute-median-of-values-stored-in-vector-c
  size_t size = scores.size();

  if (size == 0)
  {
    return 0;  // Undefined, really.
  }
  else
  {
    sort(scores.begin(), scores.end());
    if (size % 2 == 0)
    {
      return (scores[size / 2 - 1] + scores[size / 2]) / 2;
    }
    else
    {
      return scores[size / 2];
    }
  }
}

float mean(std::vector<float> v){
  //assert(v.size() >0);
  float m = 0;

  for(int i=0; i<v.size(); i++){
    m+=v[i];
  }
  return m/v.size();
}

float modulo(float a, float b){
  float r = a/b - floor(a/b);
  if(r<0){
    r+=1;
  }
  return r*b;
}

int sign(float x){
  if(x < 0){
    return -1;
  }
  return 1;
}

//variables
IntervalVector state(3, Interval::ALL_REALS); // état estimé du robot

const float pix = 103;//99.3;//107.; //pixels entre chaque lignes

//taille du carrelage en mètre
float size_carrelage_x = 2.025/12.;
float size_carrelage_y = 2.025/12.;

float percent = 0.9; //de combien max on estime qu'on aura bougé (là, de moins d'un carreau donc le calcul est possible)

float frame_height, frame_width;

const float scale_pixel = 1./pix;

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

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    cv::Mat in = cv_bridge::toCvShare(msg, "bgr8")->image;
    cv::Mat out;
    cv::flip(in, out, 1); //L'image doit être retournée en raison de l'utilisation de la caméra de VREP

    if(display_window){
      cv::imshow("View base", out);
    }

    cv::Mat grey;
    cv::Mat bin;
    cvtColor(out, grey, CV_BGR2GRAY);  // put in grayscale

    frame_width = out.size[1];
    frame_height = out.size[0];
    Mat src = Mat::zeros(Size(frame_width, frame_height), CV_8UC3);

    // Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    // Gradient X
    Sobel(grey, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );

    // Gradient Y
    Sobel(grey, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    // Total Gradient (approximate)
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    Mat edges;
    Canny(grad, edges, 50, 255,3);

//    dilate(edges, edges, Mat(), cv::Point(-1,-1), 1, BORDER_CONSTANT, morphologyDefaultBorderValue());
    int morph_elem = 0;
    int morph_size = 0;
    Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );
    morphologyEx(edges, edges, MORPH_CLOSE, element);

    dilate(edges, edges, Mat(), cv::Point(-1,-1), 1, BORDER_CONSTANT, morphologyDefaultBorderValue());

    std::vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI/180., 60, 120, 50);

    std::vector<line_struct> lines_points;
    std::vector<float> lines_angles;

    float x1,x2,y1,y2;
    for(int i=0; i<lines.size(); i++) {

      cv::Vec4i l = lines[i];
      x1 = l[0],y1 = l[1],x2 = l[2],y2 = l[3];
      float angle_line = atan2(y2-y1,x2-x1);
      line(src, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), Scalar(255, 0, 0), 3, LINE_AA);
      line_struct ln;
      ln.p1 = cv::Point(l[0], l[1]);
      ln.p2 = cv::Point(l[2], l[3]);
      ln.angle = angle_line;
      lines_points.push_back(ln);

      float angle4 = modulo(angle_line+M_PI/4., M_PI/2.)-M_PI/4.;
      lines_angles.push_back(angle4);
    }

    float median_angle = median(lines_angles);
    alpha_median = median_angle;
    std::vector<Vec4i> lines_good;

    float anglegle;
    for (int i=0; i<lines_points.size(); i++) {

      anglegle = lines_angles[i];
      if(sawtooth(anglegle-median_angle) < 0.1)
      {
        line(src, lines_points[i].p1, lines_points[i].p2, Scalar(255, 0, 0), 3, LINE_AA);
        lines_good.push_back(lines[i]);

      }else {
         cout << "prob lignes : " << lines_angles[i] << " | " << sawtooth(lines_angles[i]-median_angle) << endl;
         line(src, lines_points[i].p1, lines_points[i].p2, Scalar(0, 255, 0), 3, LINE_AA);
        }
      }

      if(lines_good.size() > 5){
        cout << "il y a : " << lines_good.size() << " lignes" << endl;

        calc_new_pos(lines_good);

      } else {
        cout << "Pas assez de lignes (" << lines_good.size() << ")" << endl;
      }

      cv::imshow("grey",grey);
      cv::imshow("Sobel",grad);
      cv::imshow("Canny",edges);
      cv::imshow("view",src);

      //envoi de la position estimé du robot
      geometry_msgs::PoseStamped msg;

      msg.header.stamp = ros::Time::now();
      msg.header.frame_id = "robot";

      //écriture du message
      geometry_msgs::Pose pose;
      geometry_msgs::Point point;

      //position
      point.x = state[0].mid();
      point.y = state[1].mid();
      point.z = 0;

      //orientation
      tf::Quaternion q;
      const double d = state[2].mid(); //d = cap
      q.setRPY(0, 0, d);
      tf::quaternionTFToMsg(q, pose.orientation);

      pose.position = point;
      msg.pose = pose;

      chatter_state.publish(msg);


    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

int main(int argc, char **argv)
{
  //init pos
  state[0] = Interval(0., 0.);
  state[1] = Interval(0., 0.);
  state[2] = Interval(0., 0.);
//state[2] = Interval(M_PI/2., M_PI/2.);
//state[2] = Interval(M_PI, M_PI);
//state[2] = Interval(-M_PI/2., -M_PI/2.);
//state[2] = Interval(20.*M_PI/180., 20.*M_PI/180.);

  last_alpha_median = state[2].mid();
  quart = floor(state[2].mid()/(M_PI/2.));
  cout << "quart start : " << quart << endl;

  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;
  ros::NodeHandle nh_("~");
  display_window =  nh_.param<bool>("display_window", true);

  if(display_window)
  {
    cv::namedWindow("view");
    cv::namedWindow("View base");
    cv::namedWindow("rot");
    cv::namedWindow("grey");
    cv::namedWindow("Sobel");
    cv::namedWindow("Canny");

    cv::startWindowThread();
  }

  vibes::beginDrawing();
  VIBesFigMap fig_map("MapIntervals");
  vibes::setFigureProperties("MapIntervals",vibesParams("x", 10, "y", -10, "width", 100, "height", 100));
  vibes::axisLimits(-10, 10, -10, 10, "MapIntervals");

  fig_map.show();

  chatter_state = nh.advertise<geometry_msgs::PoseStamped>("state_estimated", 1000);
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("image", 1, imageCallback);
  ros::spin();
  //cv::destroyWindow("view");
  return 0;
}
