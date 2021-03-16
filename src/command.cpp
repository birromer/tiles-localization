#include "ros/ros.h"

#include <cmath>

#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Quaternion.h"
#include "geometry_msgs/PoseStamped.h"
#include "tf/tf.h"

#include <tf2/LinearMath/Quaternion.h>

using namespace std;

int main(int argc, char **argv)
{

  float lx = 10;
  float ly = 10;
  float omega = 0.01;
  double t = 0;

  ros::init(argc, argv, "Consigne_node");
  ros::NodeHandle n;

  ros::Publisher pub_cons = n.advertise<geometry_msgs::PoseStamped>("consigne", 1000);
  ros::Rate loop_rate(10);
  float t_start = ros::Time::now().toSec();

  while (ros::ok()){
    //envoi de la position estimé du robot
    geometry_msgs::PoseStamped msg;

    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = "robot consigne";

    //écriture du message
    geometry_msgs::Pose pose;
    geometry_msgs::Point point;

    //position

    //lissajou
    t = ros::Time::now().toSec()-t_start;

    point.x = lx*cos(omega*t + M_PI/2.);
    point.y = ly*sin(2*(omega*t + M_PI/2.));
    point.z = 0;

    //orientation
    tf::Quaternion q;
    float dx = -omega*lx*sin(omega*t + M_PI/2.);
    float dy = 2*ly*omega*cos(2*(omega*t + M_PI/2.));
    double d = atan2(dy, dx); //d = cap

    if(dx < 0){
        //d+=M_PI;
    }

    q.setRPY(0, 0, d);
    tf::quaternionTFToMsg(q, pose.orientation);

    pose.position = point;
    msg.pose = pose;

    pub_cons.publish(msg);

    ros::spinOnce();

    loop_rate.sleep();

  }

return 0;
}
