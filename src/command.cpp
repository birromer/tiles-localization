/*
** This node is responsible for creating the waypoint to be followed by the robot.
** It may be replaced by an external command of the waypoint.
**
** Subscribers:
**   - none
**
** Publishers:
**   - geometry_msgs::PoseStamped waypoint // the target position
 */

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

  ros::init(argc, argv, "waypoint_node");
  ros::NodeHandle n;

  ros::Publisher pub_cons = n.advertise<geometry_msgs::PoseStamped>("waypoint", 1000);
  ros::Rate loop_rate(10);
  float t_start = ros::Time::now().toSec();

  while (ros::ok()){
    // sends message with desired position
    geometry_msgs::PoseStamped msg;

    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = "robot waypoint";

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
    ROS_INFO("[COMMAND] Sent waypoint -> x1: [%f] | x2: [%f] | x3: [%]", point.x, point.y, d);

    ros::spinOnce();

    loop_rate.sleep();

  }

return 0;
}
