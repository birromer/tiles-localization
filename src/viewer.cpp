#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
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
#include <tubex-3rd.h>

using namespace cv;
using namespace std;
using namespace ibex;
using namespace tubex;

void consigneCallback(const geometry_msgs::PoseStamped::ConstPtr& msg){
	float cons_x, cons_y, cons_cap;
	cons_x = msg->pose.position.x;
	cons_y = msg->pose.position.y;
 	cons_cap = tf::getYaw(msg->pose.orientation);

	vibes::drawVehicle(cons_x, cons_y, cons_cap*180./M_PI, 0.2, "green");
}

void stateCallback(const geometry_msgs::PoseStamped::ConstPtr& msg){
	float x, y, c;
	x = msg->pose.position.x;
	y = msg->pose.position.y;
 	c = tf::getYaw(msg->pose.orientation);

	vibes::drawVehicle(x, y, c*180./M_PI, 0.4, "blue");

}

void poseCallback(const geometry_msgs::Pose& msg){
	float x, y, c;
	x = msg.position.x;
	y = msg.position.y;
 	c = tf::getYaw(msg.orientation);

	vibes::drawVehicle(x, y, c*180./M_PI, 0.3, "red");
}

int main(int argc, char **argv){
	vibes::beginDrawing();
	VIBesFigMap fig_map("Map");
	vibes::setFigureProperties("Map",vibesParams("x", 10, "y", -10, "width", 100, "height", 100));
	vibes::axisLimits(-10, 10, -10, 10, "Map");

    fig_map.show();

	ros::init(argc, argv, "Viewer_node");

	ros::NodeHandle n;

	ros::Subscriber sub_consigne = n.subscribe("consigne", 1000, consigneCallback);
	ros::Subscriber sub_state = n.subscribe("state_estimated", 1000, stateCallback);
	ros::Subscriber sub_pose = n.subscribe("pose", 1000, poseCallback);

	ros::spin();

	vibes::endDrawing();
	return 0;
}