#include "ros/ros.h"
#include "std_msgs/Float32.h"

#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Quaternion.h"
#include "geometry_msgs/PoseStamped.h"
#include "tf/tf.h"

ros::Publisher pub_cmd_l, pub_cmd_r;


float currentSpeed = 150.;
int maxSpeed = 300; //on dis que consigne entre 0 et 255, que entier

float cons_x=0, cons_y=0, cons_cap=0;

double sawtooth(double x){
	return 2.*atan(tan(x/2.));
}

float max(float a, float b){
	if(a > b){
		return a;
	}
	return b;
}

float min(float a, float b){
	return -max(-a, -b);
}

float sign(float a){
	if(a < 0){
		return -1;

	}
	return 1;
}

void consigneCallback(const geometry_msgs::PoseStamped::ConstPtr& msg){
	cons_x = msg->pose.position.x;
	cons_y = msg->pose.position.y;
 	cons_cap = tf::getYaw(msg->pose.orientation);
}
float last_L = 0;
float f = 0;

void stateCallback(const geometry_msgs::PoseStamped::ConstPtr& msg){
	float X, Y, C;
	std_msgs::Float32 msg_r, msg_l;

	X = msg->pose.position.x;
	Y = msg->pose.position.y;
 	C = tf::getYaw(msg->pose.orientation);

 	float L = cos(cons_cap)*(cons_x-X) + sin(cons_cap)*(cons_y-Y);
 	float dist = sqrt(pow(cons_x-X, 2) + pow(cons_y-Y, 2));
 	//currentSpeed += sqrt(abs(L))*sign(L)/2.;
 	if(dist > 1.){
 		currentSpeed = maxSpeed*0.90;
 	}else{
 		currentSpeed = 20*dist;
	 	currentSpeed = max(currentSpeed, 100);
	 	f += 1.*(L-last_L);
	 	currentSpeed += f;
 	}
 	
 	last_L = L;

 	currentSpeed = max(min(currentSpeed, maxSpeed), 0);

 	float angle_avec_la_consigne = atan2(cons_y-Y, cons_x-X);

 	float c2 = (0.1*cons_cap+0.9*angle_avec_la_consigne);
 	//float c2 = atan2(Y-cons_y, X-cons_x);

 	//float cmd_r = (0.5 - sawtooth(C-c2)/M_PI)*round(currentSpeed);
 	//float cmd_l = (0.5 + sawtooth(C-c2)/M_PI)*round(currentSpeed);
 	float max_diff = 9.*currentSpeed/10.*2.;
 	float diff = sawtooth(c2-C)/M_PI*max_diff;

 	float cmd_l = currentSpeed-diff;
 	float cmd_r = currentSpeed+diff;

 	cmd_r = max(min(cmd_r, maxSpeed), -0);
 	cmd_l = max(min(cmd_l, maxSpeed), -0);

 	ROS_WARN("L : %f | currentSpeed : %f | C : %f", L, currentSpeed, C*180/M_PI);
 	ROS_WARN("dist : %f", dist);

 	ROS_WARN("cmd_l : %f | cmd_r : %f ", cmd_l, cmd_r);
 	ROS_WARN("cons_cap : %f | angle : %f | c2 : %f ", cons_cap*180/M_PI, angle_avec_la_consigne*180/M_PI, c2*180/M_PI);
 	ROS_WARN("sawtooth : %f", sawtooth(C-c2)/M_PI);
 	ROS_WARN(" ");

 	cmd_l /= 100;
 	cmd_r /= 100;

 	msg_l.data = cmd_l;
 	msg_r.data = cmd_r;

	pub_cmd_l.publish(msg_l);
	pub_cmd_r.publish(msg_r);

}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "Control_node");

	ros::NodeHandle n;
	pub_cmd_l = n.advertise<std_msgs::Float32>("cmd_ul", 1000);
	pub_cmd_r = n.advertise<std_msgs::Float32>("cmd_ur", 1000);

	ros::Subscriber sub_cons = n.subscribe("consigne", 1000, consigneCallback);
	ros::Subscriber sub_state = n.subscribe("state_estimated", 1000, stateCallback);

	ros::spin();
	return 0;
}