#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <vector>
#include <cmath>
#include "line_follow/Message_Line.h"


#include <ibex.h>
#include <tubex.h>
#include <tubex-rob.h>
#include <tubex-3rd.h>

using namespace cv;
using namespace std;
using namespace ibex;
using namespace tubex;


ros::Publisher chatter_line;
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
IntervalVector State(3, Interval::ALL_REALS); // état estimé du robot


const float pix = 107.; //pixels entre chaque lignes

	//taille du carrelage en mètre
float size_carrelage_x = 1/6.;
float size_carrelage_y = 1/6.;

float percent = 0.9; //de combien max on estime qu'on aura bougé (là, de moins d'un carreau donc le calcul est possible)

float frame_height, frame_width;

const float scale_pixel = 1./pix;

float alpha_median;
int quart;
float last_alpha_median = 0;
int nn = 0;
bool ff = false;

int nx =0;

//variable sobel
int scale = 1;
int delta = 0;
int ddepth = CV_16S;


//traitement d'image pour détecter les lignes
Mat src, src_, src_gray;
Mat grad;


void calc_new_pos(std::vector<Vec4i> lines){

	//cout << "state début : " << State << " | " << State[2]*180/M_PI << endl;

	//conversion from cartesian to polar form :
	float x1,x2,y1,y2;

	vector<float> Msn, Mew;

	alpha_median = alpha_median;// /4.;
	//cout << endl;
	cout << "alpha_median : " << alpha_median*180/M_PI << endl;

	//int quart = floor((State[2].mid()+M_PI/4.)/(M_PI/2.));

	//à la limite, on a le quart qui fluctue donc on veut éviter ça 
	if(nn ==0){
		if(abs(abs(alpha_median)-M_PI/4.)<0.1){
			if(alpha_median >= 0 & last_alpha_median < 0){
				cout << "####################################################" << endl << "Changement de quart : " << quart << endl << "###############################################################" << endl << endl;
				if(ff){
					//quart -=1;

				}else{
					ff = true;
				}
				quart -=1;
				/*
				if(quart == 0){
					quart = 1;
				}else{
					quart = 0;
				}
				*/
				nn = 1;
			}else if(alpha_median <= 0 & last_alpha_median > 0){
				cout << "####################################################" << endl << "Changement de quart : " << quart << endl << "###############################################################" << endl << endl;
				/*
				if(quart == 0){
					quart = 1;
				}else{
					quart = 0;
				}
				*/
				quart +=1;
				ff = false;
				nn = 1;
			}
		}
	}else{
		nn+=1;
	}
	if(nn == 100){
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

	    //bool flag = (modulo(State[2].mid()+M_PI/2, M_PI)-M_PI/2.) < 0;
	    //bool flag = sawtooth(alpha-last_alpha_median) > 0.2;

	    //float angle = (fmod(alpha+M_PI/2., M_PI/2.)-M_PI/2.);
	    float angle = modulo(alpha+M_PI/4., M_PI/2)-M_PI/4.;
	    //angle += (modulo(quart,2))*M_PI/2;
	    angle += quart*M_PI/2.;
	    //cout << "alpha lignes : " << alpha*180/M_PI << " | " << angle*180/M_PI << " | " << alpha_median*180/M_PI<< endl;

    	double s = sin(-angle);
		double c = cos(-angle);
		//cout << s << " " << c << " " << endl;

		float x1b=x1, y1b=y1, x2b=x2, y2b=y2;
		
		x1 = x1b*c - y1b*s, 
		y1 = +x1b*s + y1b*c;

		x2 = x2b*c - y2b*s, 
		y2 = x2b*s + y2b*c;
		
		//translation pour l'affichage
		x1+=frame_width/2., y1+=frame_height/2., x2+=frame_width/2., y2+=frame_height/2.;

		float alpha2 = atan2(y2-y1,x2-x1);
		float x11=x1, y11=y1, x22=x2, y22=y2;
	    //cout << "l : " << l[0] << " | " << l[1] << " | " << l[2] << " | " << l[3] << endl;
	    //cout << "x1 y1 x2 y2 : " << x1 << " | " << y1<< " | " << x2 << " | " << y2 << endl;

		//calcul pour medx et medy
		x1=((float)l[0]-frame_width/2.)*scale_pixel, y1=((float)l[1]-frame_height/2.)*scale_pixel; 
		x2=((float)l[2]-frame_width/2.)*scale_pixel, y2=((float)l[3]-frame_height/2.)*scale_pixel;
	    //cout << "x1 y1 x2 y2 : " << x1 << " | " << y1<< " | " << x2 << " | " << y2 << endl;

	    float d = ((x2-x1)*(y1)-(x1)*(y2-y1)) / sqrt(pow(x2-x1, 2)+pow(y2-y1, 2));
	    //cout << "d : " << d << endl;
	    //cout << "line : " << alpha << "|" << alpha_median << " | " << d << " | " << d2 << " | " << d3 << " | " << l << endl;
	    //cout << "X.append([" << x1 << ", " << x2 << "])" << endl;
	    //cout << "Y.append([" << y1 << ", " << y2 << "])" << endl;
	    //cout << endl;

		//vibes::selectFigure("Hough");
		//vibes::drawCircle(100*alpha, d, 2, "green[green]");

		float val;
		val = (d+0.5)-floor(d+0.5)-0.5;

	   	if(abs(cos(alpha2)) < 0.2){
			//vibes::drawCircle(val, 0.05*Msn.size(), 0.02, "blue");
			//vibes::drawCircle(d, 0.05*Msn.size()+20, 0.02, "blue");
			line(rot, cv::Point(x11, y11), cv::Point(x22, y22), Scalar(255, 255, 255), 1, CV_AA);

	    	Msn.push_back(val);

	    }else if(abs(sin(alpha2)) < 0.2){

			//vibes::drawCircle(val, 0.05*Mew.size()+10, 0.02, "green");
			//vibes::drawCircle(d, 0.05*Msn.size()+30, 0.02, "green");
			line(rot, cv::Point(x11, y11), cv::Point(x22, y22), Scalar(0, 0, 255), 1, CV_AA);

	    	Mew.push_back(val);

	    }else {
	    	//cout << "euh ..." << alpha*180./M_PI << " | " << alpha_median*180./M_PI << " | " << sin(alpha-alpha_median) << " | " << cos(alpha-alpha_median) << endl;
	    }

	    /*
    	vibes::selectFigure("Map");
    	std::vector<double> X0, Y0;
    	
    	
    	X0.push_back(x1);
    	X0.push_back(x2);
    	Y0.push_back(y1);
    	Y0.push_back(y2);
		*/
    	//vibes::drawLine(X0, Y0);
	} 
	//cout << "fin de la rotation" << endl;

	imshow("rot", rot);


	alpha_median = alpha_median + quart*M_PI/2;
	//alpha_median = alpha_median + (sign(quart)*abs(quart%2))*M_PI/2;
	//alpha_median += (quart%2)*M_PI/2.;
	cout << "alpha_median : " << alpha_median*180/M_PI << " " << quart%2<< endl;

	alpha_median = modulo(alpha_median, 2*M_PI);

	cout << "alpha_median : " << alpha_median*180/M_PI << " " << quart%2<< endl;

	/*
	float medx = median(Msn);
    float medy = -median(Mew);
	*/
    
    float medx = sign(cos(State[2].mid()))*median(Mew);
    float medy = sign(sin(State[2].mid()))*median(Msn);

    //medx = mean(Mew);
    //medy = mean(Msn);

    /*
    cout << "Msn : ";
    for(int i=0; i<Msn.size(); i++){
		cout << Msn[i] << "|";
	}
	cout << endl;

	cout << "Mew : ";
    for(int i=0; i<Mew.size(); i++){
		cout << Mew[i] << "|";
	}
	cout << endl;
	

    cout << lines.size() << endl;
    cout << Msn.size() << endl;
    cout << Mew.size() << endl;
	*/
    cout << "medx : " << medx << " | " << "medy : " << medy << endl;
    cout << "medx : " << mean(Msn) << " | " << "medy : " << mean(Mew) << endl;
	
	
	IntervalVector X(3, Interval::ALL_REALS);
	//cout << State[0].mid() << " | " << State[0].mid() - percent*size_carrelage_x/2. << " | " << State[0].mid() + percent*size_carrelage_x/2. << endl;

	X[0] = Interval(State[0].mid()).inflate(percent*size_carrelage_x/2.);
	X[1] = Interval(State[1].mid()).inflate(percent*size_carrelage_y/2.);
	

	X[2] = Interval(modulo(alpha_median, 2*M_PI)).inflate(0.1);
	//X[2] = Interval(State[2].mid()-0.1, State[2].mid()+0.1);
	//X[2] = Interval(State[2].mid()).inflate(0.03);

	//normalisation :
	X[0] = X[0]/size_carrelage_x;
	X[1] = X[1]/size_carrelage_y;
	//vibes::selectFigure("Map");


	//test avec le nouveau doc de Jaulin
	IntervalVector box0(6, Interval::ALL_REALS);
	IntervalVector box1(6, Interval::ALL_REALS);
	//IntervalVector box2(6, Interval::ALL_REALS);
	//IntervalVector box3(6, Interval::ALL_REALS);
	//alpha_median += M_PI;
	box0[0] = X[0], box0[1] = X[1], box0[2] = X[2], box0[3] = Interval(medx).inflate(0.03), box0[4] = Interval(medy).inflate(0.03), box0[5] = Interval(alpha_median).inflate(0.1); //X[2];
	box1[0] = X[0], box1[1] = X[1], box1[2] = X[2], box1[3] = Interval(medx).inflate(0.03), box1[4] = Interval(medy).inflate(0.03), box1[5] = Interval(alpha_median).inflate(0.1); //X[2];
	//box2[0] = X[0], box2[1] = X[1], box2[2] = X[2], box2[3] = Interval(medx).inflate(0.03), box2[4] = Interval(medy).inflate(0.03), box2[5] = Interval(alpha_median-0.1, alpha_median+0.1); //X[2];
	//box3[0] = X[0], box3[1] = X[1], box3[2] = X[2], box3[3] = Interval(medx).inflate(0.03), box3[4] = Interval(medy).inflate(0.03), box3[5] = Interval(alpha_median-0.1, alpha_median+0.1); //X[2];

	//nouveau code de Jaulin:
	ibex::Function f1("x[3]", "y[3]", "(sin(pi*(x[0]-y[0])) ; sin(pi*(x[1]-y[1])) ; sin(x[2]-y[2]))");
	ibex::Function f2("x[3]", "y[3]", "(sin(pi*(x[0]-y[1])) ; sin(pi*(x[1]-y[0])) ; cos(x[2]-y[2]))");

	//ibex::Function f3("x[3]", "y[3]", "(sin(pi*(x[0]-y[0])) ; sin(pi*(x[1]-y[1])) ; cos(x[2]-y[2]))");
	//ibex::Function f4("x[3]", "y[3]", "(sin(pi*(x[0]-y[1])) ; sin(pi*(x[1]-y[0])) ; sin(x[2]-y[2]))");
	//tubex::CtcFunction ctc_g = ctc_g1|ctc_g2;

	ibex::CtcFwdBwd c1(f1);
	ibex::CtcFwdBwd c2(f2);
	//ibex::CtcFwdBwd c3(f1);
	//ibex::CtcFwdBwd c4(f2);

	//cout << "box0 0 : " << box0 << endl;
	//cout << "box1 0 : " << box1 << endl;


	c1.contract(box0);
	//cout << "box0 1 : " << box0 << endl;
	//cout << "box1 1 : " << box1 << endl;

	c2.contract(box1);
	//cout << "box0 2 : " << box0 << endl;
	//cout << "box1 2 : " << box1 << endl;

	//c3.contract(box2);
	//c4.contract(box3);

	//c0.contract(box0);

	IntervalVector box(3, Interval::ALL_REALS);
	box[0] = box0[0]|box1[0]/*|box2[0]|box3[0]*/;
	box[1] = box0[1]|box1[1]/*|box2[0]|box3[0]*/;
	box[2] = box0[2]|box1[2]/*|box2[0]|box3[0]*/;

	//cout << "X : " << X << endl;
	//cout << "box : " << box << endl;

	if(box[0].is_empty() or box[1].is_empty()){
		cout << "X empty" << endl;
	}else{	

		State[0] = box[0];
		State[1] = box[1];

		//vibes::drawBox(State.subvector(0,1), "blue");

		State[0] = State[0]*size_carrelage_x;
		State[1] = State[1]*size_carrelage_y;
		State[2] = box[2];

		/*
		//on remets l'angle entre 0 ep PI/2
		float lb = State[2].lb();
		float ub = State[2].ub();

		if(lb < 0){
			lb += 2*M_PI;
			ub += 2*M_PI;
		}

		if(ub > 2*M_PI){
			ub -= 2*M_PI;
		}

		if(lb > ub){
			float f = lb;
			lb = ub;
			ub = f;
		}
		State[2] = Interval(lb, ub);
		*/

		//vibes::drawCircle(100*alpha, d, 2, "blue[blue]");
		float a = (State[2].mid())*180./M_PI;
		cout << "angle robot : " << a << endl;
		vibes::drawBox(State.subvector(0, 1), "pink");

		vibes::drawVehicle(State[0].mid(), State[1].mid(), a, 0.4, "blue");
		
		//vibes::drawCircle(nx/1000., a/360. - 4, 0.01, "green[green]");

	}

	//vibes::drawCircle(State[0].mid(), medx, 0.01, "blue[blue]");
	//vibes::drawCircle(State[0].mid(), medy - 1, 0.01, "red[red]");


	//vibes::drawCircle(nx/1000., medx - 2, 0.01, "blue[blue]");
	//vibes::drawCircle(nx/1000., medy - 3, 0.01, "red[red]");

	//vibes::drawCircle(nx/1000., alpha_median/(2*M_PI) - 5, 0.01, "magenta[magenta]");
	
	//vibes::drawCircle(nx/1000., mean(Mew) - 6, 0.01, "blue[blue]");
	//vibes::drawCircle(nx/1000., mean(Msn) - 7, 0.01, "red[red]");

	nx +=2;

	cout << "state fin : " << State << endl;

	cout << endl;

}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    cv::Mat in = cv_bridge::toCvShare(msg, "bgr8")->image;
    
    //cv::Size size(350,350);
    //cout << in.size[0] << " " <<in.size[1] << endl;
    cv::Mat out;
    //cv::resize(in,out,size);
    //transpose(matSRC, matROT);  
	//flip(matROT, matROT,1); //transpose+flip(1)=CW
    //out = in;
    cv::flip(in,out,1); //L'image doit être retournée en raison de l'utilisation de la caméra de VREP
    cv::Mat grey;
    cv::Mat bin;
    if(display_window){
      cv::imshow("View base", out);
    }

	cvtColor(out, grey, CV_BGR2GRAY);
	
	frame_width = out.size[1];
	frame_height = out.size[0];
	Mat src = Mat::zeros(Size(frame_width, frame_height), CV_8UC3);

	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	Sobel(grey, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );

	/// Gradient Y
	Sobel(grey, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
	//imshow("Sobel",grad);

	//imshow("Traitement",grad);

	Mat edges;
    Canny(grad, edges, 50, 255,3);
    std::vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI/180., 60, 120, 50);

    std::vector<line_struct> lines_points;
    std::vector<float> lines_angles;
    

	float x1,x2,y1,y2;
    for(int i=0; i<lines.size(); i++){

		cv::Vec4i l = lines[i];
		x1 = l[0],y1 = l[1],x2 = l[2],y2 = l[3];
	    float angle_line = atan2(y2-y1,x2-x1);
		line(src, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), Scalar(255, 0, 0), 3, CV_AA);
		line_struct ln;
		ln.p1 = cv::Point(l[0], l[1]);
		ln.p2 = cv::Point(l[2], l[3]);
		ln.angle = angle_line;
		lines_points.push_back(ln);
		
		//float angle4 = fmod(angle_line*4.,(float)(2.*M_PI));
		float angle4 = modulo(angle_line+M_PI/4., M_PI/2.)-M_PI/4.;
		//float angle4 = modulo(angle_line+M_PI/3., M_PI*2/3.)-M_PI/3.;

		//if(angle4 < 0)
		//{
		//	angle4 += 2.*M_PI;
		//}
		
		lines_angles.push_back(angle4);
    }

    float median_angle = median(lines_angles);
    alpha_median = median_angle;
    std::vector<Vec4i> lines_good;

    float anglegle;
    for (int i=0; i<lines_points.size(); i++){

		//cout << "ligne : " << lines[i]<< " | " << lines_angles[i] << " | " << sawtooth(lines_angles[i]-median_angle) << endl;
		//cout << lines_angles[i]-median_angle << endl;

		//anglegle = modulo(lines_angles[i]+M_PI/4., M_PI/2.)-M_PI/4.;
		anglegle = lines_angles[i];
    	if(sawtooth(anglegle-median_angle) < 0.1)
    	{
			line(src, lines_points[i].p1, lines_points[i].p2, Scalar(255, 0, 0), 3, CV_AA);
    		lines_good.push_back(lines[i]);
    	}
    	else
    	{
    		cout << "prob lignes : " << lines_angles[i] << " | " << sawtooth(lines_angles[i]-median_angle) << endl;
    		line(src, lines_points[i].p1, lines_points[i].p2, Scalar(0, 255, 0), 3, CV_AA);
    	}
    }

    if(lines_good.size() > 5){
    	cout << "il y a : " << lines_good.size() << " lignes" << endl;

	    calc_new_pos(lines_good);

    }else{
    	cout << "Pas assez de lignes (" << lines_good.size() << ")" << endl;
    }
    
	//imshow("Sobel",grad);
	//imshow("Canny",edges);
	imshow("view",src);

	/*
    int n = 10;
    float sumPos = 100;
	float sumTheta = 0.;
    line_follow::Message_Line msg;
    if(n > 0) // On a trouvé une ligne correcte
    {
      
      msg.pos = ((sumPos/n)/out.rows - 0.5);
      msg.angle = sumTheta/n;
      msg.data_correct = true;        
    }
    else // On n'a pas trouvé de ligne correcte
    {
      msg.pos = 0;
      msg.angle = 0;
      msg.data_correct = false;
    }
    chatter_line.publish(msg);
    */

    /*
    if(display_window)
    {
      cv::imshow("View", out);
    }*/
    

  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

int main(int argc, char **argv)
{
	//init pos
	State[0] = Interval(0., 0.);
	State[1] = Interval(0., 0.);
	State[2] = Interval(0., 0.);
 	//State[2] = Interval(M_PI/2., M_PI/2.);
	//State[2] = Interval(M_PI, M_PI);
	//State[2] = Interval(-M_PI/2., -M_PI/2.);

	//State[2] = Interval(20.*M_PI/180., 20.*M_PI/180.);

	last_alpha_median = State[2].mid();
	quart = floor(State[2].mid()/(M_PI/2.));
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

	cv::startWindowThread();
	}

	vibes::beginDrawing();
	VIBesFigMap fig_map("Map");
	vibes::setFigureProperties("Map",vibesParams("x", 10, "y", -10, "width", 100, "height", 100));
	vibes::axisLimits(-10, 10, -10, 10, "Map");

    fig_map.show();

	chatter_line = nh.advertise<line_follow::Message_Line>("line_data", 1000);
	image_transport::ImageTransport it(nh);
	image_transport::Subscriber sub = it.subscribe("image", 1, imageCallback);
	ros::spin();
	//cv::destroyWindow("view");
	return 0;
}