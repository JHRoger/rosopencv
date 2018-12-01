#include <ros/ros.h> 
#include <image_transport/image_transport.h> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp> 
#include <cv_bridge/cv_bridge.h> 
#include <stdio.h>
#include "armordetection.h"
#include "serialport.h"



using namespace cv;
using namespace std;

Serialport serial("/dev/ttyUSB0");
armordetection armor;

void imageCallback(const sensor_msgs::ImageConstPtr& msg) 
{ 
    try //尝试执行代码，数据不符合的扔出
    { 
        cout <<"\n\n--------------------第\t" << msg->header.seq << "帧--------------------"<<endl;

        // cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
        cout << "frameNumber:" << msg->header.seq << endl;
        cout << "stamp:" << msg->header.stamp << endl;
        cout << "lis_WallTime:" << ros::WallTime::now() << endl;

        //clock start
        double time_Start = (double)clock();
        cv::Mat g_srcImage = cv_bridge::toCvShare(msg, "bgr8")->image;

        double time_read = (double)clock();
        cout << "read time:" << (time_read - time_Start) / 1000 << endl;

        armor.cut(g_srcImage);
        
        double time_cut = (double)clock();
        cout << "cut time:" << (time_cut - time_read) / 1000 << endl;

        armor.Bright();
        armor.BrighttoCanny();
        armor.filter();

        imshow("view",armor.filterRectImage);
        // cout << "1" << endl;
        double time_wait = (double)clock();
        cout << "filter time:" << (time_wait - time_cut)/ 1000<< endl;


        //time end
        double time_End = (double)clock();

        //显示小数点后三位并补全****rbl2333*****
        cout.setf(ios::fixed);
        cout << "detecting armor in this frame takes\t" << fixed << setprecision(3)<< (time_End - time_Start) / 1000 << "\tms!\n";
        cout << "------------------------------------------------\n" << endl;


    } 
    catch (cv_bridge::Exception& e) //获取指定的扔出数据
    { 
        
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str()); 
    } 
} 

int main(int argc, char** argv) 
{ 
    ros::init(argc, argv, "image_listener"); 
    ros::NodeHandle nh; 
    cv::namedWindow("view");
    cv::startWindowThread();

    char r_b;
    std::cout << "Red armor?[Y/n]";
    std::cin >> r_b;

    
    armor.setup();
    // serial.open_port("/dev/ttyUSB0");
    serial.set_opt(9600,8,'O',2);

    if( r_b == 'y' || r_b == 'Y')
    {
        armor.R_B = false;
        cout << "Red armor!" << endl;
    }
    else
    {
        armor.R_B = true;
        cout << "Blue armor!" << endl;
    }    
        
    image_transport::ImageTransport it(nh); 
    image_transport::Subscriber sub = it.subscribe("camera/image", 1, imageCallback); 
    
    // ros::Rate loop_rate(30);
    while (ros::ok())
    {
              
        ros::spinOnce();
        
       
        // loop_rate.sleep();
    }
    
}

