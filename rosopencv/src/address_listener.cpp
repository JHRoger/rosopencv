#include <ros/ros.h> 
#include <image_transport/image_transport.h> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp> 
#include <cv_bridge/cv_bridge.h> 
#include <stdio.h>

using namespace cv;
using namespace std;
void imageCallback(const sensor_msgs::ImageConstPtr& msg) 
{ 
    try //尝试执行代码，数据不符合的扔出
    { 
        cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
        cout << "frameNumber:" << msg->header.seq << endl;
        cout << "stamp:" << msg->header.stamp << endl;
        cout << "lis_WallTime:" << ros::WallTime::now() << endl;  
    } 
    catch (cv_bridge::Exception& e) //获取指定的扔出数据
    { 
        cv::destroyWindow("view");
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str()); 
    } 
} 
int main(int argc, char **argv) 
{ 
    ros::init(argc, argv, "image_listener"); 
    ros::NodeHandle nh; 
    cv::namedWindow("view"); 
    cv::startWindowThread(); 
    image_transport::ImageTransport it(nh); 
    image_transport::Subscriber sub = it.subscribe("camera/image", 2, imageCallback); 
    
    ros::Rate loop_rate(30);
    while (ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }
}