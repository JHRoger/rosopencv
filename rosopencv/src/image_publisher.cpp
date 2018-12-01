#include <ros/ros.h> 
#include <image_transport/image_transport.h> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp> 
#include <cv_bridge/cv_bridge.h> 
#include <stdio.h>
#include <time.h>
#include "rosopencv/ImageFrame.h"
#include "rosopencv/ImageAddress.h"
#include <sys/file.h>
#include <string.h>

#define FFF 666

using namespace cv;
using namespace std;
int main(int argc, char** argv) 
{ 
    ros::init(argc, argv, "image_publisher"); 
    ros::NodeHandle nh; 
    image_transport::ImageTransport it(nh); 
    image_transport::Publisher pub = it.advertise("camera/image", 1);
    // ros::Publisher pub_a = nh.advertise<rosopencv::ImageAddress>("camera/address",1);

    cv::VideoCapture capture(1);
    cv::Mat image;
    capture.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
    capture.set(CV_CAP_PROP_FRAME_WIDTH,1920);
    int rate;//帧率
    long sec;//视频时间戳
    rate = capture.get(CV_CAP_PROP_FPS);
    cout << "rate:" << rate << endl;
    
    rosopencv::ImageFrame msgs;

    ros::Rate loop_rate(rate);
    
    while (nh.ok()) 
    { 
        double time_Start_Frame = (double)clock();
        capture >> image;
        sec = capture.get(CV_CAP_PROP_POS_MSEC);
        cout << "camera_time:" << sec << endl;
        
        if(image.empty())
        { 
            printf("open error\n"); 
        } 
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg(); 
        msg->header.stamp = ros::Time::now();
        cout << "ROS_time:" << msg->header.stamp << endl;
        // cout << " msg :" << msg << endl;
        // msgs.image = *msg;
        
        // msgs.frame_time.actor_time = sec;
        
        // rosopencv::ImageAddress msg_a;

        // pub_a.publish(msg_a);
        // for( int j = 0 ; j < 32 ; j++)
        // cout << msg_a.address[j] ;
        // cout <<endl;
        pub.publish(msg);
        
        double time_End_Frame = (double)clock();
        cout << "pub time:" << (time_End_Frame - time_Start_Frame) / 1000 << endl;//发布消息计时
        ros::spinOnce(); 
        loop_rate.sleep();         
    }
    
}
