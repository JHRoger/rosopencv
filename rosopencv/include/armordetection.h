#ifndef ARMORDETECTION_H
#define ARMORDETECTION_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <math.h>

#define pi 3.1415926

using namespace std;
using namespace cv;

class armordetection
{
public:
    armordetection();
    ~armordetection();
    void setup();
    void cut(Mat &g_srcImage);
    void Bright();
    void Bright(int alpha , int nThresh);
    void BrighttoCanny();
    void BrighttoCanny(int CannyLowThreshold);
    void ShowAreaRect();
    void filter();

    Mat srcImage ;
    vector<Mat> rgbChannels;
    bool R_B;
    vector<vector<Point> > BrighttoCannyContours;
    vector<Vec4i> BrighttoCannyHierarchy;
    Mat thresholdImageRed;
    Mat thresholdImageBlue;
    Mat BrighttoCannyImage;
    Mat filterRectImage;
    vector<bool> armor_frame;
    vector<Point2i> chasecenter;
    vector<int>     chasedistance;
    bool armor_location;
    Point2i pre_pt;
    int width ;
    int height;

private:
    Mat Kern = (cv::Mat_<char>(3,3) << 1,1,1,1,1,1,1,1,1);//卷积核 具有增强效果
    /** @name效果不错的卷积核
      1,1,1,
      1,1,1,
      1,1,1

      1,0,1,
      0,1,0,
      1,0,1
    */

};

void armordetection::setup()
{
    srcImage = Mat::zeros(height, width, CV_8UC3);

    thresholdImageRed = Mat::zeros(height, width, CV_8UC3);
    thresholdImageBlue = Mat::zeros(height, width, CV_8UC3);
    BrighttoCannyImage = Mat::zeros(height, width, CV_8UC3);
    filterRectImage = Mat::zeros(height, width, CV_8UC3);

    BrighttoCannyContours.resize(0);
    BrighttoCannyHierarchy.resize(0);
    chasecenter.resize(0);
    chasedistance.resize(0);
    armor_frame.resize(0);

    rgbChannels.resize(3);
    R_B = false;
    armor_location = false;

    pre_pt.x = 0;
    pre_pt.y = 0;

    width = 640;
    height = 480;
}

void armordetection::cut(Mat &g_srcImage)
{
    Mat cutImage;
    g_srcImage.copyTo(cutImage);

    Point2i center_point;
    Point2i cur_pt;

    if(armor_location)
    {
        center_point.x = chasecenter.back().x;
        center_point.y = chasecenter.back().y;
    }
    else
    {
        center_point.x = 960;
        center_point.y = 540;
    }
    cur_pt.x = center_point.x + width / 2;
    cur_pt.y = center_point.y + height / 2;
    pre_pt.x = center_point.x - width / 2;
    pre_pt.y = center_point.y - height / 2;

    cout << "center_point:" << center_point << endl;

    if(pre_pt.x < 0)
    {
        pre_pt.x = 0;
    }
    if(pre_pt.y < 0)
    {
        pre_pt.y = 0;
    }
    if(cur_pt.x > 1920)
    {
        pre_pt.x = pre_pt.x - (cur_pt.x - 1920);
    }
    if(cur_pt.y > 1080)
    {
        pre_pt.y = pre_pt.y - (cur_pt.y - 1080);
    }
    cout << "pre_pt:" << pre_pt <<endl;

    srcImage = cutImage(Rect(pre_pt.x,pre_pt.y,width,height));
}
//改变亮度并卷积加强边缘
void armordetection::Bright()

{
    int alpha = 10;
    int nThresh = 100;
    //改变图像亮度
    cv::Mat Bright_image = srcImage;
    double Bright_alpha = alpha/10;

    cv::Mat Bright_change;
    Bright_image.convertTo(Bright_change,-1,Bright_alpha,nThresh-255);
    //分离通道
    cv::split(Bright_change,rgbChannels);

    //卷积处理 卷积核为Kern

    filter2D(rgbChannels[0], rgbChannels[0], rgbChannels[0].depth(), Kern);

    filter2D(rgbChannels[2], rgbChannels[2], rgbChannels[2].depth(), Kern);
    //膨胀处理
    Mat element;

    dilate(rgbChannels[0],rgbChannels[0],element,Point(-1,-1),2);

    dilate(rgbChannels[2],rgbChannels[2],element,Point(-1,-1),2);

    // imshow("RED",rgbChannels[2]);
    // imshow("BLUE",rgbChannels[0]);

}

void armordetection::Bright(int alpha, int nThresh)

{
    //改变图像亮度
    cv::Mat Bright_image = srcImage;
    double Bright_alpha = alpha/10;

    cv::Mat Bright_change;
    Bright_image.convertTo(Bright_change,-1,Bright_alpha,nThresh-255);
    //分离通道
    cv::split(Bright_change,rgbChannels);

    //卷积处理 卷积核为Kern

    filter2D(rgbChannels[0], rgbChannels[0], rgbChannels[0].depth(), Kern);

    filter2D(rgbChannels[2], rgbChannels[2], rgbChannels[2].depth(), Kern);
    //膨胀处理
    Mat element;

    dilate(rgbChannels[0],rgbChannels[0],element,Point(-1,-1),2);

    dilate(rgbChannels[2],rgbChannels[2],element,Point(-1,-1),2);

    // imshow("RED",rgbChannels[2]);
    // imshow("BLUE",rgbChannels[0]);

}


void armordetection::BrighttoCanny()
//卷积后Canny处理并输出轮廓
{
    int CannyLowThreshold = 150;

    threshold(rgbChannels[0],thresholdImageBlue,160,255,THRESH_BINARY);
    threshold(rgbChannels[2],thresholdImageRed,160,255,THRESH_BINARY);

    //Canny边缘检测
    if(R_B)
    {
        Canny(thresholdImageBlue,BrighttoCannyImage, CannyLowThreshold, CannyLowThreshold*3, 3);
    }
    else
    {
        Canny(thresholdImageRed,BrighttoCannyImage, CannyLowThreshold, CannyLowThreshold*3, 3);
    }

    //寻找轮廓
    findContours(BrighttoCannyImage,BrighttoCannyContours,BrighttoCannyHierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE,Point(0,0));
}

void armordetection::BrighttoCanny(int CannyLowThreshold)
//卷积后Canny处理并输出轮廓
{
    threshold(rgbChannels[0],thresholdImageBlue,160,255,THRESH_BINARY);
    threshold(rgbChannels[2],thresholdImageRed,160,255,THRESH_BINARY);

    //Canny边缘检测
    if(R_B)
    {
        Canny(thresholdImageBlue,BrighttoCannyImage, CannyLowThreshold, CannyLowThreshold*3, 3);
    }
    else
    {
        Canny(thresholdImageRed,BrighttoCannyImage, CannyLowThreshold, CannyLowThreshold*3, 3);
    }

    //寻找轮廓
    findContours(BrighttoCannyImage,BrighttoCannyContours,BrighttoCannyHierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE,Point(0,0));

}

//旋转矩形包围轮廓方便观察
void armordetection::ShowAreaRect()
{
    RotatedRect minRect;
    Mat minAreaRectImage = Mat::zeros(BrighttoCannyImage.size(), CV_8UC3);
    for(int j = 0 ; j < BrighttoCannyContours.size(); j++)
    {
        //最小旋转矩形包围轮廓并输出四顶点
        minRect = minAreaRect(Mat(BrighttoCannyContours[j]));
        Point2f rect_points[4];// The order is bottomRight, bottomLeft, topLeft, topRight
        minRect.points(rect_points);


        //画旋转矩形
        for (int k = 0; k < 4; k++)
        {
            line(minAreaRectImage, rect_points[k], rect_points[(k + 1) % 4],Scalar(255,0,255), 2);
        }
    }

    namedWindow("minAreaRect",WINDOW_AUTOSIZE);
    imshow("minAreaRect",minAreaRectImage);
}

void armordetection::filter()
//旋转矩形包围轮廓并筛选出合适的矩形对
{
    vector<RotatedRect> filteredRect (0);
    Point2f frect_points[4];
    float real_angle;
    filterRectImage = srcImage.clone();
    int counter = 0;

    //遍历轮廓
    for(int m = 0 ; m < BrighttoCannyContours.size();m++)
    {
        RotatedRect minRotatedRect = minAreaRect(BrighttoCannyContours[m]);
        minRotatedRect.points(frect_points);
        float real_angle;
        real_angle = 180*(atan2(frect_points[1].y - frect_points[2].y, frect_points[1].x - frect_points[2].x))/pi;
        float width = minRotatedRect.size.width;
        float height = minRotatedRect.size.height;

        if( width > height )
        {
            std::swap(width,height);
        }

        //调取点参数
        Point2i Contour_center;
        Contour_center = minRotatedRect.center;

        //初步筛选旋转矩阵并将符合标准的旋转矩阵另存在filteredRect
        vector<float> s(7);//s = size 外形标准
        s[0] = height/width;
        s[1] = height*width;
        s[2] = minRotatedRect.angle;
        s[3] = real_angle;
        s[4] = sqrt(pow(frect_points[1].y - frect_points[2].y,2) + pow(frect_points[1].x - frect_points[2].x,2));//real height
        s[5] = sqrt(pow(frect_points[1].y - frect_points[0].y,2) + pow(frect_points[1].x - frect_points[0].x,2));//real width



        bool armor_exist;

        //长度 面积 角度初步筛选
        if( (s[0] < 7 ||s[0] > 1.2) && (1) && (s[1] > 20) && ( s[3] > 70 || s[3] < 110 ) )
        {
            //判断颜色
            if(R_B)
            {
                cout << "cont" << Contour_center<< endl;
                int red = rgbChannels[2].at<uchar>(Contour_center);
                cout << "red" <<  red << endl;
                if( 1 )
                    armor_exist = true;
            }
            else
            {
                cout << "cont" << Contour_center<< endl;
                int blue = rgbChannels[0].at<uchar>(Contour_center);
                cout << "blue" <<  blue << endl;
                if( 1 )
                    armor_exist = true;
            }

            if(armor_exist)
            {
                armor_exist = false;
                for (int k = 0; k < 4; k++)
                {
                    line(filterRectImage, frect_points[k], frect_points[(k + 1) % 4],Scalar(255,0,0), 2);
                }

                filteredRect.push_back( minRotatedRect);
                counter++;

                 //标中心点 筛选颜色后
                 circle(filterRectImage, minRotatedRect.center, 7, cv::Scalar(0, 0, 255), 7);
            }
        }
    }

    //寻找合适的矩形对
    int length;//相对位置
    float height_quotient;//高度比
    float width_quotient;//宽度比
    float a_min;
    float a_max;
    float b_min;
    float b_max;
    float a_real_angle;
    float b_real_angle;
    int theta_angle;//相对角度
    int Rects_number = 0;

    vector<RotatedRect> armorDetectionLeft(0);
    vector<RotatedRect> armorDetectionRight(0);

    if( counter > 1 )
    {
        //冒泡
        for(int a = 0; a < counter ;a++)
        {
            for(int b = a; b < counter ;b++)
            {
                a_min = (filteredRect[a].size.height>filteredRect[a].size.width) ? filteredRect[a].size.width : filteredRect[a].size.height;
                a_max = (filteredRect[a].size.height>filteredRect[a].size.width) ? filteredRect[a].size.height : filteredRect[a].size.width;
                b_min = (filteredRect[b].size.height>filteredRect[b].size.width) ? filteredRect[b].size.width : filteredRect[b].size.height;
                b_max = (filteredRect[b].size.height>filteredRect[b].size.width) ? filteredRect[b].size.height : filteredRect[b].size.width;

                length = sqrt(pow(filteredRect[a].center.x - filteredRect[b].center.x,2) + pow(filteredRect[a].center.y - filteredRect[b].center.y,2)) ;
                theta_angle = abs(int(filteredRect[a].angle - filteredRect[b].angle));
                height_quotient = b_max/a_max;
                width_quotient = b_min/a_min;

                if( length > a_min*1.5 && length > b_min*1.5 && length < a_max*3.0 && length < b_max*3.0
                 && (theta_angle < 10 || theta_angle > 80)
                 && height_quotient < 1.8 && height_quotient > 0.4 && width_quotient < 1.8 && width_quotient > 0.4)
                {

                   if( abs(int(filteredRect[a].center.y - filteredRect[b].center.y)) < 20 )
                    {
                    line(filterRectImage,filteredRect[a].center,filteredRect[b].center,Scalar(255,255,0),2);
                    circle(filterRectImage,filteredRect[a].center/2+filteredRect[b].center/2,8,Scalar(0,255,0),8);

                    Rects_number++;
                    if(filteredRect[b].center.x > filteredRect[a].center.x)
                    {
                        armorDetectionRight.push_back(filteredRect[b]);
                        armorDetectionLeft.push_back(filteredRect[a]);
                    }
                    else
                    {
                        armorDetectionRight.push_back(filteredRect[a]);
                        armorDetectionLeft.push_back(filteredRect[b]);
                    }
                    }//线->角度
                }//匹配矩形对
            }//冒泡
        }//冒泡
    }

    vector<int> filteredcenter;

    if(!armorDetectionLeft.empty())
    {
        armor_location = true;
        armor_frame.push_back(true);

        Point2i armor_center_f = armorDetectionLeft[Rects_number - 1].center/2 + armorDetectionRight[Rects_number - 1].center/2;
        cout << "armor is at:" << armor_center_f + pre_pt << endl;
        filteredcenter.push_back(Rects_number - 1);//最后一个

        for(int p = 0 ; p < Rects_number - 1 ; p++)
        {

            for(int q = p + 1 ; q < Rects_number; q++)
            {
                Point2i armor_center_p = armorDetectionLeft[p].center/2 + armorDetectionRight[p].center/2;
                Point2i armor_center_q = armorDetectionLeft[q].center/2 + armorDetectionRight[q].center/2;
                if( abs(armor_center_p.x - armor_center_q.x) < 10 && abs(armor_center_p.y - armor_center_q.y) < 10)
                {
//                    cout << p << q << endl;
//                    cout << armor_center_p << armor_center_q << endl;
                break;
                }
                if( q == Rects_number - 1)
                {
                    cout << "armor is at:" << armor_center_p + pre_pt << endl;
                    filteredcenter.push_back(p);//第p+1个


                }
            }
        }
        //        cout << "the number of detected centers is:" << filteredcenter.size() << endl;

        //跟踪
        //选一个最合适的中心点
        int hh = 0;
        for(int h = 1 ; h < filteredcenter.size() ; h++)
        {

            if(armorDetectionRight[h].center.x - armorDetectionLeft[h].center.x > armorDetectionRight[hh].center.x - armorDetectionLeft[hh].center.x)
                hh = h;
        }
//            cout << "hh=" << hh << endl;

        //保存中心点
        if(chasecenter.size() < 10)//只保留后 x 个元素
            chasecenter.push_back(armorDetectionLeft[filteredcenter[hh]].center/2 + armorDetectionRight[filteredcenter[hh]].center/2);
        else
        {

            chasecenter.erase(chasecenter.begin());
            chasecenter.push_back(Point2i(armorDetectionLeft[filteredcenter[hh]].center/2 + armorDetectionRight[filteredcenter[hh]].center/2) + pre_pt);
            double len = sqrt(pow(armorDetectionLeft[filteredcenter[hh]].center.x - armorDetectionRight[filteredcenter[hh]].center.x,2) + pow(armorDetectionLeft[filteredcenter[hh]].center.y - armorDetectionRight[filteredcenter[hh]].center.y,2));
            cout << len << endl;

            double height_L;
            double height_R;

            if(armorDetectionLeft[filteredcenter[hh]].size.height > armorDetectionLeft[filteredcenter[hh]].size.width)
                height_L = armorDetectionLeft[filteredcenter[hh]].size.height;
            else
                height_L = armorDetectionLeft[filteredcenter[hh]].size.width;
            cout << "height_L" << height_L << endl;
            if(armorDetectionRight[filteredcenter[hh]].size.height > armorDetectionRight[filteredcenter[hh]].size.width)
                height_R = armorDetectionRight[filteredcenter[hh]].size.height;
            else
                height_R = armorDetectionRight[filteredcenter[hh]].size.width;

            double L_distance = 9000 / height_L ;
            double R_distance = 9000 / height_R ;
            cout << "L_distance:" << L_distance << endl;
            cout << "R_distance:" << R_distance << endl;

            Point2i L_center = Point2i(armorDetectionLeft[filteredcenter[hh]].center) + pre_pt;
            cout << "L_center:" << L_center <<endl;
            Point2i R_center = Point2i(armorDetectionRight[filteredcenter[hh]].center) + pre_pt;
            cout << "R_center:" << R_center <<endl;

            Point2i cen;
            cen.x = 540;
            cen.y = 960;
            double L_X = (L_center - cen).x / 1920. /1.19 ;
            double L_Y = (L_center - cen).y / 1080. /1.19 / 1.777 ;
            cout << "L_X:" << L_X << endl;
            cout << "L_Y:" << L_Y << endl;

        }
    }
    else
    {
        cout << "no armor found!" << endl;
        armor_frame.push_back(false);
        if(armor_frame.size() > 5)
        for(int s = 0; s < 5 ; s++)
        {
            if(armor_frame[armor_frame.size() - 1 - s] == true)
                break;
            if(s == 4)
                armor_location = false;
        }
        else
        {
            armor_location = false;
        }
    }

}

armordetection::armordetection()
{
}
armordetection::~armordetection()
{
}


#endif // ARMORDETECTION_H
