/*
 *  stereo_match.cpp
 *  calibration
 *
 *  Created by Victor  Eruhimov on 1/18/10.
 *  Copyright 2010 Argus Corp. All rights reserved.
 *
 */
#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include <stdio.h>

using namespace cv;
using namespace std;

#define LEFT_CAM 1
#define RIGHT_CAM 2


int x_int = 0;
int y_int = 0;

void my_mouse_callback(int event,int x,int y,int flags,void* param)
{
  switch (event) {
  case CV_EVENT_LBUTTONDOWN:{
    x_int = x;
    y_int = y;
    cout<<"x_int = "<<x_int<<"\t"<<"y_int = "<<y_int<<endl;
  }
  break;
  }
}


static void saveXYZ(const char* filename, const Mat& mat)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}

int main(int argc, char** argv)
{
    namedWindow("left", 1);
    namedWindow("right", 1);
    namedWindow("disparity", 0);
    cvSetMouseCallback("disparity",my_mouse_callback,NULL);


    enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3, STEREO_3WAY=4 };
    int alg = STEREO_BM;
    std::string intrinsic_filename = "intrinsics.yml";
    std::string extrinsic_filename = "extrinsics.yml";
    std::string disparity_filename = "disparity.bmp";
    std::string point_cloud_filename = "PointCloud";
    std::string pointcloud_img_filename = "pointcloud_img.bmp";
    std::string left_img_filename = "left_img.bmp";
    Ptr<StereoBM> bm = StereoBM::create(16,9);
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);


    float scale = 1.0;
    int SADWindowSize = 15;//5-21//15
    int numberOfDisparities=320;//32
    bool no_display=false;

    int color_mode = alg == STEREO_BM ? 0 : -1;

    if ( numberOfDisparities < 1 || numberOfDisparities % 16 != 0 )
    {
        printf("Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer divisible by 16\n");
        return -1;
    }
    if (scale < 0)
    {
        printf("Command-line parameter error: The scale factor (--scale=<...>) must be a positive floating-point number\n");
        return -1;
    }
    if (SADWindowSize < 1 || SADWindowSize % 2 != 1)
    {
        printf("Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n");
        return -1;
    }

    FileStorage fs(intrinsic_filename, FileStorage::READ);
    if(!fs.isOpened())
    {
        printf("Failed to open file %s\n", intrinsic_filename.c_str());
        return -1;
    }

    Mat M1, D1, M2, D2;
    fs["M1"] >> M1;
    fs["D1"] >> D1;
    fs["M2"] >> M2;
    fs["D2"] >> D2;

    fs.open(extrinsic_filename, FileStorage::READ);
    if(!fs.isOpened())
    {
        printf("Failed to open file %s\n", extrinsic_filename.c_str());
        return -1;
    }

    Mat R, T, R1, P1, R2, P2;
    fs["R"] >> R;
    fs["T"] >> T;
//          fs["R1"] >> R1;
//          fs["P1"] >> P1;
//          fs["R2"] >> R2;
//          fs["P2"] >> P2;
//          fs["Q"] >> Q;


    while(1)
    {
          cout<<"x_int = "<<x_int<<endl;
          cout<<"y_int = "<<y_int<<endl;

          VideoCapture capture_stero_l(LEFT_CAM);//select left camera
          VideoCapture capture_stero_r(RIGHT_CAM);//select right camera
          Mat img1,img2;
          capture_stero_l >> img1;
          capture_stero_r >> img2;
          imwrite(left_img_filename, img1);
          cvtColor(img1,img1,COLOR_BGR2GRAY);
          cvtColor(img2,img2,COLOR_BGR2GRAY);
          cout<<"img_size = "<<img1.cols<<"*"<<img1.rows<<endl;

          if((img1.empty())||(img2.empty()))
          {
                std::cout << "Camera capture_stero failed!"<<std::endl;
          }
          if (scale != 1.f)
          {
              Mat temp1, temp2;
              int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
              resize(img1, temp1, Size(), scale, scale, method);
              img1 = temp1;
              resize(img2, temp2, Size(), scale, scale, method);
              img2 = temp2;
          }

          Size img_size = img1.size();
          Rect roi1, roi2;
          Mat Q;

//          cout<<"M1 = "<<M1<<endl;
//          cout<<"D1 = "<<D1<<endl;
//          cout<<"M2 = "<<M2<<endl;
//          cout<<"D2 = "<<D2<<endl;

          M1 *= scale;
          M2 *= scale;

          stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );

//          cout<<"R = "<<R<<endl;
//          cout<<"T = "<<T<<endl;
//          cout<<"R1 = "<<R1<<endl;
//          cout<<"P1 = "<<P1<<endl;
//          cout<<"R2 = "<<R2<<endl;
//          cout<<"P2 = "<<P2<<endl;
//          cout<<"Q = "<<Q<<endl;

          Mat map11, map12, map21, map22;
          initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
          initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

          Mat img1r, img2r;
          remap(img1, img1r, map11, map12, INTER_LINEAR);
          remap(img2, img2r, map21, map22, INTER_LINEAR);

          img1 = img1r;
          img2 = img2r;


          numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;

          bm->setROI1(roi1);
          bm->setROI2(roi2);
          bm->setPreFilterType(CV_STEREO_BM_NORMALIZED_RESPONSE);
          bm->setPreFilterSize(13);//5-21
          bm->setPreFilterCap(13);//1-31
          bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
          bm->setMinDisparity(0);
          bm->setNumDisparities(numberOfDisparities);
          bm->setTextureThreshold(100);//10
          bm->setUniquenessRatio(10);//5-15//1//5
          bm->setSpeckleWindowSize(9);//1//9
          bm->setSpeckleRange(4);//100
          bm->setDisp12MaxDiff(1);//1

          sgbm->setPreFilterCap(13);
          int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
          sgbm->setBlockSize(sgbmWinSize);
          int cn = img1.channels();
          sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);
          sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
          sgbm->setMinDisparity(0);
          sgbm->setNumDisparities(numberOfDisparities);
          sgbm->setUniquenessRatio(10);
          sgbm->setSpeckleWindowSize(9);
          sgbm->setSpeckleRange(32);
          sgbm->setDisp12MaxDiff(1);
          if(alg==STEREO_HH)
              sgbm->setMode(StereoSGBM::MODE_HH);
          else if(alg==STEREO_SGBM)
              sgbm->setMode(StereoSGBM::MODE_SGBM);
          else if(alg==STEREO_3WAY)
              sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);

          Mat disp, disp8;
          Mat img1p, img2p, dispp;
          copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
          copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);


          int64 t = getTickCount();
          if( alg == STEREO_BM )
              bm->compute(img1p, img2p, dispp);
          else if( alg == STEREO_SGBM || alg == STEREO_HH || alg == STEREO_3WAY )
              sgbm->compute(img1p, img2p, dispp);
          t = getTickCount() - t;
          printf("Time elapsed: %fms\n", t*1000/getTickFrequency());
          disp = dispp.colRange(numberOfDisparities, img1p.cols);

          if( alg != STEREO_VAR )
              disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
          else
              disp.convertTo(disp8, CV_8U);

          normalize( disp, disp, 0, 256, CV_MINMAX );
          if( !no_display )
          {
              imshow("left", img1);
              imshow("right", img2);
              imshow("disparity", disp8);
          }

          imwrite(disparity_filename, disp8);
          printf("storing the point cloud...");
          fflush(stdout);
          Mat xyz;
          reprojectImageTo3D(disp, xyz, Q, true);

          vector<cv::Mat> rgbchannels(3);
          split(xyz,rgbchannels);
          Mat channel_x = rgbchannels[0];
          Mat channel_y = rgbchannels[1];
          Mat channel_z = rgbchannels[2];
//          namedWindow("pointcloud_x", 1);
//          imshow("pointcloud_x",channel_x);
//          namedWindow("pointcloud_y", 1);
//          imshow("pointcloud_y",channel_y);
//          namedWindow("pointcloud_z", 1);
//          imshow("pointcloud_z",channel_z);

          namedWindow("pointcloud", 1);
          imshow("pointcloud",xyz);
          imwrite(pointcloud_img_filename, xyz);

          Vec3f point_ = xyz.at<Vec3f>(x_int, y_int);
          cout<<"act position of mouse = "<<float(point_[0])<<"\t"<<float(point_[1])<<"\t"<<float(point_[2])<<"\t"<<endl;

          saveXYZ(point_cloud_filename.c_str(), xyz);
          printf("\n");


          char c = (char)waitKey(1);
          if( c == 27 || c == 'q' || c == 'Q' )
              break;
    }
    return 0;
}
