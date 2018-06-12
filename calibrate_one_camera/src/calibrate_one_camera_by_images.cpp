#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>
#include <iostream>
#include <vector>
#include <string.h>
#include <sstream>
#include <time.h>

using namespace std;
using namespace cv;


int n_boards = 0;
int board_w;
int board_h;

int getFiles( string path, vector<string>& files ) {
    DIR* pDir;
    struct dirent* ptr;
    if( !(pDir = opendir(path.c_str())) )
        return -1;
    while( (ptr = readdir(pDir)) != 0 )
    {
      if(strcmp(ptr->d_name,".")!=0 && strcmp(ptr->d_name,"..")!=0)
      {
        files.push_back(ptr->d_name);
      }
    }
    closedir(pDir);
    return 0;
}


int main(int argc,char** argv)
{
//  board_w = 9;
  board_w = 8;
  board_h = 6;
  n_boards = 25;
  int board_n = board_h*board_w;
  CvSize board_sz = cvSize(board_w,board_h);

  double width_ = 2448;
  double height_ = 2048;
  std::cout<<"video width = "<<width_<<"\t"<<"height = "<<height_<<std::endl;

  cvNamedWindow("calibration",CV_WINDOW_NORMAL);
  CvMat* image_points = cvCreateMat(n_boards*board_n,2,CV_32FC1);
  CvMat* object_points = cvCreateMat(n_boards*board_n,3,CV_32FC1);
  CvMat* point_counts = cvCreateMat(n_boards,1,CV_32SC1);
  CvPoint2D32f* corners = new CvPoint2D32f[board_n];
  int corner_count;
  int successes = 0;
  int step = 0;

  //read images in file
  string folder = "/home/zb/BoZhan/OpenCV/Calibration/cal_image";
  vector<string> files;
  getFiles(folder, files );  //files为返回的文件名构成的字符串向量组
  printf("image numbers is %d\n",files.size());

  IplImage *image = cvLoadImage((folder + "/" + files[0]).c_str());
  std::cout<<"image 0 is "<<(folder + "/" + files[0]).c_str()<<std::endl;
  std::cout<<"origin of image = "<<image->origin<<std::endl;
  IplImage *gray_image = cvCreateImage(cvGetSize(image),8,1);
  int img_num = 0;
  while(successes < n_boards)
  {
    img_num ++;
    int found = cvFindChessboardCorners(image,board_sz,corners,&corner_count,
                                      CV_CALIB_CB_ADAPTIVE_THRESH|CV_CALIB_CB_FILTER_QUADS);
    std::cout<<"found = "<<found<<std::endl;
    cvCvtColor(image,gray_image,CV_BGR2GRAY);
    cvFindCornerSubPix(gray_image,corners,corner_count,cvSize(11,11),cvSize(-1,-1),
                       cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER,30,0.1));
    cvDrawChessboardCorners(image,board_sz,corners,corner_count,found);
    cvShowImage("calibration",image);

    if(corner_count == board_n)
    {
      step = successes*board_n;
      for(int i =step,j=0,k=0;j<board_n;++i,++j,++k)
      {
        CV_MAT_ELEM(*image_points,float,i,0) = corners[j].x;
        CV_MAT_ELEM(*image_points,float,i,1) = corners[j].y;
//          CV_MAT_ELEM(*object_points,float,i,0) = j/board_w;
//          CV_MAT_ELEM(*object_points,float,i,1) = j%board_w;
        CV_MAT_ELEM(*object_points,float,i,0) = j%board_w;
        CV_MAT_ELEM(*object_points,float,i,1) = j/board_w;
        CV_MAT_ELEM(*object_points,float,i,2) = 0.0f;
      }
      CV_MAT_ELEM(*point_counts,int,successes,0) = board_n;
      successes++;
    }
    std::cout<<"image"<< img_num <<" is "<<(folder + "/" + files[img_num]).c_str()<<std::endl;
    image = cvLoadImage((folder + "/" + files[img_num]).c_str());
    waitKey(150);
  }

  std::cout<<"successes = "<<successes<<std::endl;

  Mat cameraMatrix_(3,3,CV_32FC1,Scalar(0)), distCoeff_(1,5,CV_32FC1,Scalar(0));
  std::vector<Mat> seqRotation, seqTranslation;
  cv::Point3f point3D;
  cv::Point2f point2D;
  std::vector<cv::Point3f> objectPoints;
  std::vector<cv::Point2f> corners_temp;
  std::vector<std::vector<cv::Point3f> > seqObjectPoints;
  std::vector<std::vector<cv::Point2f> > seqCorners;
  for ( int t=0; t<successes; t++ )
  {
      objectPoints.clear();
      corners_temp.clear();
      for ( int i=0; i<board_n; i++ )
      {
          point3D.x = CV_MAT_ELEM(*object_points,float,t*board_n+i,0);
          point3D.y = CV_MAT_ELEM(*object_points,float,t*board_n+i,1);
          point3D.z = 0;
          point2D.x = CV_MAT_ELEM(*image_points,float,t*board_n+i,0);
          point2D.y = CV_MAT_ELEM(*image_points,float,t*board_n+i,1);
          objectPoints.push_back(point3D);
          corners_temp.push_back(point2D);
      }
      seqObjectPoints.push_back(objectPoints);
      seqCorners.push_back(corners_temp);
  }


  cvReleaseMat(&object_points);
  cvReleaseMat(&image_points);
  cvReleaseMat(&point_counts);

  cv::Size image_size;
  image_size = cv::Size(image->width,image->height);
  double re_pre_err;
  re_pre_err = calibrateCamera(seqObjectPoints,seqCorners,image_size,
                               cameraMatrix_,distCoeff_,
                               seqRotation,seqTranslation,CV_CALIB_FIX_K3|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);//test ok,result is the same as Matlab
                               //ROS:CV_CALIB_FIX_K3|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5;
                               //Opencv:CV_CALIB_FIX_K4|CV_CALIB_FIX_K5|CV_CALIB_ZERO_TANGENT_DIST,test bad because k3 doesnot set zero
//  cvCalibrateCamera2(object_points2,image_points2,
//                     point_counts2,cvGetSize(image),
//                     intrinsic_matrix,distortion_coeffs,
//                     NULL,NULL,0);
  std::cout<<"re_prejection error = "<<re_pre_err<<std::endl;
  std::cout<<"cameraMatrix_ = "<<std::endl;
  std::cout<<cameraMatrix_<<std::endl;
  std::cout<<"distCoeff_ = "<<std::endl;
  std::cout<<distCoeff_<<std::endl;
//  for(size_t i=0;i<distCoeff_.rows;i++)
//  {
//    for(size_t j=0;j<distCoeff_.cols;j++)
//    {
//      std::cout<<*(double*)(distCoeff_.data+i*distCoeff_.step[0]+j*distCoeff_.step[1])<<"\t";
////      std::cout<<distCoeff_.at<float>(i,j)<<"\t";
//    }
//    std::cout<<std::endl;
//  }
  FileStorage fs("computer_camera_intrinsic.xml",FileStorage::WRITE);
  fs << "camera_matrix" << cameraMatrix_;
  fs << "distortion_coefficients" << distCoeff_;
  fs.release();

  IplImage* mapx = cvCreateImage(cvGetSize(image),IPL_DEPTH_32F,1);
  IplImage* mapy = cvCreateImage(cvGetSize(image),IPL_DEPTH_32F,1);
  CvMat cameraMatrix_cv = cameraMatrix_;
  CvMat distCoeff_cv = distCoeff_;

  cvInitUndistortMap(&cameraMatrix_cv,&distCoeff_cv,mapx,mapy);
  cvNamedWindow("Undistort",CV_WINDOW_NORMAL);


  IplImage *image_test = cvLoadImage("/home/zb/BoZhan/OpenCV/Calibration/BMT0610/4.bmp");
  IplImage *t = cvCloneImage(image_test);
  cvShowImage("calibration",image_test);
  cvRemap(t,image_test,mapx,mapy);
  cvReleaseImage(&t);
  cvShowImage("Undistort",image_test);
  waitKey(0);

  return 0;

}



