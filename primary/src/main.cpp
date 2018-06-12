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


using namespace cv;
int n_boards = 0;
const int board_dt = 20;
int board_w;
int board_h;

int main(int argc,char** argv)
{
  board_w = 9;
  board_h = 6;
  n_boards = 25;
  int board_n = board_h*board_w;
  CvSize board_sz = cvSize(board_w,board_h);
  CvCapture* capture = cvCreateCameraCapture(0);//select which camera
  assert(capture);
  if(capture==NULL)
    std::cout << "Camera capture failed!"<<std::endl;
  double width_ = cvGetCaptureProperty(capture,3);
  double height_ = cvGetCaptureProperty(capture,4);
  double fps_ = cvGetCaptureProperty(capture,5);
  double fourcc_ = cvGetCaptureProperty(capture,6);
  std::cout<<"video width = "<<width_<<"\t"<<"height = "<<height_<<std::endl;
  std::cout<<"video fps = "<<fps_<<"\t"<<"fourcc = "<<fourcc_<<std::endl;

  cvNamedWindow("calibration");
  CvMat* image_points = cvCreateMat(n_boards*board_n,2,CV_32FC1);
  CvMat* object_points = cvCreateMat(n_boards*board_n,3,CV_32FC1);
  CvMat* point_counts = cvCreateMat(n_boards,1,CV_32SC1);
  CvPoint2D32f* corners = new CvPoint2D32f[board_n];
  int corner_count;
  int successes = 0;
  int step,frame = 0;
  IplImage *image = cvQueryFrame(capture);
  std::cout<<"origin of image = "<<image->origin<<std::endl;
  IplImage *gray_image = cvCreateImage(cvGetSize(image),8,1);

  while(successes < n_boards)
  {
    //save image
    std::stringstream stream;
    stream<<successes;
    std::string name_1,name_2,name_3;
    name_1 = stream.str();
    name_2 = ".jpg";
    name_3 = name_1 + name_2;
    cvSaveImage(name_3.c_str(),image);
    if(frame++ % board_dt == 0)
    {
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
          CV_MAT_ELEM(*object_points,float,i,0) = j/board_w;
          CV_MAT_ELEM(*object_points,float,i,1) = j%board_w;
          CV_MAT_ELEM(*object_points,float,i,2) = 0.0f;
        }
        CV_MAT_ELEM(*point_counts,int,successes,0) = board_n;
        successes++;
      }
    }
    char c =cvWaitKey(15);
    if(c == 'p')
    {
      c = 0;
      while(c !='p' && c != 27)
      {
        c = cvWaitKey(250);
      }
    }
    if(c == 27)
      return 0;
      image = cvQueryFrame(capture);
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
    cvNamedWindow("Undistort");
    while (image) {
      IplImage *t = cvCloneImage(image);
      cvShowImage("calibration",image);
      cvRemap(t,image,mapx,mapy);
      cvReleaseImage(&t);
      cvShowImage("Undistort",image);

      int c = waitKey(15);
      if(c == 'p')
      {
        c = 0;
        while(c !='p' && c != 27)
        {
          c = cvWaitKey(250);
        }
      }
      if(c == 27)
        break;
        image = cvQueryFrame(capture);
    }
  return 0;

}



