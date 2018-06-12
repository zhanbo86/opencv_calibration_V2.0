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

#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;

#define LEFT_CAM 1
#define RIGHT_CAM 2


int n_boards = 0;
const int board_dt = 20;
int board_w;
int board_h;
float squareSize = 2.33;//cm


static void
StereoCalib(Size boardSize, float squareSize,Mat *cameraMatrix_left,Mat *cameraMatrix_right,Mat *distCoeff_left,Mat *distCoeff_right, bool displayCorners=true, bool useCalibrated=true, bool showRectified=true)
{
    const int maxScale = 1;
    // ARRAY AND VECTOR STORAGE:

    vector<vector<Point2f> > imagePoints[2];
    vector<vector<Point3f> > objectPoints;
    Size imageSize;

    int i, j, k, nimages = 20;

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);

    for( i = j = 0; i < nimages; i++ )
    {
      VideoCapture capture_stero_l(LEFT_CAM);//select left camera
      VideoCapture capture_stero_r(RIGHT_CAM);//select right camera
      Mat img_l,img_r;
      capture_stero_l >> img_l;
      capture_stero_r >> img_r;
      if((img_l.empty())||(img_r.empty()))
      {
            std::cout << "Camera capture_stero failed!"<<std::endl;
            break;
      }

      if( imageSize == Size() )
          imageSize = img_l.size();
      else if( img_l.size() != imageSize )
      {
          cout << "The image has the size different from the first image size. Skipping the pair\n";
          break;
      }
      cvtColor(img_l,img_l,COLOR_BGR2GRAY);
      cvtColor(img_r,img_r,COLOR_BGR2GRAY);
      bool found_l = false;
      bool found_r = false;
      vector<Point2f>& corners_l = imagePoints[0][j];
      vector<Point2f>& corners_r = imagePoints[1][j];
      for( int scale = 1; scale <= maxScale; scale++ )
      {
          Mat timg_l,timg_r;
          if( scale == 1 )
          {
              timg_l = img_l;
              timg_r = img_r;
          }
          else
          {
              resize(img_l, timg_l, Size(), scale, scale);
              resize(img_r, timg_r, Size(), scale, scale);
          }

          found_l = findChessboardCorners(timg_l, boardSize, corners_l,
              CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
          found_r = findChessboardCorners(timg_r, boardSize, corners_r,
              CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
          if( displayCorners )
          {
              Mat cimg_l, cimg1_l,cimg_r,cimg1_r;
              cvtColor(img_l, cimg_l, COLOR_GRAY2BGR);
              cvtColor(img_r, cimg_r, COLOR_GRAY2BGR);
              drawChessboardCorners(cimg_l, boardSize, corners_l, found_l);
              drawChessboardCorners(cimg_r, boardSize, corners_r, found_r);
              double sf = 640./MAX(img_l.rows, img_l.cols);
              resize(cimg_l, cimg1_l, Size(), sf, sf);
              resize(cimg_r, cimg1_r, Size(), sf, sf);
              imshow("left camera corners", cimg1_l);
              imshow("right camera corners", cimg1_r);
              char c = (char)waitKey(500);
              if( c == 27 || c == 'q' || c == 'Q' ) //Allow ESC to quit
                  exit(-1);
          }
          else
              putchar('.');
          if( found_l&&found_r)
          {
              std::stringstream stream;
              stream<<i;
              std::string name_1,name_2_l,name_2_r,name_3_l,name_3_r;
              name_1 = stream.str();
              name_2_l = "_l.jpg";
              name_2_r = "_r.jpg";
              name_3_l = name_1 + name_2_l;
              name_3_r = name_1 + name_2_r;
              imwrite(name_3_l.c_str(),img_l);
              imwrite(name_3_r.c_str(),img_r);

              cornerSubPix(img_l, corners_l, Size(11,11), Size(-1,-1),
                           TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
                                        30, 0.01));
              cornerSubPix(img_r, corners_r, Size(11,11), Size(-1,-1),
                           TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
                                        30, 0.01));
              if( scale > 1 )
              {
                  Mat cornersMat_l(corners_l);
                  Mat cornersMat_r(corners_r);
                  cornersMat_l *= 1./scale;
                  cornersMat_r *= 1./scale;
              }
              j++;
              break;
          }
       }
       if((!found_l)||(!found_r))
       {
         i--;
       }
    }
    cvDestroyAllWindows();
    cout << j << " pairs have been successfully detected.\n";
    cout<<"size of imagePoints[0] = "<<imagePoints[0].size()<<endl;
    cout<<"size of imagePoints[1] = "<<imagePoints[1].size()<<endl;
    nimages = j;
    if( nimages < 2 )
    {
        cout << "Error: too little pairs to run the calibration\n";
        return;
    }

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    objectPoints.resize(nimages);

    for( i = 0; i < nimages; i++ )
    {
        for( j = 0; j < boardSize.height; j++ )
            for( k = 0; k < boardSize.width; k++ )
                objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
    }

    cout << "Running stereo calibration ...\n";

    Mat cameraMatrix[2], distCoeffs[2];
//    cameraMatrix[0] = initCameraMatrix2D(objectPoints,imagePoints[0],imageSize,0);
//    cameraMatrix[1] = initCameraMatrix2D(objectPoints,imagePoints[1],imageSize,0);
    cameraMatrix[0] = *cameraMatrix_left;
    cameraMatrix[1] = *cameraMatrix_right;
    distCoeffs[0] = *distCoeff_left;
    distCoeffs[1] = *distCoeff_right;
    Mat R, T, E, F;

    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
                    cameraMatrix[0], distCoeffs[0],
                    cameraMatrix[1], distCoeffs[1],
                    imageSize, R, T, E, F,
                    CALIB_FIX_ASPECT_RATIO +
                    CALIB_ZERO_TANGENT_DIST +
                    CALIB_USE_INTRINSIC_GUESS +
                    CALIB_SAME_FOCAL_LENGTH +
                    CALIB_RATIONAL_MODEL +
                    CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
                    TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-5) );
    cout << "done with RMS error=" << rms << endl;

// CALIBRATION QUALITY CHECK
// because the output fundamental matrix implicitly
// includes all the output information,
// we can check the quality of calibration using the
// epipolar geometry constraint: m2^t*F*m1=0
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for( i = 0; i < nimages; i++ )
    {
        int npt = (int)imagePoints[0][i].size();
        Mat imgpt[2];
        for( k = 0; k < 2; k++ )
        {
            imgpt[k] = Mat(imagePoints[k][i]);
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
            computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
        }
        for( j = 0; j < npt; j++ )
        {
            double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
                                imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
                           fabs(imagePoints[1][i][j].x*lines[0][j][0] +
                                imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "average epipolar err = " <<  err/npoints << endl;

    // save intrinsic parameters
    FileStorage fs("intrinsics.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
            "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";

    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];

    stereoRectify(cameraMatrix[0], distCoeffs[0],
                  cameraMatrix[1], distCoeffs[1],
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);
    cout<<"stero extrinsics:"<<endl;
    cout<<"R1="<<R1<<endl;
    cout<<"R2="<<R2<<endl;
    cout<<"P1="<<P1<<endl;
    cout<<"P2="<<P2<<endl;
    cout<<"Q="<<Q<<endl;

    fs.open("extrinsics.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
    }
    else
        cout << "Error: can not save the extrinsic parameters\n";

    // OpenCV can handle left-right
    // or up-down camera arrangements
    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

// COMPUTE AND DISPLAY RECTIFICATION
    if( !showRectified )
        return;

    Mat rmap[2][2];
// IF BY CALIBRATED (BOUGUET'S METHOD)
    if( useCalibrated )
    {
        // we already computed everything
    }
// OR ELSE HARTLEY'S METHOD
    else
 // use intrinsic parameters of each camera, but
 // compute the rectification transformation directly
 // from the fundamental matrix
    {
        vector<Point2f> allimgpt[2];
        for( k = 0; k < 2; k++ )
        {
            for( i = 0; i < nimages; i++ )
                std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
        }
        F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
        Mat H1, H2;
        stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

        R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
        R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
        P1 = cameraMatrix[0];
        P2 = cameraMatrix[1];
    }

    //Precompute maps for cv::remap()
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    Mat canvas;
    double sf;
    int w, h;
    if( !isVerticalStereo )
    {
        sf = 600./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h, w*2, CV_8UC3);
    }
    else
    {
        sf = 300./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h*2, w, CV_8UC3);
    }

      Mat rimg_l, rimg_r;
      Mat img_l,img_r;
      img_l = imread("4_l.jpg",0);
      img_r = imread("4_r.jpg",0);
      remap(img_l, rimg_l, rmap[0][0], rmap[0][1], INTER_LINEAR);
      remap(img_r, rimg_r, rmap[1][0], rmap[1][1], INTER_LINEAR);
      imshow("4_l.jpg",rimg_l);
      imshow("4_r.jpg",rimg_r);
      cvtColor(rimg_l, rimg_l, COLOR_GRAY2BGR);
      cvtColor(rimg_r, rimg_r, COLOR_GRAY2BGR);
      Mat canvasPart_l = !isVerticalStereo ? canvas(Rect(0, 0, w, h)) : canvas(Rect(0, 0, w, h));
      Mat canvasPart_r = !isVerticalStereo ? canvas(Rect(w, 0, w, h)) : canvas(Rect(0, h, w, h));
      resize(rimg_l, canvasPart_l, canvasPart_l.size(), 0, 0, INTER_AREA);
      resize(rimg_r, canvasPart_r, canvasPart_r.size(), 0, 0, INTER_AREA);
      if( useCalibrated )
      {
          Rect vroi_l(cvRound(validRoi[0].x*sf), cvRound(validRoi[0].y*sf),
                    cvRound(validRoi[0].width*sf), cvRound(validRoi[0].height*sf));
          Rect vroi_r(cvRound(validRoi[1].x*sf), cvRound(validRoi[1].y*sf),
                    cvRound(validRoi[1].width*sf), cvRound(validRoi[1].height*sf));
          rectangle(canvasPart_l, vroi_l, Scalar(0,0,255), 3, 8);
          rectangle(canvasPart_r, vroi_r, Scalar(0,0,255), 3, 8);
      }

      if( !isVerticalStereo )
          for( j = 0; j < canvas.rows; j += 16 )
              line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
      else
          for( j = 0; j < canvas.cols; j += 16 )
              line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
      imshow("rectified", canvas);
      char c = (char)waitKey();
      while (1)
      {
        if( c == 27 || c == 'q' || c == 'Q' )
            break;
      }


}


int main(int argc,char** argv)
{
  board_w = 8;
  board_h = 6;
  n_boards = 25;
  int board_n = board_h*board_w;
  CvSize board_sz = cvSize(board_w,board_h);

  //--------------------------------------------calibrate left camera-----------------------------------------------
  CvCapture* capture_l = cvCreateCameraCapture(LEFT_CAM);//select which camera
  assert(capture_l);
  if(capture_l==NULL)
    std::cout << "Camera capture left failed!"<<std::endl;
  double width_ = cvGetCaptureProperty(capture_l,3);
  double height_ = cvGetCaptureProperty(capture_l,4);
  double fps_ = cvGetCaptureProperty(capture_l,5);
  double fourcc_ = cvGetCaptureProperty(capture_l,6);
  std::cout<<"video width = "<<width_<<"\t"<<"height = "<<height_<<std::endl;
  std::cout<<"video fps = "<<fps_<<"\t"<<"fourcc = "<<fourcc_<<std::endl;

  cvNamedWindow("calibration_left");
  CvMat* image_points_l = cvCreateMat(n_boards*board_n,2,CV_32FC1);
  CvMat* object_points_l = cvCreateMat(n_boards*board_n,3,CV_32FC1);
  CvMat* point_counts_l = cvCreateMat(n_boards,1,CV_32SC1);
  CvPoint2D32f* corners_l = new CvPoint2D32f[board_n];
  int corner_count;
  int successes = 0;
  int step,frame = 0;
  IplImage *image = cvQueryFrame(capture_l);
  std::cout<<"origin of image = "<<image->origin<<std::endl;
  IplImage *gray_image = cvCreateImage(cvGetSize(image),8,1);

  while(successes < n_boards)
  {
    //save image
//    std::stringstream stream;
//    stream<<successes;
//    std::string name_1,name_2,name_3;
//    name_1 = stream.str();
//    name_2 = "_l.jpg";
//    name_3 = name_1 + name_2;
//    cvSaveImage(name_3.c_str(),image);
    if(frame++ % board_dt == 0)
    {
      int found = cvFindChessboardCorners(image,board_sz,corners_l,&corner_count,
                                          CV_CALIB_CB_ADAPTIVE_THRESH|CV_CALIB_CB_FILTER_QUADS);
      std::cout<<"found = "<<found<<std::endl;
      cvCvtColor(image,gray_image,CV_BGR2GRAY);
      cvFindCornerSubPix(gray_image,corners_l,corner_count,cvSize(11,11),cvSize(-1,-1),
                         cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER,30,0.1));
      cvDrawChessboardCorners(image,board_sz,corners_l,corner_count,found);
      cvShowImage("calibration_left",image);
//      cout<<"corners_l[0] = "<<corners_l[0].x <<"\t"<<corners_l[0].y<<endl;
//      cout<<"corners_l[1] = "<<corners_l[1].x <<"\t"<<corners_l[1].y<<endl;
//      cout<<"corners_l[2] = "<<corners_l[2].x <<"\t"<<corners_l[2].y<<endl;

      if(corner_count == board_n)
      {
        step = successes*board_n;
        for(int i =step,j=0,k=0;j<board_n;++i,++j,++k)
        {
          CV_MAT_ELEM(*image_points_l,float,i,0) = corners_l[j].x;
          CV_MAT_ELEM(*image_points_l,float,i,1) = corners_l[j].y;
//          CV_MAT_ELEM(*object_points_l,float,i,0) = j/board_w;
//          CV_MAT_ELEM(*object_points_l,float,i,1) = j%board_w;
          CV_MAT_ELEM(*object_points_l,float,i,0) = j%board_w;
          CV_MAT_ELEM(*object_points_l,float,i,1) = j/board_w;
          CV_MAT_ELEM(*object_points_l,float,i,2) = 0.0f;
        }
        CV_MAT_ELEM(*point_counts_l,int,successes,0) = board_n;
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
      image = cvQueryFrame(capture_l);
  }

  std::cout<<"successes = "<<successes<<std::endl;

  Mat cameraMatrix_left(3,3,CV_32FC1,Scalar(0)), distCoeff_left(1,5,CV_32FC1,Scalar(0));
  std::vector<Mat> seqRotation_l, seqTranslation_l;
  cv::Point3f point3D_l;
  cv::Point2f point2D_l;
  std::vector<cv::Point3f> objectPoints_l;
  std::vector<cv::Point2f> corners_temp_l;
  std::vector<std::vector<cv::Point3f> > seqObjectPoints_l;
  std::vector<std::vector<cv::Point2f> > seqCorners_l;
  for ( int t=0; t<successes; t++ )
  {
      objectPoints_l.clear();
      corners_temp_l.clear();
      for ( int i=0; i<board_n; i++ )
      {
          point3D_l.x = CV_MAT_ELEM(*object_points_l,float,t*board_n+i,0);
          point3D_l.y = CV_MAT_ELEM(*object_points_l,float,t*board_n+i,1);
          point3D_l.z = 0;
          point2D_l.x = CV_MAT_ELEM(*image_points_l,float,t*board_n+i,0);
          point2D_l.y = CV_MAT_ELEM(*image_points_l,float,t*board_n+i,1);
          objectPoints_l.push_back(point3D_l);
          corners_temp_l.push_back(point2D_l);
      }
      seqObjectPoints_l.push_back(objectPoints_l);
      seqCorners_l.push_back(corners_temp_l);
  }


  cvReleaseMat(&object_points_l);
  cvReleaseMat(&image_points_l);
  cvReleaseMat(&point_counts_l);

  cv::Size image_size;
  image_size = cv::Size(image->width,image->height);
  double re_pre_err;
  re_pre_err = calibrateCamera(seqObjectPoints_l,seqCorners_l,image_size,
                               cameraMatrix_left,distCoeff_left,
                               seqRotation_l,seqTranslation_l,CV_CALIB_FIX_K3|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);//test ok,result is the same as Matlab
                               //ROS:CV_CALIB_FIX_K3|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5;
                               //Opencv:CV_CALIB_FIX_K4|CV_CALIB_FIX_K5|CV_CALIB_ZERO_TANGENT_DIST,test bad because k3 doesnot set zero
//  cvCalibrateCamera2(object_points2,image_points2,
//                     point_counts2,cvGetSize(image),
//                     intrinsic_matrix,distortion_coeffs,
//                     NULL,NULL,0);
  std::cout<<"re_prejection error = "<<re_pre_err<<std::endl;
  std::cout<<"cameraMatrix_left = "<<std::endl;
  std::cout<<cameraMatrix_left<<std::endl;
  std::cout<<"distCoeff_left = "<<std::endl;
  std::cout<<distCoeff_left<<std::endl;
//  for(size_t i=0;i<distCoeff_.rows;i++)
//  {
//    for(size_t j=0;j<distCoeff_.cols;j++)
//    {
//      std::cout<<*(double*)(distCoeff_.data+i*distCoeff_.step[0]+j*distCoeff_.step[1])<<"\t";
////      std::cout<<distCoeff_.at<float>(i,j)<<"\t";
//    }
//    std::cout<<std::endl;
//  }
    FileStorage fs_l("computer_camera_intrinsic_left.xml",FileStorage::WRITE);
    fs_l << "camera_matrix_left" << cameraMatrix_left;
    fs_l << "distortion_coefficients_left" << distCoeff_left;
    fs_l.release();
    cvDestroyWindow("calibration_left");
    cvReleaseImage(&gray_image);
    cvReleaseCapture(&capture_l);

//-------------------------------------calibrate right camera---------------------------------------------
    CvCapture* capture_r = cvCreateCameraCapture(RIGHT_CAM);//select right camera
    assert(capture_r);
    if(capture_r==NULL)
      std::cout << "Camera capture right failed!"<<std::endl;
     width_ = cvGetCaptureProperty(capture_r,3);
     height_ = cvGetCaptureProperty(capture_r,4);
     fps_ = cvGetCaptureProperty(capture_r,5);
     fourcc_ = cvGetCaptureProperty(capture_r,6);
    std::cout<<"video width = "<<width_<<"\t"<<"height = "<<height_<<std::endl;
    std::cout<<"video fps = "<<fps_<<"\t"<<"fourcc = "<<fourcc_<<std::endl;

    cvNamedWindow("calibration_right");
    CvMat* image_points_r = cvCreateMat(n_boards*board_n,2,CV_32FC1);
    CvMat* object_points_r = cvCreateMat(n_boards*board_n,3,CV_32FC1);
    CvMat* point_counts_r = cvCreateMat(n_boards,1,CV_32SC1);
    CvPoint2D32f* corners_r = new CvPoint2D32f[board_n];
    corner_count = 0;
    successes = 0;
    frame = 0;
    step = 0;
    image = cvQueryFrame(capture_r);
    std::cout<<"origin of image = "<<image->origin<<std::endl;
    IplImage *gray_image_ = cvCreateImage(cvGetSize(image),8,1);

    while(successes < n_boards)
    {
      //save image
//      std::stringstream stream;
//      stream<<successes;
//      std::string name_1,name_2,name_3;
//      name_1 = stream.str();
//      name_2 = "_r.jpg";
//      name_3 = name_1 + name_2;
//      cvSaveImage(name_3.c_str(),image);
      if(frame++ % board_dt == 0)
      {
        int found = cvFindChessboardCorners(image,board_sz,corners_r,&corner_count,
                                            CV_CALIB_CB_ADAPTIVE_THRESH|CV_CALIB_CB_FILTER_QUADS);
        std::cout<<"found = "<<found<<std::endl;
        cvCvtColor(image,gray_image_,CV_BGR2GRAY);
        cvFindCornerSubPix(gray_image_,corners_r,corner_count,cvSize(11,11),cvSize(-1,-1),
                           cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER,30,0.1));
        cvDrawChessboardCorners(image,board_sz,corners_r,corner_count,found);
        cvShowImage("calibration_right",image);

        if(corner_count == board_n)
        {
          step = successes*board_n;
          for(int i =step,j=0,k=0;j<board_n;++i,++j,++k)
          {
            CV_MAT_ELEM(*image_points_r,float,i,0) = corners_r[j].x;
            CV_MAT_ELEM(*image_points_r,float,i,1) = corners_r[j].y;
//            CV_MAT_ELEM(*object_points_r,float,i,0) = j/board_w;
//            CV_MAT_ELEM(*object_points_r,float,i,1) = j%board_w;
            CV_MAT_ELEM(*object_points_r,float,i,0) = j%board_w;
            CV_MAT_ELEM(*object_points_r,float,i,1) = j/board_w;
            CV_MAT_ELEM(*object_points_r,float,i,2) = 0.0f;
          }
          CV_MAT_ELEM(*point_counts_r,int,successes,0) = board_n;
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
        image = cvQueryFrame(capture_r);
    }

    std::cout<<"successes = "<<successes<<std::endl;

    Mat cameraMatrix_right(3,3,CV_32FC1,Scalar(0)), distCoeff_right(1,5,CV_32FC1,Scalar(0));
    std::vector<Mat> seqRotation_r, seqTranslation_r;
    cv::Point3f point3D_r;
    cv::Point2f point2D_r;
    std::vector<cv::Point3f> objectPoints_r;
    std::vector<cv::Point2f> corners_temp_r;
    std::vector<std::vector<cv::Point3f> > seqObjectPoints_r;
    std::vector<std::vector<cv::Point2f> > seqCorners_r;

    for ( int t=0; t<successes; t++ )
    {
        objectPoints_r.clear();
        corners_temp_r.clear();
        for ( int i=0; i<board_n; i++ )
        {
            point3D_r.x = CV_MAT_ELEM(*object_points_r,float,t*board_n+i,0);
            point3D_r.y = CV_MAT_ELEM(*object_points_r,float,t*board_n+i,1);
            point3D_r.z = 0;
            point2D_r.x = CV_MAT_ELEM(*image_points_r,float,t*board_n+i,0);
            point2D_r.y = CV_MAT_ELEM(*image_points_r,float,t*board_n+i,1);
            objectPoints_r.push_back(point3D_r);
            corners_temp_r.push_back(point2D_r);
        }
        seqObjectPoints_r.push_back(objectPoints_r);
        seqCorners_r.push_back(corners_temp_r);
    }


    cvReleaseMat(&object_points_r);
    cvReleaseMat(&image_points_r);
    cvReleaseMat(&point_counts_r);

    image_size = cv::Size(image->width,image->height);
    re_pre_err = calibrateCamera(seqObjectPoints_r,seqCorners_r,image_size,
                                 cameraMatrix_right,distCoeff_right,
                                 seqRotation_r,seqTranslation_r,CV_CALIB_FIX_K3|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);//test ok,result is the same as Matlab
                                 //ROS:CV_CALIB_FIX_K3|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5;
                                 //Opencv:CV_CALIB_FIX_K4|CV_CALIB_FIX_K5|CV_CALIB_ZERO_TANGENT_DIST,test bad because k3 doesnot set zero
  //  cvCalibrateCamera2(object_points2,image_points2,
  //                     point_counts2,cvGetSize(image),
  //                     intrinsic_matrix,distortion_coeffs,
  //                     NULL,NULL,0);
    std::cout<<"re_prejection error = "<<re_pre_err<<std::endl;
    std::cout<<"cameraMatrix_right = "<<std::endl;
    std::cout<<cameraMatrix_right<<std::endl;
    std::cout<<"distCoeff_right = "<<std::endl;
    std::cout<<distCoeff_right<<std::endl;
  //  for(size_t i=0;i<distCoeff_.rows;i++)
  //  {
  //    for(size_t j=0;j<distCoeff_.cols;j++)
  //    {
  //      std::cout<<*(double*)(distCoeff_.data+i*distCoeff_.step[0]+j*distCoeff_.step[1])<<"\t";
  ////      std::cout<<distCoeff_.at<float>(i,j)<<"\t";
  //    }
  //    std::cout<<std::endl;
  //  }
      FileStorage fs_r("computer_camera_intrinsic_right.xml",FileStorage::WRITE);
      fs_r << "camera_matrix_right" << cameraMatrix_right;
      fs_r << "distortion_coefficients_right" << distCoeff_right;
      fs_r.release();
      cvDestroyWindow("calibration_right");
      cvReleaseCapture(&capture_r);

//--------------------------------------------------------stero calibrate--------------------------------------------------------
      Size boardSize;
      bool showRectified = true;
      boardSize.width = board_w;
      boardSize.height = board_h;


//      FileStorage fs_left("computer_camera_intrinsic_left.xml", FileStorage::READ);
//      if(!fs_left.isOpened())
//      {
//          printf("Failed to open file %s\n", "computer_camera_intrinsic_left.xml");
//          return -1;
//      }

//      Mat M1, D1, M2, D2;
//      fs_left["camera_matrix_left"]>> M1;
//      fs_left["distortion_coefficients_left"]>> D1;
//      fs_left.release();

//      FileStorage fs_right("computer_camera_intrinsic_right.xml", FileStorage::READ);
//      if(!fs_right.isOpened())
//      {
//          printf("Failed to open file %s\n", "computer_camera_intrinsic_right.xml");
//          return -1;
//      }
//      fs_right["camera_matrix_right"]>> M2;
//      fs_right["distortion_coefficients_right"]>> D2;
//      fs_right.release();

////      cout<<"M1 = "<<M1<<endl;
//      StereoCalib(boardSize, squareSize,&M1,&M2,&D1,&D2, true, true, showRectified);

      StereoCalib(boardSize, squareSize,&cameraMatrix_left,&cameraMatrix_right,&distCoeff_left,&distCoeff_right, true, true, showRectified);

  return 0;

}



