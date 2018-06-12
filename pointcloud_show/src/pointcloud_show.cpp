#include<ros/ros.h>
//#include<pcl/point_cloud.h>
//#include<pcl_conversions/pcl_conversions.h>
//#include <pcl/io/pcd_io.h>
//#include <pcl/point_types.h>
//#include<sensor_msgs/PointCloud2.h>
//#include <pcl/visualization/cloud_viewer.h>

//#include <pcl/visualization/pcl_visualizer.h>
//#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/imgproc.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/core/utility.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/io/io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/image_viewer.h>

#include <iostream>
#include <string>


using namespace cv;
using namespace std;

// 相机内参
//const double camera_factor = 1000;
//const double camera_cx = 325.5;
//const double camera_cy = 253.5;
//const double camera_fx = 518.0;
//const double camera_fy = 519.0;

main (int argc, char **argv)
{
  ros::init (argc, argv, "pcl_create");
  ros::NodeHandle nh;
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
  std::string rgb_img_filename = "left_img.bmp";
  std::string pointcloud_img_filename = "pointcloud_img.bmp";
  std::string PCD_filename = "pointcloud_pcd.pcd";

//  pcl::PCLPointCloud2 cloud_read;
//  pcl::PCDReader reader;
//  reader.readHeader("pointcloud_pcd.pcd",cloud_read);
//  for(int i=0;i<cloud_read.fields.size();i++)
//      std::cout<<"cloud_read field = "<<cloud_read.fields[i].name<<std::endl;
//  std::cout<<"height = "<<cloud_read.height<<std::endl;
//  std::cout<<"width = "<<cloud_read.width<<std::endl;
////  pcl::fromPCLPointCloud2(cloud_read,*cloud_test);
//  if(pcl::io::loadPCDFile<pcl::PointXYZRGBA>("pointcloud_pcd.pcd",*cloud_test)==-1)
//  {
//    PCL_ERROR("couldn't read file test_pcd.pcd\n");
//    exit(0);
//  }


  Mat rgb,depth;
  rgb = imread( rgb_img_filename,-1);
  if(rgb.data==NULL)
  {
      printf("Failed to open file %s\n", rgb_img_filename.c_str());
      return -1;
  }
  depth = imread( pointcloud_img_filename, -1 );
  if(depth.data==NULL)
  {
      printf("Failed to open file %s\n", rgb_img_filename.c_str());
      return -1;
  }

  cout<<"pointcloud img depth= "<<depth.depth()<<endl;
  cout<<"pointcloud img channels= "<<depth.channels()<<endl;
  cout<<"pointcloud img channel type= "<<depth.elemSize1()<<endl;
  cout<<"pointcloud img type= "<<depth.type()<<endl;
  cout<<"rgb img depth= "<<rgb.depth()<<endl;
  cout<<"rgb img channels= "<<rgb.channels()<<endl;
  cout<<"rgb img channel type= "<<rgb.elemSize1()<<endl;
  cout<<"rgb img type= "<<rgb.type()<<endl;
  cout<<"rgb img size = "<<rgb.rows<<"*"<<rgb.cols<<endl;
  namedWindow("point", 1);
  namedWindow("rgb", 1);
  imshow("point",depth);
  imshow("rgb",rgb);
  while(1)
  {

     char c = (char)waitKey(0);
     if( c == 27 || c == 'q' || c == 'Q' )
     break;

  }



  for (int m = 0; m < depth.rows; m++)
  {
      for (int n=0; n < depth.cols; n++)
      {
        pcl::PointXYZRGBA p;

        // 计算这个点的空间坐标
        p.x = double(depth.ptr<uchar>(m)[n*3]);
        p.y = double(depth.ptr<uchar>(m)[n*3+1]);
        p.z = double(depth.ptr<uchar>(m)[n*3+2]);

        // 从rgb图像中获取它的颜色
        // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
        p.b = rgb.ptr<uchar>(m)[n*3];
        p.g = rgb.ptr<uchar>(m)[n*3+1];
        p.r = rgb.ptr<uchar>(m)[n*3+2];

        // 把p加入到点云中
        cloud->points.push_back( p );
      }

  }
  // 设置并保存点云
  cloud->height = rgb.rows;
  cloud->width = rgb.cols;
  cout<<"point cloud size = "<<cloud->points.size()<<endl;
  cloud->is_dense = false;
  pcl::io::savePCDFile<pcl::PointXYZRGBA>( PCD_filename, *cloud );

  pcl::visualization::CloudViewer viewer("pcd viewer");
  viewer.showCloud(cloud);
  pcl::visualization::ImageViewer viewer_img("rgb_viewer");
//  viewer_img.addRGBImage(rgb.ptr(),rgb.cols,rgb.rows);
//  viewer_img.addRGBImage<pcl::PointXYZRGBA>(cloud);
//  viewer_img.showRGBImage(rgb.ptr(),rgb.cols,rgb.rows);
//    viewer_img.showRGBImage(rgb_pcl.ptr(),rgb.cols,rgb.rows);
  viewer_img.showRGBImage<pcl::PointXYZRGBA>(*cloud);
  while(!viewer.wasStopped ())
  {}

  // 清除数据并退出
  cloud->points.clear();
  cout<<"Point cloud saved."<<endl;
  return 0;

}
