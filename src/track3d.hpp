#pragma once

#include "calibration.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

//REVIEW: This is a bad practice!
using namespace std;

class Track3d {
public:

  Track3d(string& _detectorType, string& _descriptorType, bool _bVis = false)
    : detectorType(_detectorType)
    , descriptorType(_descriptorType)
    , bVis(_bVis) {

    descriptorClass = getDescriptorClass(descriptorType);
    std::tie(P_rect_00, R_rect_00, RT) = CalibParams::Initialize();
  }


  inline void visualize(BoundingBox* currBB, double ttcLidar, double ttcCamera) {
    cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
    showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
    cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);

    char str[200];
    sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
    putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255));

    string windowName = "Final Results : TTC";
    cv::namedWindow(windowName, 4);
    cv::imshow(windowName, visImg);
    cout << "Press key to continue to next frame" << endl;
    cv::waitKey(0);

  }

  // Main loop
  void run_tracking_loop();

private:

  inline void detectKeypoints(cv::Mat& imgGray, vector<cv::KeyPoint>& keypoints) {
    if (detectorType.compare(DetectorTypes::SHITOMASI) == 0)
    {
      detKeypointsShiTomasi(keypoints, imgGray, bVis);
    }
    else if (detectorType.compare(DetectorTypes::HARRIS) == 0)
    {
      detKeypointsHarris(keypoints, imgGray, bVis);
    }
    else
    {
      detKeypointsModern(keypoints, imgGray, detectorType, bVis);
    }
  }

  // data location
  string dataPath = "../";

  // camera
  string imgBasePath = dataPath + "images/";
  string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
  string imgFileType = ".png";
  int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
  int imgEndIndex = 18;   // last file index to load
  int imgStepWidth = 1;
  int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

  // object detection
  string yoloBasePath = dataPath + "dat/yolo/";
  string yoloClassesFile = yoloBasePath + "coco.names";
  string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
  string yoloModelWeights = yoloBasePath + "yolov3.weights";

  // Lidar
  string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
  string lidarFileType = ".bin";

  // calibration data for camera and lidar
  cv::Mat P_rect_00, R_rect_00, RT;
  // misc
  double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
  int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
  vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time

  string detectorType;
  string descriptorType;
  bool bVis;

  string matcherType = MatcherTypes::MAT_BF;        // MAT_BF, MAT_FLANN
  string descriptorClass;                            // DES_BINARY, DES_HOG
  string selectorType = SelectorTypes::SEL_KNN;       // SEL_NN, SEL_KNN

};
