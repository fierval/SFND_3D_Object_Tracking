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
#include "logger.hpp"

//REVIEW: This is a bad practice!
using namespace std;

class Track3d {
public:

  enum class Visualize : int {
    None = 0,
    Detections = 1,
    Lidar = 2,
    TTC = 4,
    All = 7
  };

  Track3d(string& _detectorType, string& _descriptorType, int _bVis = (int)Visualize::None)
    : detectorType(_detectorType)
    , descriptorType(_descriptorType)
    , descriptorClass(getDescriptorClass(_descriptorType))
    , bVis(_bVis)
    , bDataCollection(_bVis == (int)Visualize::None) {

    std::tie(P_rect_00, R_rect_00, RT) = CalibParams::Initialize();
  }


  // Quick & dirty
  static inline std::shared_ptr<CsvLogger<float>> lidar_logger;
  static inline std::shared_ptr<CsvLogger<float>> camera_logger;

  // data location
  static inline string dataPath = "../";

  // log location
  static inline string docPath = dataPath + "doc/";

  // Main loop
  void run_tracking_loop();

private:

  inline string load_image(int imgIndex) {
    // assemble filenames for current index
    ostringstream imgNumber;
    imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
    string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

    // load image from file 
    cv::Mat img = cv::imread(imgFullFilename);

    // push image into data frame buffer
    DataFrame frame;
    frame.cameraImg = img;
    if (imgIndex >= dataBufferSize) {
      // rotate out
      std::rotate(dataBuffer.begin(), dataBuffer.begin() + 1, dataBuffer.end());

      // insert at the end
      *dataBuffer.rbegin() = frame;
    }
    else {
      dataBuffer.push_back(frame);
    }

    return imgNumber.str();
  }

  inline void visualize(BoundingBox* currBB, double ttcLidar, double ttcCamera) {
    cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
    showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
    cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);

    char str[200];
    sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
    putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

    string windowName = "Final Results : TTC";
    cv::namedWindow(windowName, 4);
    cv::imshow(windowName, visImg);
    cout << "Press key to continue to next frame" << endl;
    cv::waitKey(0);

  }


  inline void detectKeypoints(cv::Mat& imgGray, vector<cv::KeyPoint>& keypoints) {

    // Turn off keypoints visualization

    if (detectorType.compare(DetectorTypes::SHITOMASI) == 0)
    {
      detKeypointsShiTomasi(keypoints, imgGray, false);
    }
    else if (detectorType.compare(DetectorTypes::HARRIS) == 0)
    {
      detKeypointsHarris(keypoints, imgGray, false);
    }
    else
    {
      detKeypointsModern(keypoints, imgGray, detectorType, false);
    }
  }

  // camera
  static inline string imgBasePath = dataPath + "images/";
  static inline string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
  static inline string imgFileType = ".png";
  static inline int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
  static inline int imgEndIndex = 18;   // last file index to load
  static inline int imgStepWidth = 1;
  static inline int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

  // object detection
  static inline string yoloBasePath = dataPath + "dat/yolo/";
  static inline string yoloClassesFile = yoloBasePath + "coco.names";
  static inline string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
  static inline string yoloModelWeights = yoloBasePath + "yolov3.weights";

  // Lidar
  static inline string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
  static inline string lidarFileType = ".bin";

  // misc
  static inline double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
  static inline int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time


  static inline string matcherType = MatcherTypes::MAT_BF;        // MAT_BF, MAT_FLANN
  static inline string selectorType = SelectorTypes::SEL_KNN;       // SEL_NN, SEL_KNN

  vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
  string detectorType;
  string descriptorType;
  int bVis;
  bool bDataCollection;

  string descriptorClass;                            // DES_BINARY, DES_HOG

  // calibration data for camera and lidar
  cv::Mat P_rect_00, R_rect_00, RT;
};
