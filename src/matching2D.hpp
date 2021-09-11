#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"


void detKeypointsHarris(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img, bool bVis = false);
void detKeypointsShiTomasi(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img, bool bVis = false);
void detKeypointsModern(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img, std::string detectorType, bool bVis = false);
void descKeypoints(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img, cv::Mat& descriptors, std::string descriptorType);
void matchDescriptors(std::vector<cv::KeyPoint>& kPtsSource, std::vector<cv::KeyPoint>& kPtsRef, cv::Mat& descSource, cv::Mat& descRef,
  std::vector<cv::DMatch>& matches, std::string descriptorType, std::string matcherType, std::string selectorType);

inline void visualizeKeypoints(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img, const std::string& windowName) {
  cv::Mat visImage = img.clone();
  cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::namedWindow(windowName, 6);
  imshow(windowName, visImage);
  cv::waitKey(0);
}

inline cv::Ptr<cv::FeatureDetector> createDetectorOfType(const std::string& detType) {

  if (detType.compare(DetectorTypes::AKAZE) == 0) {
    return cv::AKAZE::create();
  }
  if (detType.compare(DetectorTypes::BRISK) == 0) {
    return cv::BRISK::create();
  }
  if (detType.compare(DetectorTypes::ORB) == 0) {
    return cv::ORB::create();
  }
  if (detType.compare(DetectorTypes::SIFT) == 0) {
    return cv::SIFT::create();
  }
  if (detType.compare(DetectorTypes::FAST) == 0) {
    int threshold = 30;                                                              // difference between intensity of the central pixel and pixels of a circle around this pixel
    bool bNMS = true;                                                                // perform non-maxima suppression on keypoints
    cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
    return cv::FastFeatureDetector::create(threshold, bNMS, type);
  }


  assert(false);
  return nullptr;
}

inline cv::Ptr<cv::DescriptorExtractor> createDescriptorOfType(const std::string& detType) {

  int threshold = 30;        // FAST/AGAST detection threshold score.
  int octaves = 3;           // detection octaves (use 0 to do single scale)
  float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
  float contrastThreshold = 0.4f;

  if (detType.compare(DescriptorTypes::AKAZE) == 0) {
    float akaze_threshold = 0.001f;
    return cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, threshold);
  }
  if (detType.compare(DescriptorTypes::BRISK) == 0) {
    return cv::BRISK::create(threshold, octaves, patternScale);
  }
  if (detType.compare(DescriptorTypes::ORB) == 0) {
    return cv::ORB::create();
  }
  if (detType.compare(DescriptorTypes::SIFT) == 0) {
    return cv::SIFT::create(threshold, octaves, contrastThreshold);
  }
  if (detType.compare(DescriptorTypes::BRIEF) == 0) {
    return cv::xfeatures2d::BriefDescriptorExtractor::create();
  }
  if (detType.compare(DescriptorTypes::FREAK) == 0) {
    return cv::xfeatures2d::FREAK::create();
  }

  assert(false);
  return nullptr;
}

inline const std::string getDescriptorClass(const std::string& descType) {

  if (descType.compare(DescriptorTypes::SIFT) == 0) {
    return DescriptorClasses::DES_HOG;
  }

  // be quick about it, although this should probably check for invalid type
  return DescriptorClasses::DES_BINARY;
}
#endif /* matching2D_hpp */
