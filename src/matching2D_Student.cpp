#include <numeric>
#include "matching2D.hpp"

using namespace std;

// For debugging
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch (depth) {
  case CV_8U:  r = "8U"; break;
  case CV_8S:  r = "8S"; break;
  case CV_16U: r = "16U"; break;
  case CV_16S: r = "16S"; break;
  case CV_32S: r = "32S"; break;
  case CV_32F: r = "32F"; break;
  case CV_64F: r = "64F"; break;
  default:     r = "User"; break;
  }

  r += "C";
  r += (chans + '0');

  return r;
}

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    cv::Mat descSourceToUse, descRefToUse;
    bool useConvertedMats = descSource.type() != CV_32F 
      && matcherType.compare(MatcherTypes::MAT_FLANN) == 0 
      && selectorType.compare(SelectorTypes::SEL_KNN) == 0;

    // crude but effective in case we need to do conversions
    if (useConvertedMats)
    { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
      descSource.convertTo(descSourceToUse, CV_32F);
      descRef.convertTo(descRefToUse, CV_32F);
    }
    else {
      descSource.copyTo(descSourceToUse);
      descRef.copyTo(descRefToUse);
    }

    if (matcherType.compare(MatcherTypes::MAT_BF) == 0)
    {
      int normType = descriptorType.compare(DescriptorClasses::DES_BINARY) == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
      matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare(MatcherTypes::MAT_FLANN) == 0)
    {
      matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
      cout << "FLANN matching";
    }
    else {
      assert(false);
    }

    double t = (double)cv::getTickCount();

    // perform matching task
    if (selectorType.compare(SelectorTypes::SEL_NN) == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare(SelectorTypes::SEL_KNN) == 0)
    { // k nearest neighbors (k=2)

      int k = 2;
      vector<vector<cv::DMatch>> knn_matches;
      matcher->knnMatch(descSourceToUse, descRefToUse, knn_matches, k); // finds the 2 best matches

      // filter matches using descriptor distance ratio test
      double minDescDistRatio = 0.8;
      auto new_it = std::remove_if(knn_matches.begin(), knn_matches.end(), 
        [&minDescDistRatio](vector<cv::DMatch>& m) { return m[0].distance >= minDescDistRatio * m[1].distance; });

      knn_matches.erase(new_it, knn_matches.end());

      std::transform(knn_matches.begin(), knn_matches.end(), back_inserter(matches), [](vector<cv::DMatch>& v) {return v[0]; });
    }
    else {
      assert(false);
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << selectorType << " with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;

}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor = createDescriptorOfType(descriptorType);

    if (extractor.empty()) {
      throw std::invalid_argument("Unknown descriptor type");
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
      visualizeKeypoints(keypoints, img, "Shi-Tomasi Corner Detector Results");
    }
}

void detKeypointsHarris(vector<cv::KeyPoint>& keypoints, cv::Mat& img, bool bVis)
{
  // Detector parameters
  int blockSize = 4;     // for every pixel, a blockSize × blockSize neighborhood is considered
  int apertureSize = 5;  // aperture parameter for Sobel operator (must be odd)
  int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
  double k = 0.04;       // Harris parameter (see equation for details)
  // timing
  double t = (double)cv::getTickCount();

  // Detect Harris corners and normalize output
  cv::Mat dst, dst_norm, dst_norm_scaled;
  dst = cv::Mat::zeros(img.size(), CV_32FC1);
  cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
  cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
  cv::convertScaleAbs(dst_norm, dst_norm_scaled);

  // From the solution to an excercise, threshold & NMS
  double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
  for (size_t j = 0; j < dst_norm.rows; j++)
  {
    for (size_t i = 0; i < dst_norm.cols; i++)
    {
      int response = (int)dst_norm.at<float>(j, i);
      if (response > minResponse)
      { // only store points above a threshold

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f(i, j);
        newKeyPoint.size = 2 * apertureSize;
        newKeyPoint.response = response;

        // perform non-maximum suppression (NMS) in local neighbourhood around new key point
        bool bOverlap = false;
        for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
        {
          double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
          if (kptOverlap > maxOverlap)
          {
            bOverlap = true;
            if (newKeyPoint.response > (*it).response)
            {                      // if overlap is >t AND response is higher for new kpt
              *it = newKeyPoint; // replace old key point with new one
              break;             // quit loop over keypoints
            }
          }
        }
        if (!bOverlap)
        {                                     // only add new key point if no overlap has been found in previous NMS
          keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
        }
      }
    } // eof loop over cols
  }     // eof loop over rows
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  cout << "Harris corner detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

  // visualize results
  if (bVis)
  {
    visualizeKeypoints(keypoints, img, "Harris Corner Detector Results");
  }
}

void detKeypointsModern(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img, std::string detectorType, bool bVis) {

  cv::Ptr<cv::FeatureDetector> detector = createDetectorOfType(detectorType);

  if (detector.empty()) {
    throw std::invalid_argument("Unknown detector type");
  }

  std::ostringstream outSs;
  outSs << detectorType << " Keypoint Detector Results";

  double t = (double)cv::getTickCount();
  detector->detect(img, keypoints);
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  cout << outSs.str() << " with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

  if (bVis)
  {
    visualizeKeypoints(keypoints, img, outSs.str());
  }
}