
#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <map>
#include <opencv2/core.hpp>

struct LidarPoint { // single lidar point in space
    double x,y,z,r; // x,y,z in [m], r is point reflectivity
};

struct BoundingBox { // bounding box around a classified object (contains both 2D and 3D data)
    
    int boxID; // unique identifier for this bounding box
    int trackID; // unique identifier for the track to which this bounding box belongs
    
    cv::Rect roi; // 2D region-of-interest in image coordinates
    int classID; // ID based on class file provided to YOLO framework
    double confidence; // classification trust

    std::vector<LidarPoint> lidarPoints; // Lidar 3D points which project into 2D image roi
    std::vector<cv::KeyPoint> keypoints; // keypoints enclosed by 2D roi
    std::vector<cv::DMatch> kptMatches; // keypoint matches enclosed by 2D roi
};

struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
    std::vector<LidarPoint> lidarPoints;

    std::vector<BoundingBox> boundingBoxes; // ROI around detected objects in 2D image coordinates
    std::map<int,int> bbMatches; // bounding box matches between previous and current frame
};

struct DetectorTypes {
  constexpr static char SHITOMASI[] = "SHITOMASI";
  constexpr static char HARRIS[] = "HARRIS";
  constexpr static char FAST[] = "FAST";
  constexpr static char BRISK[] = "BRISK";
  constexpr static char ORB[] = "ORB";
  constexpr static char AKAZE[] = "AKAZE";
  constexpr static char SIFT[] = "SIFT";
};

struct DescriptorTypes {
  constexpr static char FREAK[] = "FREAK";
  constexpr static char BRIEF[] = "BRIEF";
  constexpr static char BRISK[] = "BRISK";
  constexpr static char ORB[] = "ORB";
  constexpr static char AKAZE[] = "AKAZE";
  constexpr static char SIFT[] = "SIFT";
};

struct DescriptorClasses {
  constexpr static char DES_BINARY[] = "DES_BINARY";
  constexpr static char DES_HOG[] = "DES_HOG";
};

struct SelectorTypes {
  constexpr static char SEL_NN[] = "SEL_NN";
  constexpr static char SEL_KNN[] = "SEL_KNN";
};

struct MatcherTypes {
  constexpr static char MAT_BF[] = "MAT_BF";
  constexpr static char MAT_FLANN[] = "MAT_FLANN";
};

#endif /* dataStructures_h */
