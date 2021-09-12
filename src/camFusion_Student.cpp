
#include <iostream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox>& boundingBoxes, std::vector<LidarPoint>& lidarPoints, float shrinkFactor, cv::Mat& P_rect_xx, cv::Mat& R_rect_xx, cv::Mat& RT)
{
  // loop over all Lidar points and associate them to a 2D bounding box
  cv::Mat X(4, 1, cv::DataType<double>::type);
  cv::Mat Y(3, 1, cv::DataType<double>::type);

  for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
  {
    // assemble vector for matrix-vector-multiplication
    X.at<double>(0, 0) = it1->x;
    X.at<double>(1, 0) = it1->y;
    X.at<double>(2, 0) = it1->z;
    X.at<double>(3, 0) = 1;

    // project Lidar point into camera
    Y = P_rect_xx * R_rect_xx * RT * X;
    cv::Point pt;
    // pixel coordinates
    pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
    pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

    vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
    for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
    {
      // shrink current bounding box slightly to avoid having too many outlier points around the edges
      cv::Rect smallerBox;
      smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
      smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
      smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
      smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

      // check wether point is within current bounding box
      if (smallerBox.contains(pt))
      {
        enclosingBoxes.push_back(it2);
      }

    } // eof loop over all bounding boxes

    // check wether point has been enclosed by one or by multiple boxes
    if (enclosingBoxes.size() == 1)
    {
      // add Lidar point to bounding box
      enclosingBoxes[0]->lidarPoints.push_back(*it1);
    }

  } // eof loop over all Lidar points
}

/*
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size.
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox>& boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
  // create topview image
  cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

  for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
  {
    // create randomized color for current 3D object
    cv::RNG rng(it1->boxID);
    cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

    // plot Lidar points into top view image
    int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
    float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
    for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
    {
      // world coordinates
      float xw = (*it2).x; // world position in m with x facing forward from sensor
      float yw = (*it2).y; // world position in m with y facing left from sensor
      xwmin = xwmin < xw ? xwmin : xw;
      ywmin = ywmin < yw ? ywmin : yw;
      ywmax = ywmax > yw ? ywmax : yw;

      // top-view coordinates
      int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
      int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

      // find enclosing rectangle
      top = top < y ? top : y;
      left = left < x ? left : x;
      bottom = bottom > y ? bottom : y;
      right = right > x ? right : x;

      // draw individual point
      cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
    }

    // draw enclosing rectangle
    cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

    // augment object with some key data
    char str1[200], str2[200];
    sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
    putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
    sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
    putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
  }

  // plot distance markers
  float lineSpacing = 2.0; // gap between distance markers
  int nMarkers = floor(worldSize.height / lineSpacing);
  for (size_t i = 0; i < nMarkers; ++i)
  {
    int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
    cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
  }

  // display image
  string windowName = "3D Objects";
  cv::namedWindow(windowName, 1);
  cv::imshow(windowName, topviewImg);

  if (bWait)
  {
    cv::waitKey(0); // wait for key to be pressed
  }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox& boundingBox, std::vector<cv::KeyPoint>& kptsPrev, std::vector<cv::KeyPoint>& kptsCurr, std::vector<cv::DMatch>& kptMatches)
{
  //calculatig the mean kptMatches
  std::vector<float> dist;

  // matches contained within current roi
  std::vector<cv::DMatch> relevant;
  
  std::copy_if(kptMatches.begin(), kptMatches.end(), std::back_inserter(relevant), [&kptsCurr, &boundingBox](cv::DMatch& m) {
    return boundingBox.roi.contains(kptsCurr[m.trainIdx].pt); });

  std::transform(relevant.begin(), relevant.end(), std::back_inserter(dist), [&kptsPrev, &kptsCurr](cv::DMatch& m) {
    return cv::norm(kptsCurr[m.trainIdx].pt - kptsPrev[m.queryIdx].pt); });

  // shouldn't be?.. just in case
  if (dist.empty()) {
    return;
  }

  float distanceMean = std::accumulate(dist.begin(), dist.end(), 0.0) / dist.size();

  // no need for hit-test inside this loop
  // distances have been memoized as well
  for (int i = 0; i < relevant.size(); i++) {
    if (dist[i] < distanceMean * 1.5) {

      cv::DMatch& match = relevant[i];
      cv::KeyPoint currKeyPt = kptsCurr[match.trainIdx];
      boundingBox.kptMatches.push_back(match);
      boundingBox.keypoints.push_back(currKeyPt);
    }

  }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
// Straight from the excercise.
void computeTTCCamera(std::vector<cv::KeyPoint>& kptsPrev, std::vector<cv::KeyPoint>& kptsCurr,
  std::vector<cv::DMatch> kptMatches, double frameRate, double& TTC, cv::Mat* visImg)
{
  // compute distance ratios between all matched keypoints
  vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
  for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
  { // outer kpt. loop

      // get current keypoint and its matched partner in the prev. frame
    cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
    cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

    for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
    { // inner kpt.-loop

      double minDist = 100.0; // min. required distance

      // get next keypoint and its matched partner in the prev. frame
      cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
      cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

      // compute distances and distance ratios
      double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
      double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

      if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
      { // avoid division by zero

        double distRatio = distCurr / distPrev;
        distRatios.push_back(distRatio);
      }
    } // eof inner loop over all matched kpts
  }     // eof outer loop over all matched kpts

  // only continue if list of distance ratios is not empty
  if (distRatios.size() == 0)
  {
    TTC = std::numeric_limits<double>::infinity();
    return;
  }


  // STUDENT TASK (replacement for meanDistRatio)
  std::sort(distRatios.begin(), distRatios.end());
  long medIndex = floor(distRatios.size() / 2.0);

  // compute median dist. ratio to remove outlier influence
  double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex];

  float dT = 1 / frameRate;
  TTC = -dT / (1 - medDistRatio);
  // EOF STUDENT TASK

}


double get_median_lidar_point(std::vector<LidarPoint>& points) {

  float halfLane = 2.0; //assumed width of the ego lane: 4.0

  std::vector<LidarPoint> relevant;
  std::copy_if(points.begin(), points.end(), std::back_inserter(relevant), [&halfLane](LidarPoint& p) { return abs(p.y) <= halfLane; });

  if (relevant.empty()) {
    return std::numeric_limits<double>::infinity();
  }

  std::sort(relevant.begin(), relevant.end(), [](LidarPoint a, LidarPoint b) {return a.x < b.x; });

  int medIndex = relevant.size() >> 1;

  // compute median dist. ratio to remove outlier influence
  return (relevant.size() & 1) == 0 ? (relevant[medIndex - 1].x + relevant[medIndex].x) / 2.0 : relevant[medIndex].x;

}

void computeTTCLidar(std::vector<LidarPoint>& lidarPointsPrev,
  std::vector<LidarPoint>& lidarPointsCurr, double frameRate, double& TTC)
{
  if (lidarPointsPrev.empty() || lidarPointsCurr.empty()) {
    TTC = NAN;
    return;
  }

  float dt = 1.0 / frameRate; 

  double d0 = get_median_lidar_point(lidarPointsPrev);
  double d1 = get_median_lidar_point(lidarPointsCurr);

  TTC = d1 * dt / (d0 - d1);
}


void matchBoundingBoxes(std::vector<cv::DMatch>& matches, std::unordered_map<int, int>& bbBestMatches, DataFrame& prevFrame, DataFrame& currFrame)
{
  auto& prev_bboxes = prevFrame.boundingBoxes;
  auto& cur_bboxes = currFrame.boundingBoxes;

  // map of all hit tests
  // TODO: This is one ugly data structure!
  std::unordered_map<int, std::unordered_map<int, int>> all_matches;

  for (auto& m : matches) {
    cv::KeyPoint prevPt = prevFrame.keypoints[m.queryIdx];
    cv::KeyPoint curPt = currFrame.keypoints[m.trainIdx];

    for (int idx_prev = 0; idx_prev < prev_bboxes.size(); idx_prev++) {

      BoundingBox& prev_bb = prev_bboxes[idx_prev];
      if (!prev_bb.roi.contains(prevPt.pt)) {
        continue;
      }
      // now loop over current matches and find the bounding box
      else {
        if (all_matches.count(idx_prev) == 0) {
          all_matches.insert(std::make_pair(idx_prev, std::unordered_map<int, int>()));
        }

        auto& cur_map = all_matches[idx_prev];

        for (int idx_cur = 0; idx_cur < cur_bboxes.size(); idx_cur++) {

          BoundingBox& cur_bb = cur_bboxes[idx_cur];
          if (!cur_bb.roi.contains(curPt.pt)) {
            continue;
          }

          // and create a pair
          if (cur_map.count(idx_cur) == 0) {
            cur_map.insert(std::make_pair(idx_cur, 0));
          }
          cur_map[idx_cur]++;
        }
      }
    }
  }

  // for each prev bbox, find the best match and store in the return structure
  std::transform(all_matches.begin(), all_matches.end(), std::inserter(bbBestMatches, bbBestMatches.end()), [&bbBestMatches](std::pair<int, std::unordered_map<int, int>> cur_matches) {

    // find the box with the maximum number of matches
    std::pair<int, int> best_match = *std::max_element(cur_matches.second.begin(), cur_matches.second.end(),
      [](std::pair<int, int> e1, std::pair<int, int> e2) {return e1.second < e2.second; });

    return std::make_pair(cur_matches.first, best_match.first);

    });
}
