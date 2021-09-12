#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <tuple>

struct CalibParams {

  // calibration data for camera and lidar

  inline static std::tuple<cv::Mat, cv::Mat, cv::Mat> Initialize() {

    static cv::Mat P_rect_00(3, 4, cv::DataType<double>::type); // 3x4 projection matrix after rectification
    static cv::Mat R_rect_00(4, 4, cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    static cv::Mat RT(4, 4, cv::DataType<double>::type); // rotation matrix and translation vector

    static bool inited(false);

    if (!inited) {

      RT.at<double>(0, 0) = 7.533745e-03; RT.at<double>(0, 1) = -9.999714e-01; RT.at<double>(0, 2) = -6.166020e-04; RT.at<double>(0, 3) = -4.069766e-03;
      RT.at<double>(1, 0) = 1.480249e-02; RT.at<double>(1, 1) = 7.280733e-04; RT.at<double>(1, 2) = -9.998902e-01; RT.at<double>(1, 3) = -7.631618e-02;
      RT.at<double>(2, 0) = 9.998621e-01; RT.at<double>(2, 1) = 7.523790e-03; RT.at<double>(2, 2) = 1.480755e-02; RT.at<double>(2, 3) = -2.717806e-01;
      RT.at<double>(3, 0) = 0.0; RT.at<double>(3, 1) = 0.0; RT.at<double>(3, 2) = 0.0; RT.at<double>(3, 3) = 1.0;

      R_rect_00.at<double>(0, 0) = 9.999239e-01; R_rect_00.at<double>(0, 1) = 9.837760e-03; R_rect_00.at<double>(0, 2) = -7.445048e-03; R_rect_00.at<double>(0, 3) = 0.0;
      R_rect_00.at<double>(1, 0) = -9.869795e-03; R_rect_00.at<double>(1, 1) = 9.999421e-01; R_rect_00.at<double>(1, 2) = -4.278459e-03; R_rect_00.at<double>(1, 3) = 0.0;
      R_rect_00.at<double>(2, 0) = 7.402527e-03; R_rect_00.at<double>(2, 1) = 4.351614e-03; R_rect_00.at<double>(2, 2) = 9.999631e-01; R_rect_00.at<double>(2, 3) = 0.0;
      R_rect_00.at<double>(3, 0) = 0; R_rect_00.at<double>(3, 1) = 0; R_rect_00.at<double>(3, 2) = 0; R_rect_00.at<double>(3, 3) = 1;

      P_rect_00.at<double>(0, 0) = 7.215377e+02; P_rect_00.at<double>(0, 1) = 0.000000e+00; P_rect_00.at<double>(0, 2) = 6.095593e+02; P_rect_00.at<double>(0, 3) = 0.000000e+00;
      P_rect_00.at<double>(1, 0) = 0.000000e+00; P_rect_00.at<double>(1, 1) = 7.215377e+02; P_rect_00.at<double>(1, 2) = 1.728540e+02; P_rect_00.at<double>(1, 3) = 0.000000e+00;
      P_rect_00.at<double>(2, 0) = 0.000000e+00; P_rect_00.at<double>(2, 1) = 0.000000e+00; P_rect_00.at<double>(2, 2) = 1.000000e+00; P_rect_00.at<double>(2, 3) = 0.000000e+00;

      inited = true;
    }
    return std::make_tuple(P_rect_00, R_rect_00, RT);
  }

};