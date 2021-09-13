#include "track3d.hpp"
using namespace std;

void Track3d::run_tracking_loop()
{

  /* MAIN LOOP OVER ALL IMAGES */

  for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex += imgStepWidth)
  {
    /* LOAD IMAGE INTO BUFFER */

    string imgNumber = load_image(imgIndex);

    cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;


    /* DETECT & CLASSIFY OBJECTS */

    float confThreshold = 0.2f;
    float nmsThreshold = 0.4f;
    detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
      yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis & (int)Visualize::Detections);

    cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;


    /* CROP LIDAR POINTS */

    // load 3D Lidar points from file
    string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber + lidarFileType;
    std::vector<LidarPoint> lidarPoints;
    loadLidarFromFile(lidarPoints, lidarFullFilename);

    // remove Lidar points based on distance properties
    float minZ = -1.5f, maxZ = -0.9f, minX = 2.0f, maxX = 20.0f, maxY = 2.0f, minR = 0.1f; // focus on ego lane
    cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);

    (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

    cout << "#3 : CROP LIDAR POINTS done" << endl;


    /* CLUSTER LIDAR POINT CLOUD */

    // associate Lidar points with camera-based ROI
    float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
    clusterLidarWithROI((dataBuffer.end() - 1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

    // Visualize 3D objects
    if (bVis & (int)Visualize::Lidar)
    {
      show3DObjects((dataBuffer.end() - 1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);
    }

    cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;

    /* DETECT IMAGE KEYPOINTS */

    // convert current image to grayscale
    cv::Mat imgGray;
    cv::cvtColor((dataBuffer.end() - 1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

    // extract 2D keypoints from current image
    vector<cv::KeyPoint> keypoints; // create empty feature list for current image
    detectKeypoints(imgGray, keypoints);

    // push keypoints and descriptor for current frame to end of data buffer
    (dataBuffer.end() - 1)->keypoints = keypoints;

    cout << "#5 : DETECT KEYPOINTS done" << endl;


    /* EXTRACT KEYPOINT DESCRIPTORS */

    cv::Mat descriptors;
    descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);

    // push descriptors for current frame to end of data buffer
    (dataBuffer.end() - 1)->descriptors = descriptors;

    cout << "#6 : EXTRACT DESCRIPTORS done" << endl;


    if (dataBuffer.size() > 1) // wait until at least two images have been processed
    {

      /* MATCH KEYPOINT DESCRIPTORS */

      vector<cv::DMatch> matches;

      matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
        (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
        matches, descriptorClass, matcherType, selectorType);

      // store matches in current data frame
      (dataBuffer.end() - 1)->kptMatches = matches;

      cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;


      /* TRACK 3D OBJECT BOUNDING BOXES */

      //// STUDENT ASSIGNMENT
      //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
      std::unordered_map<int, int> bbBestMatches;
      std::unordered_map<int, int> relevant_bbs;

      matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end() - 2), *(dataBuffer.end() - 1)); // associate bounding boxes between current and previous frame using keypoint matches

      // remove irrelevant matches - those that don't contain tracked lidar points
      std::remove_copy_if(bbBestMatches.begin(), bbBestMatches.end(), inserter(relevant_bbs, relevant_bbs.end()), [&](std::pair<int, int> el) {

        bool no_lidar = std::find_if(dataBuffer.rbegin()->boundingBoxes.begin(), dataBuffer.rbegin()->boundingBoxes.end(), [&](BoundingBox& bb) {return el.second == bb.boxID; })->lidarPoints.empty();
        if (no_lidar) {
          return true;
        }

        bool prev_no_lidar = std::find_if(dataBuffer.begin()->boundingBoxes.begin(), dataBuffer.begin()->boundingBoxes.end(), [&](BoundingBox& bb) {return el.first == bb.boxID; })->lidarPoints.empty();
        return prev_no_lidar;

        });

      //// EOF STUDENT ASSIGNMENT

      // store matches in current data frame
      (dataBuffer.end() - 1)->bbMatches = relevant_bbs;

      cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;


      /* COMPUTE TTC ON OBJECT IN FRONT */
      // loop over all BB match pairs
      for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
      {
        // find bounding boxes associates with current match
        BoundingBox* prevBB, * currBB;
        for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
        {
          if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB
          {
            currBB = &(*it2);
            break;
          }
        }

        for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
        {
          if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
          {
            prevBB = &(*it2);
            break;
          }
        }

        // compute TTC for current match
          //// STUDENT ASSIGNMENT
          //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
        double ttcLidar;
        computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);
        
        if (bDataCollection) {
          lidar_logger->add_result("Lidar", ttcLidar);
        }
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
        //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
        double ttcCamera;
        clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);
        computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);

        if (bDataCollection) {
          camera_logger->add_result(detectorType, descriptorType, ttcCamera);
        }
        //// EOF STUDENT ASSIGNMENT

        if (bVis & (int)Visualize::TTC)
        {
          visualize(currBB, ttcLidar, ttcCamera);
        }

      } // eof loop over all BB matches            
    }
  } // eof loop over all images
}