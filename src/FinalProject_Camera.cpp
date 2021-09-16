
/* INCLUDES FOR THIS PROJECT */
using namespace std;
#include "track3d.hpp"
#include "logger.hpp"

int main(int argc, const char* argv[]) {

  int bVis = (int)Track3d::Visualize::None;

  // data collection
  vector<string> detectors{ DetectorTypes::AKAZE, DetectorTypes::BRISK, DetectorTypes::FAST, DetectorTypes::HARRIS, DetectorTypes::ORB, DetectorTypes::SHITOMASI, DetectorTypes::SIFT };
  vector<string> descriptors{ DescriptorTypes::BRIEF, DescriptorTypes::BRISK, DescriptorTypes::FREAK, DescriptorTypes::ORB, DescriptorTypes::SIFT };

#ifdef VISUALIZE_LIDAR_ONLY
  Track3d tracker(string(DetectorTypes::FAST), string(DescriptorTypes::ORB), (int)Track3d::Visualize::Lidar);
  tracker.run_tracking_loop();
#endif  

  vector<std::pair<string, string>> detector_descriptor;
  
  // AKAZE descriptor requires AKAZE points
  detector_descriptor.push_back(std::make_pair(DetectorTypes::AKAZE, DescriptorTypes::AKAZE));

  // for each of the project tasks
  for (string& detectorType : detectors) {
    vector<string> actual_descriptors;
    std::copy(descriptors.begin(), descriptors.end(), back_inserter(actual_descriptors));

    // AKAZE descriptor can only be used with AKAZE points
    if (detectorType == DetectorTypes::AKAZE) {
      actual_descriptors.push_back(DescriptorTypes::AKAZE);
    }

    for (string& descriptorType : actual_descriptors) {
      if (detectorType == DetectorTypes::SIFT && descriptorType == DescriptorTypes::ORB) {
        continue;
      }
      std::cout << std::endl << "Running: " << detectorType << "/" << descriptorType << std::endl;
      std::cout << std::endl << "==========================================================" << std::endl;
      Track3d tracker(detectorType, descriptorType, bVis);
      
      if (!Track3d::camera_logger) {
        Track3d::lidar_logger = std::make_shared<CsvLogger<float>>("Method", 18, Track3d::docPath + "lidar.csv");
        Track3d::camera_logger = std::make_shared<CsvLogger<float>>("Detector_Descriptor", 18, Track3d::docPath + "camera.csv");
      }

      tracker.run_tracking_loop();
    }
  }
  Track3d::lidar_logger->dump();
  Track3d::camera_logger->dump();
}