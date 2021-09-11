
/* INCLUDES FOR THIS PROJECT */
using namespace std;
#include "track3d.hpp"

int main(int argc, const char* argv[]) {

  bool bVis = true;
  Track3d tracker(string(DetectorTypes::FAST), string(DescriptorTypes::ORB), bVis);

  tracker.run_tracking_loop();
}