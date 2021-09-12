
/* INCLUDES FOR THIS PROJECT */
using namespace std;
#include "track3d.hpp"

int main(int argc, const char* argv[]) {

  int bVis = (int)Track3d::Visualize::TTC;

  Track3d tracker(string(DetectorTypes::FAST), string(DescriptorTypes::ORB), bVis);

  tracker.run_tracking_loop();
}