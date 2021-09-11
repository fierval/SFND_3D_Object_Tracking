cmake_minimum_required(VERSION 3.2)
project(camera_fusion CXX)

set(CMAKE_CXX_STANDARD 17)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")


find_package(OpenCV 4.1 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

set(SOURCE_FILES 
    src/camFusion_Student.cpp 
    src/FinalProject_Camera.cpp 
    src/lidarData.cpp 
    src/matching2D_Student.cpp 
    src/objectDetection2D.cpp
    src/track3d.cpp)

# Executable for create matrix exercise
add_executable (3D_object_tracking ${SOURCE_FILES})
target_link_libraries (3D_object_tracking ${OpenCV_LIBRARIES})