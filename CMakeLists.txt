cmake_minimum_required(VERSION 3.10)

project(vslam_simulator)
set(Pangolin_DIR "/home/huminghao/lib/pangolin_0.5_orb_install/lib/cmake/Pangolin")
find_package(Pangolin REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    ${Pangolin_INCLUDE_DIRS}
)

add_executable(vslam_simulator main.cpp src/viewer.cpp)

target_link_libraries(
    vslam_simulator
    ${Pangolin_LIBRARIES}
    ${OpenCV_LIBS}
)