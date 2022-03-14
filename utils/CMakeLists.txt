cmake_minimum_required(VERSION 3.15)

find_package(OpenCV 4 REQUIRED)
find_package(fmt REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(utils OBJECT fps.hpp reset_mv_camera.hpp)
target_link_libraries(utils ${OpenCV_LIBS} fmt::fmt)