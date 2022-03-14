cmake_minimum_required(VERSION 3.15)

find_package(OpenCV 4 REQUIRED)
find_package(fmt REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})


aux_source_directory(camera camera_src)
aux_source_directory(serial serial_src)

add_library(devices OBJECT ${camera_src} ${serial_src})
target_link_libraries(devices ${OpenCV_LIBS} fmt::fmt)