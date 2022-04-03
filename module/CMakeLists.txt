cmake_minimum_required(VERSION 3.15)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres 2 REQUIRED)
find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)
find_package(fmt REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
# include_directories(${TensorRT_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

# openvino加速Nanodet
aux_source_directory(openvinoNanodet openvinoNanodet_src)
# pnp解算
aux_source_directory(angle_solve angle_src)
# 装甲板检测
aux_source_directory(armor armor_src)
# 滤波器
aux_source_directory(filter filter_src)
# 视频录制
aux_source_directory(record record_src)
# ROI
aux_source_directory(roi roi_src)

add_library(
  module SHARED
  ${openvinoNanodet_src}
  ${angle_src}
  ${armor_src}
  ${filter_src}
  ${record_src}
  ${roi_src}
  )

target_link_libraries(module  ${OpenCV_LIBS} ${CERES_LIBRARIES} ${InferenceEngine_LIBRARIES} ${NGRAPH_LIBRARIES} fmt::fmt) 