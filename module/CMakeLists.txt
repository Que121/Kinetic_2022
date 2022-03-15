cmake_minimum_required(VERSION 3.15)

find_package(OpenCV 4 REQUIRED)
find_package(fmt REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})


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
  ${angle_src}
  ${armor_src}
  ${filter_src}
  ${record_src}
  ${roi_src}
  )

target_link_libraries(module ${OpenCV_LIBS} fmt::fmt) 