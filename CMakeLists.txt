# Create project
cmake_minimum_required(VERSION 3.4.1) # 版本要求
set(CMAKE_CXX_STANDARD 14)
project(sentry_2022)

# Find and add opencv as required
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})

# Find and add openvino as required
find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)


add_executable(sentry_2022 main.cpp main.hpp)
target_link_libraries(
  sentry_2022 
  ${OpenCV_LIBS}
  ${InferenceEngine_LIBRARIES} 
  ${NGRAPH_LIBRARIES}
  )

# 自瞄
include_directories(autoaim)
add_subdirectory(autoaim)
target_link_libraries(sentry_2022 autoaim)

# 相机与串口
include_directories(devices)
add_subdirectory(devices)
target_link_libraries(sentry_2022 devices)

# fps显示和相机重启
include_directories(utils)
add_subdirectory(utils)
target_link_libraries(sentry_2022 utils)







set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
