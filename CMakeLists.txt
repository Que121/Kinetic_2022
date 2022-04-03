# Create project
cmake_minimum_required(VERSION 3.15.0) # 版本要求
project(Kinetic_2022)

# Use standard c++20
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# Add mindvision-sdk directory and include path for target
add_library(MVSDK SHARED IMPORTED)
get_filename_component(MVSDK_PATH 3rdparty/mindvision/linux REALPATH)

# Check system architecture for mindvision-sdk library
execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message(STATUS "Checking platform architecture for MVSDK: ${ARCHITECTURE}")
if(${ARCHITECTURE} STREQUAL x86_64)
    set(MIMDVISION_SDK_LIBRARY ${MVSDK_PATH}/lib/x64/libMVSDK.so)
elseif(${ARCHITECTURE} STREQUAL x86)
    set(MIMDVISION_SDK_LIBRARY ${MVSDK_PATH}/lib/x86/libMVSDK.so)
elseif(${ARCHITECTURE} STREQUAL aarch64)
    set(MIMDVISION_SDK_LIBRARY ${MVSDK_PATH}/lib/arm64/libMVSDK.so)
elseif(${ARCHITECTURE} STREQUAL armv7)
    set(MIMDVISION_SDK_LIBRARY ${MVSDK_PATH}/lib/arm/libMVSDK.so)
else()
    message(FATAL_ERROR "Not supported architecture for MVSDK: ${ARCHITECTURE}")
endif()
set_property(TARGET MVSDK PROPERTY IMPORTED_LOCATION ${MIMDVISION_SDK_LIBRARY})
# Add mindvision lib to build target
list(APPEND EXTRA_LIBS MVSDK)
# Add mindvision include path
list(APPEND EXTRA_INCLUDES 3rdparty/mindvision/linux/include)

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

# Find and add fmt & Threads as required
find_package(fmt REQUIRED)
find_package(Threads REQUIRED)

# Add main.cpp to executable
add_executable(Kinetic_2022 main.cpp)

target_link_libraries(Kinetic_2022
  Threads::Threads
  fmt::fmt
  ${OpenCV_LIBS}
  ${InferenceEngine_LIBRARIES} 
  ${NGRAPH_LIBRARIES}
  )

# 相机与串口
include_directories(devices)
add_subdirectory(devices)
target_link_libraries(Kinetic_2022 devices)

# fps显示和相机重启
include_directories(utils)
add_subdirectory(utils)
target_link_libraries(Kinetic_2022 utils)

# 功能模块
include_directories(module)
add_subdirectory(module)
target_link_libraries(Kinetic_2022 module)

# Set built binary to ~/bin
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# Set configs folder absolute path
set(CONFIG_FILE_PATH ${PROJECT_SOURCE_DIR}/configs)
set(SOURCE_PATH ${PROJECT_SOURCE_DIR})
target_compile_definitions(Kinetic_2022 PRIVATE "CONFIG_FILE_PATH=\"${CONFIG_FILE_PATH}\"")