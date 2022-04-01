#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <inference_engine.hpp>
#include <array>
#include <opencv2/opencv.hpp>
#include "detector/nanodet_openvino.hpp"

/**
 * @brief openvino加速nanodet识别装甲板主函数
 *
 * @param detector
 * @param image
 * @return int
 */
int openvinoNanodet_armorDetection(NanoDet &detector, cv::Mat image);
