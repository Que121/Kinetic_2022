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
 */
void openvinoNanodet_armorDetection(NanoDet &detector, cv::Mat image);

/**
 * @brief 归一化
 *
 * @param src 输入图像
 * @param dst 输出图像
 * @param dst_size  输出大小
 * @param effect_area roi区域
 * @return int
 */
int resize_uniform(cv::Mat &src, cv::Mat &dst, cv::Size dst_size, object_rect &effect_area);