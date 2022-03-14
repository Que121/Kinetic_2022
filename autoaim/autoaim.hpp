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

// 哨兵自瞄启动
int sentryAutoaim(NanoDet &detector);

struct object_rect {
    int x;
    int y;
    int width;
    int height;
};

