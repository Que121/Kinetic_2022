#ifndef _NANODET_OPENVINO_H_
#define _NANODET_OPENVINO_H_

#include <string>
#include <opencv2/core.hpp>
#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>

typedef struct HeadInfo
{
  std::string cls_layer;
  std::string dis_layer;
  int stride;
} HeadInfo;

struct CenterPrior
{
  int x;
  int y;
  int stride;
};

typedef struct BoxInfo
{
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  int label;
} BoxInfo;

class NanoDet
{
public:
  NanoDet(const char *param);

  ~NanoDet();

  InferenceEngine::ExecutableNetwork network_;
  InferenceEngine::InferRequest infer_request_;
  // static bool hasGPU;

  // modify these parameters to the same with your config if you want to use your own model
  int input_size[2] = {416, 416};             // input height and width
  int num_class = 14;                         // number of classes. 80 for COCO
  int reg_max = 7;                            // `reg_max` set in the training config. Default: 7.
  std::vector<int> strides = {8, 16, 32, 64}; // strides of the multi-level feature.

  std::vector<BoxInfo> detect(cv::Mat image, float score_threshold, float nms_threshold);

private:
  void preprocess(cv::Mat &image, InferenceEngine::Blob::Ptr &blob);
  void decode_infer(const float *&pred, std::vector<CenterPrior> &center_priors, float threshold, std::vector<std::vector<BoxInfo>> &results);
  BoxInfo disPred2Bbox(const float *&dfl_det, int label, float score, int x, int y, int stride);
  static void nms(std::vector<BoxInfo> &result, float nms_threshold);
  std::string input_name_ = "data";
  std::string output_name_ = "output";
};

const int color_list[80][3] =
    {
        //{255 ,255 ,255}, //bg
        {216, 82, 24},
        {236, 176, 31},
        {125, 46, 141},
        {118, 171, 47},
        {76, 189, 237},
        {238, 19, 46},
        {76, 76, 76},
        {153, 153, 153},
        {255, 0, 0},
        {255, 127, 0},
        {190, 190, 0},
        {0, 255, 0},
        {0, 0, 255},
        {170, 0, 255},
        {84, 84, 0},
        {84, 170, 0},
        {84, 255, 0},
        {170, 84, 0},
        {170, 170, 0},
        {170, 255, 0},
        {255, 84, 0},
        {255, 170, 0},
        {255, 255, 0},
        {0, 84, 127},
        {0, 170, 127},
        {0, 255, 127},
        {84, 0, 127},
        {84, 84, 127},
        {84, 170, 127},
        {84, 255, 127},
        {170, 0, 127},
        {170, 84, 127},
        {170, 170, 127},
        {170, 255, 127},
        {255, 0, 127},
        {255, 84, 127},
        {255, 170, 127},
        {255, 255, 127},
        {0, 84, 255},
        {0, 170, 255},
        {0, 255, 255},
        {84, 0, 255},
        {84, 84, 255},
        {84, 170, 255},
        {84, 255, 255},
        {170, 0, 255},
        {170, 84, 255},
        {170, 170, 255},
        {170, 255, 255},
        {255, 0, 255},
        {255, 84, 255},
        {255, 170, 255},
        {42, 0, 0},
        {84, 0, 0},
        {127, 0, 0},
        {170, 0, 0},
        {212, 0, 0},
        {255, 0, 0},
        {0, 42, 0},
        {0, 84, 0},
        {0, 127, 0},
        {0, 170, 0},
        {0, 212, 0},
        {0, 255, 0},
        {0, 0, 42},
        {0, 0, 84},
        {0, 0, 127},
        {0, 0, 170},
        {0, 0, 212},
        {0, 0, 255},
        {0, 0, 0},
        {36, 36, 36},
        {72, 72, 72},
        {109, 109, 109},
        {145, 145, 145},
        {182, 182, 182},
        {218, 218, 218},
        {0, 113, 188},
        {80, 182, 188},
        {127, 127, 0},
};

struct object_rect
{
  int x;
  int y;
  int width;
  int height;
};

// 画框
void draw_bboxes(const cv::Mat &bgr, const std::vector<BoxInfo> &bboxes, object_rect effect_roi);

#endif //_NANODE_TOPENVINO_H_
