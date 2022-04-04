#include "nanodet_openvino.hpp"

inline float fast_exp(float x)
{
  union
  {
    uint32_t i;
    float f;
  } v{};
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
  return v.f;
}

inline float sigmoid(float x)
{
  return 1.0f / (1.0f + fast_exp(-x));
}

template <typename _Tp>
int activation_function_softmax(const _Tp *src, _Tp *dst, int length)
{
  const _Tp alpha = *std::max_element(src, src + length);
  _Tp denominator{0};

  for (int i = 0; i < length; ++i)
  {
    dst[i] = fast_exp(src[i] - alpha);
    denominator += dst[i];
  }

  for (int i = 0; i < length; ++i)
  {
    dst[i] /= denominator;
  }

  return 0;
}

static void generate_grid_center_priors(const int input_height, const int input_width, std::vector<int> &strides, std::vector<CenterPrior> &center_priors)
{
  for (int i = 0; i < (int)strides.size(); i++)
  {
    int stride = strides[i];
    int feat_w = ceil((float)input_width / stride);
    int feat_h = ceil((float)input_height / stride);
    for (int y = 0; y < feat_h; y++)
    {
      for (int x = 0; x < feat_w; x++)
      {
        CenterPrior ct;
        ct.x = x;
        ct.y = y;
        ct.stride = stride;
        center_priors.push_back(ct);
      }
    }
  }
}

NanoDet::NanoDet(const char *model_path)
{
  InferenceEngine::Core ie;
  InferenceEngine::CNNNetwork model = ie.ReadNetwork(model_path);
  // prepare input settings
  InferenceEngine::InputsDataMap inputs_map(model.getInputsInfo());
  input_name_ = inputs_map.begin()->first;
  InferenceEngine::InputInfo::Ptr input_info = inputs_map.begin()->second;
  // input_info->setPrecision(InferenceEngine::Precision::FP32);
  // input_info->setLayout(InferenceEngine::Layout::NCHW);

  // prepare output settings
  InferenceEngine::OutputsDataMap outputs_map(model.getOutputsInfo());
  for (auto &output_info : outputs_map)
  {
    // std::cout<< "Output:" << output_info.first <<std::endl;
    output_info.second->setPrecision(InferenceEngine::Precision::FP32);
  }

  // get network
  network_ = ie.LoadNetwork(model, "CPU");
  infer_request_ = network_.CreateInferRequest();
}

NanoDet::~NanoDet()
{
}

void NanoDet::preprocess(cv::Mat &image, InferenceEngine::Blob::Ptr &blob)
{
  int img_w = image.cols;
  int img_h = image.rows;
  int channels = 3;

  InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
  if (!mblob)
  {
    THROW_IE_EXCEPTION << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
                       << "but by fact we were not able to cast inputBlob to MemoryBlob";
  }
  // locked memory holder should be alive all time while access to its buffer happens
  auto mblobHolder = mblob->wmap();

  float *blob_data = mblobHolder.as<float *>();

  for (size_t c = 0; c < channels; c++)
  {
    for (size_t h = 0; h < img_h; h++)
    {
      for (size_t w = 0; w < img_w; w++)
      {
        blob_data[c * img_w * img_h + h * img_w + w] =
            (float)image.at<cv::Vec3b>(h, w)[c];
      }
    }
  }
}

std::vector<BoxInfo> NanoDet::detect(cv::Mat image, float score_threshold, float nms_threshold)
{
  // auto start = std::chrono::steady_clock::now();

  InferenceEngine::Blob::Ptr input_blob = infer_request_.GetBlob(input_name_);
  preprocess(image, input_blob);

  // do inference
  infer_request_.Infer();

  // get output
  std::vector<std::vector<BoxInfo>> results;
  results.resize(this->num_class);

  {
    const InferenceEngine::Blob::Ptr pred_blob = infer_request_.GetBlob(output_name_);

    auto m_pred = InferenceEngine::as<InferenceEngine::MemoryBlob>(pred_blob);
    auto m_pred_holder = m_pred->rmap();
    const float *pred = m_pred_holder.as<const float *>();

    // generate center priors in format of (x, y, stride)
    std::vector<CenterPrior> center_priors;
    generate_grid_center_priors(this->input_size[0], this->input_size[1], this->strides, center_priors);

    this->decode_infer(pred, center_priors, score_threshold, results);
  }

  std::vector<BoxInfo> dets;
  for (int i = 0; i < (int)results.size(); i++)
  {
    this->nms(results[i], nms_threshold);

    for (auto &box : results[i])
    {
      dets.push_back(box);
    }
  }

  // auto end = std::chrono::steady_clock::now();
  // double time = std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "inference time:" << time << "ms" << std::endl;
  return dets;
}

void NanoDet::decode_infer(const float *&pred, std::vector<CenterPrior> &center_priors, float threshold, std::vector<std::vector<BoxInfo>> &results)
{
  const int num_points = center_priors.size();
  const int num_channels = num_class + (reg_max + 1) * 4;
  // printf("num_points:%d\n", num_points);

  // cv::Mat debug_heatmap = cv::Mat::zeros(feature_h, feature_w, CV_8UC3);
  for (int idx = 0; idx < num_points; idx++)
  {
    const int ct_x = center_priors[idx].x;
    const int ct_y = center_priors[idx].y;
    const int stride = center_priors[idx].stride;

    float score = 0;
    int cur_label = 0;

    for (int label = 0; label < num_class; label++)
    {
      if (pred[idx * num_channels + label] > score)
      {
        score = pred[idx * num_channels + label];
        cur_label = label;
      }
    }
    if (score > threshold)
    {
      // std::cout << row << "," << col <<" label:" << cur_label << " score:" << score << std::endl;
      const float *bbox_pred = pred + idx * num_channels + num_class;
      results[cur_label].push_back(this->disPred2Bbox(bbox_pred, cur_label, score, ct_x, ct_y, stride));
      // debug_heatmap.at<cv::Vec3b>(row, col)[0] = 255;
      // cv::imshow("debug", debug_heatmap);
    }
  }
}

BoxInfo NanoDet::disPred2Bbox(const float *&dfl_det, int label, float score, int x, int y, int stride)
{
  float ct_x = x * stride;
  float ct_y = y * stride;
  std::vector<float> dis_pred;
  dis_pred.resize(4);
  for (int i = 0; i < 4; i++)
  {
    float dis = 0;
    float *dis_after_sm = new float[reg_max + 1];
    activation_function_softmax(dfl_det + i * (reg_max + 1), dis_after_sm, reg_max + 1);
    for (int j = 0; j < reg_max + 1; j++)
    {
      dis += j * dis_after_sm[j];
    }
    dis *= stride;
    // std::cout << "dis:" << dis << std::endl;
    dis_pred[i] = dis;
    delete[] dis_after_sm;
  }
  float xmin = (std::max)(ct_x - dis_pred[0], .0f);
  float ymin = (std::max)(ct_y - dis_pred[1], .0f);
  float xmax = (std::min)(ct_x + dis_pred[2], (float)this->input_size[1]);
  float ymax = (std::min)(ct_y + dis_pred[3], (float)this->input_size[0]);

  // std::cout << xmin << "," << ymin << "," << xmax << "," << xmax << "," << std::endl;
  return BoxInfo{xmin, ymin, xmax, ymax, score, label};
}

void NanoDet::nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH)
{
  std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b)
            { return a.score > b.score; });
  std::vector<float> vArea(input_boxes.size());
  for (int i = 0; i < int(input_boxes.size()); ++i)
  {
    vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
  }
  for (int i = 0; i < int(input_boxes.size()); ++i)
  {
    for (int j = i + 1; j < int(input_boxes.size());)
    {
      float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
      float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
      float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
      float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
      float w = (std::max)(float(0), xx2 - xx1 + 1);
      float h = (std::max)(float(0), yy2 - yy1 + 1);
      float inter = w * h;
      float ovr = inter / (vArea[i] + vArea[j] - inter);
      if (ovr >= NMS_THRESH)
      {
        input_boxes.erase(input_boxes.begin() + j);
        vArea.erase(vArea.begin() + j);
      }
      else
      {
        j++;
      }
    }
  }
}

// 画框
void draw_bboxes(const cv::Mat &bgr, const std::vector<BoxInfo> &bboxes, object_rect effect_roi)
{
  // 类别
  static const char *class_names[] = {"person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                      "train", "truck", "boat", "traffic light", "fire hydrant",
                                      "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                                      "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                      "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                      "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                      "baseball glove", "skateboard", "surfboard", "tennis racket",
                                      "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                                      "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                      "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                      "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                                      "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                      "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                      "scissors", "teddy bear", "hair drier", "toothbrush"};

  cv::Mat image = bgr.clone();

  int src_w = image.cols;
  int src_h = image.rows;

  int dst_w = effect_roi.width;
  int dst_h = effect_roi.height;

  float width_ratio = (float)src_w / (float)dst_w;
  float height_ratio = (float)src_h / (float)dst_h;

  for (size_t i = 0; i < bboxes.size(); i++)
  {
    const BoxInfo &bbox = bboxes[i];

    // 定义颜色
    cv::Scalar color = cv::Scalar(color_list[bbox.label][0],
                                  color_list[bbox.label][1],
                                  color_list[bbox.label][2]);
    // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f %.2f\n", bbox.label, bbox.score,
    //     bbox.x1, bbox.y1, bbox.x2, bbox.y2);

    cv::rectangle(image,
                  cv::Rect(cv::Point((bbox.x1 - effect_roi.x) * width_ratio,
                                     (bbox.y1 - effect_roi.y) * height_ratio),
                           cv::Point((bbox.x2 - effect_roi.x) * width_ratio,
                                     (bbox.y2 - effect_roi.y) * height_ratio)),
                  color);

    char text[256];

    /**
     * @brief 输入字符串到text
     *
     * @param text 字符数组名
     * @param class_names 识别的类别名称
     * @param bbox.score 概率
     *
     */
    sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100);

    int baseLine = 0;

    /**
     * fontFace为文本的字体类型，
     * fontScale为文本大小的倍数（以字体库中的大小为基准而放大的倍数），
     * thickness为文本的粗细。
     * 最后一个参数baseLine是指距离文本最低点对应的y坐标
     */
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

    int x = (bbox.x1 - effect_roi.x) * width_ratio;
    int y = (bbox.y1 - effect_roi.y) * height_ratio - label_size.height - baseLine;

    if (y < 0)
      y = 0;

    if (x + label_size.width > image.cols)
      x = image.cols - label_size.width;

    /**
     * @brief 画出矩形
     *
     * @param image 输入图像
     * @param Rect 输入矩形（左上角坐标，宽高）
     * @param color 颜色
     */
    cv::rectangle(image,
                  cv::Rect(cv::Point(x, y),
                           cv::Size(label_size.width,
                                    label_size.height + baseLine)),
                  color, -1);

    cv::putText(image, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
  }

  cv::imshow("image", image);
}
