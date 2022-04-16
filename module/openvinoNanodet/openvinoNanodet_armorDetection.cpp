#include "openvinoNanodet_armorDetection.hpp"

int resize_uniform(cv::Mat &src, cv::Mat &dst, cv::Size dst_size,
                   object_rect &effect_area)
{
  int w = src.cols;
  int h = src.rows;
  int dst_w = dst_size.width;
  int dst_h = dst_size.height;

  // std::cout << "src: (" << h << ", " << w << ")" << std::endl;

  dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));

  float ratio_src = w * 1.0 / h;
  float ratio_dst = dst_w * 1.0 / dst_h;

  int tmp_w = 0;
  int tmp_h = 0;
  if (ratio_src > ratio_dst)
  {
    tmp_w = dst_w;
    tmp_h = floor((dst_w * 1.0 / w) * h);
  }
  else if (ratio_src < ratio_dst)
  {
    tmp_h = dst_h;
    tmp_w = floor((dst_h * 1.0 / h) * w);
  }
  else
  {
    cv::resize(src, dst, dst_size);
    effect_area.x = 0;
    effect_area.y = 0;
    effect_area.width = dst_w;
    effect_area.height = dst_h;
    return 0;
  }

  // std::cout << "tmp: (" << tmp_h << ", " << tmp_w << ")" << std::endl;
  cv::Mat tmp;
  cv::resize(src, tmp, cv::Size(tmp_w, tmp_h));

  if (tmp_w != dst_w)
  {
    int index_w = floor((dst_w - tmp_w) / 2.0);
    // std::cout << "index_w: " << index_w << std::endl;
    for (int i = 0; i < dst_h; i++)
    {
      memcpy(dst.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3,
             tmp_w * 3);
    }
    effect_area.x = index_w;
    effect_area.y = 0;
    effect_area.width = tmp_w;
    effect_area.height = tmp_h;
  }
  else if (tmp_h != dst_h)
  {
    int index_h = floor((dst_h - tmp_h) / 2.0);
    // std::cout << "index_h: " << index_h << std::endl;
    memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);
    effect_area.x = 0;
    effect_area.y = index_h;
    effect_area.width = tmp_w;
    effect_area.height = tmp_h;
  }
  else
  {
    printf("error\n");
  }
  // cv::imshow("dst", dst);
  // cv::waitKey(0);
  return 0;
}

void finalArmor_draw_bboxes(const cv::Mat &bgr,
                            const std::vector<BoxInfo> &bboxes,
                            std::vector<BoxInfo> &openvinoNanodetBboxes,
                            object_rect effect_roi,
                            const uart::Receive_Data _receive_data)
{
  // 储存过滤后的装甲板
  std::vector<BoxInfo> bboxesFiltered;

  // 装甲板过滤器
  for (size_t i = 0, j = 0; i < bboxes.size(); i++)
  {
    switch (_receive_data.my_color)
    {
    case uart::RED:
      if (bboxes[i].label > 6)
      {
        bboxesFiltered[j] = bboxes[i];
        j++;
      }
      break;
    case uart::BLUE:
      if (bboxes[i].label < 7)
      {
        bboxesFiltered[j] = bboxes[i];
        j++;
      }
      break;
    default:
      bboxesFiltered[j] = bboxes[i];
      j++;
      break;
    }
  }

  openvinoNanodetBboxes = bboxesFiltered;

  cv::Mat image = bgr.clone();

  int src_w = image.cols;
  int src_h = image.rows;

  int dst_w = effect_roi.width;
  int dst_h = effect_roi.height;

  float width_ratio = (float)src_w / (float)dst_w;
  float height_ratio = (float)src_h / (float)dst_h;

  // 遍历过滤器后推理结果
  for (size_t i = 0; i < bboxesFiltered.size(); i++)
  {

    const BoxInfo &bbox = bboxesFiltered[i];

    // 定义颜色
    cv::Scalar color = cv::Scalar(color_list[bbox.label][0],
                                  color_list[bbox.label][1],
                                  color_list[bbox.label][2]);

    // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f %.2f\n", bbox.label, bbox.score,
    //     bbox.x1, bbox.y1, bbox.x2, bbox.y2);

    // 画出过滤后装甲板
    cv::rectangle(image,
                  cv::Rect(cv::Point((bbox.x1 - effect_roi.x) * width_ratio,
                                     (bbox.y1 - effect_roi.y) * height_ratio),
                           cv::Point((bbox.x2 - effect_roi.x) * width_ratio,
                                     (bbox.y2 - effect_roi.y) * height_ratio)),
                  color);

    /*========================================= Prints labels and probabilities ===========================================*/
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
    cv::Size label_size = cv::getTextSize(text,
                                          cv::FONT_HERSHEY_SIMPLEX,
                                          0.4, 1, &baseLine);

    int x = (bbox.x1 - effect_roi.x) * width_ratio;
    int y = (bbox.y1 - effect_roi.y) * height_ratio - label_size.height - baseLine;

    if (y < 0)
      y = 0;
    if (x + label_size.width > image.cols)
      x = image.cols - label_size.width;

    cv::rectangle(image,
                  cv::Rect(cv::Point(x, y),
                           cv::Size(label_size.width,
                                    label_size.height + baseLine)),
                  color,
                  -1);

    cv::putText(image, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    /*====================================================================================================================*/
  }

  cv::imshow("image", image);
}
