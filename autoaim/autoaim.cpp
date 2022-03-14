#include "autoaim.hpp"

int resize_uniform(cv::Mat& src, cv::Mat& dst, cv::Size dst_size,
                   object_rect& effect_area) {
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
  if (ratio_src > ratio_dst) {
    tmp_w = dst_w;
    tmp_h = floor((dst_w * 1.0 / w) * h);
  } else if (ratio_src < ratio_dst) {
    tmp_h = dst_h;
    tmp_w = floor((dst_h * 1.0 / h) * w);
  } else {
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

  if (tmp_w != dst_w) {
    int index_w = floor((dst_w - tmp_w) / 2.0);
    // std::cout << "index_w: " << index_w << std::endl;
    for (int i = 0; i < dst_h; i++) {
      memcpy(dst.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3,
             tmp_w * 3);
    }
    effect_area.x = index_w;
    effect_area.y = 0;
    effect_area.width = tmp_w;
    effect_area.height = tmp_h;
  } else if (tmp_h != dst_h) {
    int index_h = floor((dst_h - tmp_h) / 2.0);
    // std::cout << "index_h: " << index_h << std::endl;
    memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);
    effect_area.x = 0;
    effect_area.y = index_h;
    effect_area.width = tmp_w;
    effect_area.height = tmp_h;
  } else {
    printf("error\n");
  }
  // cv::imshow("dst", dst);
  // cv::waitKey(0);
  return 0;
}

// 哨兵自瞄函数
int sentryAutoaim(NanoDet& detector, cv::Mat image) {

  object_rect effect_roi;
  cv::Mat resized_img;
  resize_uniform(image, resized_img, cv::Size(detector.input_size[0], detector.input_size[1]), effect_roi);

  auto results = detector.detect(resized_img, 0.4, 0.5);
  // 画框
  draw_bboxes(image, results, effect_roi);

  cv::waitKey(1);
  return 0;
}