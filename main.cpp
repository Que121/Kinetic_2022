#include "main.hpp"

int main(int argc, char** argv)
{
  cv::Mat src_img_, roi_img_;
  mindvision::VideoCapture* mv_capture_ = new mindvision::VideoCapture(
    mindvision::CameraParam(0, mindvision::RESOLUTION_1280_X_800, mindvision::EXPOSURE_600));
     
  uart::SerialPort serial_ = uart::SerialPort(
    fmt::format("{}{}", CONFIG_FILE_PATH, "/serial/uart_serial_config.xml"));

  cv::VideoCapture cap_ = cv::VideoCapture(0);
  
  while (true) {
    global_fps_.getTick();
    if (mv_capture_->isindustryimgInput()) {
      src_img_ = mv_capture_->image();
    } else {
      cap_.read(src_img_);
    }
    if (!src_img_.empty()) {
      serial_.updateReceiveInformation();


    // 喂入模型
    auto detector = NanoDet("nanodet.xml");

    // 喂入成功
    std::cout<<"success"<<std::endl;

    // const char* path = ../..mp4;
    sentryAutoaim(detector);
    
}








