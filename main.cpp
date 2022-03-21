#include "main.hpp"

int main(int argc, char **argv)
{
  fmt::print("[{}] sentry_2022 config file path: {}\n", idntifier, CONFIG_FILE_PATH);

  cv::Mat src_img_, roi_img_;

  // 迈德威视初始化，创建相机类对象
  mindvision::VideoCapture *mv_capture_ = new mindvision::VideoCapture(
      mindvision::CameraParam(0, mindvision::RESOLUTION_1280_X_800, mindvision::EXPOSURE_600)); // 记得修改分辨率

  // 串口模块初始化
  uart::SerialPort serial_ = uart::SerialPort(
      fmt::format("{}{}", CONFIG_FILE_PATH, "/serial/uart_serial_config.xml"));

  // pnp解算模块初始化
  basic_pnp::PnP pnp_ = basic_pnp::PnP(
      fmt::format("{}{}",
                  CONFIG_FILE_PATH, "/camera/mv_camera_config_407.xml"),
      fmt::format("{}{}",
                  CONFIG_FILE_PATH, "/angle_solve/basic_pnp_config.xml"));

  // 录制视频模块初始化
  RecordMode::Record record_ = RecordMode::Record(
      fmt::format("{}{}", CONFIG_FILE_PATH, "/record/recordpath_save.yaml"),
      fmt::format("{}{}", CONFIG_FILE_PATH, "/record/record_packeg/record.avi"),
      cv::Size(1280, 800)); // 记得修改分辨率

  // 读取备用普通相机？
  cv::VideoCapture cap_ = cv::VideoCapture(0);

  // 视频保存
  cv::VideoWriter vw_src;                                                         // 实例化保存视频类
  cv::FileStorage re_config_get(record_.video_save_path_, cv::FileStorage::READ); // 打开录制是平
  re_config_get["_PATH"] >> record_.path_;
  std::string save_path_ = "/record/";
  vw_src.open(CONFIG_FILE_PATH + save_path_ + to_string(record_.path_) + ".avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
              66, cv::Size(1280, 800), true); // 记得修改分辨率

  // 显示fps
  fps::FPS global_fps_;

  // 喂入模型
  auto detector = NanoDet("nanodet.xml");

  // 喂入成功
  std::cout << "success" << std::endl;

  while (true)
  {
    // 记录第一次时间点
    global_fps_.getTick();

    // 判断迈德威视工业相机是否开启
    if (mv_capture_->isindustryimgInput())
    {
      src_img_ = mv_capture_->image(); // 如果工业相机启动，src_img为工业相机输入的视频流
    }
    else
    {
      cap_.read(src_img_);
    } //

    // 判断src_img是否为空
    if (!src_img_.empty())
    {
      // 更新串口接收信息
      serial_.updateReceiveInformation();

      // 根据电控的发送信号选择模式
      switch (serial_.returnReceiveMode())
      {
      // 基础自瞄模式
      case uart::SUP_SHOOT:
        if (basic_armor_.runBasicArmor(src_img_, serial_.returnReceive()))
        {
          pnp_.solvePnP(serial_.returnReceiveBulletVelocity(), basic_armor_.returnFinalArmorDistinguish(0), basic_armor_.returnFinalArmorRotatedRect(0));
        }

        break;

      // 能量机关击打模式
      case uart::ENERGY_AGENCY:
        break;

      // 击打哨兵模式
      case uart::SENTRY_STRIKE_MODE:
        break;

      // 反小陀螺模式（暂未完善）
      case uart::TOP_MODE:
        break;

      // 录制视频
      case uart::RECORD_MODE:
        break;

      // 无人机模式（空缺）
      case uart::PLANE_MODE:
        break;

      // 哨兵模式
      case uart::SENTINEL_AUTONOMOUS_MODE:

        sentryAutoaim(detector, src_img_);
        break;

      // 雷达模式
      case uart::RADAR_MODE:
        break;

      // 手瞄
      default:

        break;
      }
    };
  }
}
