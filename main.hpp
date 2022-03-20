#pragma once

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/format.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>


#include "autoaim/autoaim.hpp"
#include "devices/camera/mv_video_capture.hpp"
#include "devices/serial/uart_serial.hpp"
#include "utils/fps.hpp"
#include "angle_solve/basic_pnp.hpp"
#include "record/record.hpp"

auto idntifier = fmt::format(fg(fmt::color::green) | fmt::emphasis::bold, "sentry_2022");