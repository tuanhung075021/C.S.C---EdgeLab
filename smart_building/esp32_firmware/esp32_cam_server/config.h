#ifndef CONFIG_H
#define CONFIG_H

// ==========================================
// WiFi Credentials & Network Configuration
// ==========================================
#if __has_include("wifi_secrets.h")
  #include "wifi_secrets.h"
#endif

#ifndef WIFI_SSID
  #define WIFI_SSID         "YOUR_WIFI_SSID"
#endif

#ifndef WIFI_PASSWORD
  #define WIFI_PASSWORD     "YOUR_WIFI_PASSWORD"
#endif

#define STREAM_SERVER_PORT 81
#define WIFI_TIMEOUT_MS   20000

// ==========================================
// Camera Model & Pinout Definition
// Freenove ESP32-S3 WROOM CAM Board (Kit V2)
// ==========================================
#define CAMERA_MODEL_FREENOVE_ESP32S3_CAM

#if defined(CAMERA_MODEL_FREENOVE_ESP32S3_CAM)
  #define PWDN_GPIO_NUM     -1
  #define RESET_GPIO_NUM    -1
  #define XCLK_GPIO_NUM     15
  #define SIOD_GPIO_NUM      4
  #define SIOC_GPIO_NUM      5

  #define Y9_GPIO_NUM       16
  #define Y8_GPIO_NUM       17
  #define Y7_GPIO_NUM       18
  #define Y6_GPIO_NUM       12
  #define Y5_GPIO_NUM       10
  #define Y4_GPIO_NUM        8
  #define Y3_GPIO_NUM        9
  #define Y2_GPIO_NUM       11

  #define VSYNC_GPIO_NUM     6
  #define HREF_GPIO_NUM      7
  #define PCLK_GPIO_NUM     13
#else
  #error "Camera model not selected or unsupported!"
#endif

// ==========================================
// Camera Performance & Frame Config
// ==========================================
#define CAM_FRAMESIZE    FRAMESIZE_VGA  // 640x480 resolution
#define CAM_JPEG_QUALITY 12             // Quality: lower=better (0-63). 12=good balance of quality & speed
#define CAM_FB_COUNT     2              // Double buffering with PSRAM

#endif // CONFIG_H
