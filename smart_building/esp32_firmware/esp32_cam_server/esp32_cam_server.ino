#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_camera.h"
#include "esp_http_server.h"
#include "img_converters.h"
#include "driver/gpio.h"
#include "config.h"

// UDP Auto-Discovery
WiFiUDP udp;
const int UDP_PORT = 12345;
unsigned long lastBroadcastTime = 0;

// MJPEG Stream Boundary Definitions
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

httpd_handle_t stream_httpd = NULL;

struct CameraProfile {
  const char* name;
  pixformat_t pixel_format;
  framesize_t init_frame_size;
  framesize_t target_frame_size;
  int xclk_freq_hz;
  int jpeg_quality;
  int fb_count;
  camera_grab_mode_t grab_mode;
};

// ------------------------------------------------------------------
// Non-Intrusive Sync Pin Transition Counter (Diagnostic)
// Measures VSYNC, HREF, PCLK activity over 50ms when capture fails
// ------------------------------------------------------------------
static void logSyncPinDiagnostics() {
  int prev_vsync = gpio_get_level((gpio_num_t)VSYNC_GPIO_NUM);
  int prev_href  = gpio_get_level((gpio_num_t)HREF_GPIO_NUM);
  int prev_pclk  = gpio_get_level((gpio_num_t)PCLK_GPIO_NUM);
  uint32_t vsync_toggles = 0;
  uint32_t href_toggles  = 0;
  uint32_t pclk_toggles  = 0;

  unsigned long start_us = micros();
  while ((unsigned long)(micros() - start_us) < 50000UL) {
    int vsync = gpio_get_level((gpio_num_t)VSYNC_GPIO_NUM);
    int href  = gpio_get_level((gpio_num_t)HREF_GPIO_NUM);
    int pclk  = gpio_get_level((gpio_num_t)PCLK_GPIO_NUM);

    if (vsync != prev_vsync) {
      vsync_toggles++;
      prev_vsync = vsync;
    }
    if (href != prev_href) {
      href_toggles++;
      prev_href = href;
    }
    if (pclk != prev_pclk) {
      pclk_toggles++;
      prev_pclk = pclk;
    }
  }

  Serial.printf("[DIAG] 50ms pin toggles: VSYNC(%d)=%u HREF(%d)=%u PCLK(%d)=%u\n",
                VSYNC_GPIO_NUM, (unsigned)vsync_toggles,
                HREF_GPIO_NUM, (unsigned)href_toggles,
                PCLK_GPIO_NUM, (unsigned)pclk_toggles);
}

// ------------------------------------------------------------------
// MJPEG Stream Handler
// Pushes multipart JPEG frames continuously over HTTP GET /stream
// Supports both hardware JPEG and Soft-JPEG fallback (RGB565/YUV422)
// ------------------------------------------------------------------
static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len = 0;
  uint8_t * _jpg_buf = NULL;
  char part_buf[128];

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) {
    return res;
  }

  // Set HTTP CORS headers to allow cross-origin requests from web clients
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  Serial.println("[STREAM] MJPEG stream client connected.");

  while (true) {
    fb = NULL;
    _jpg_buf = NULL;
    _jpg_buf_len = 0;

    for (int retry = 0; retry < 5; retry++) {
      fb = esp_camera_fb_get();
      if (fb) {
        break;
      }
      vTaskDelay(pdMS_TO_TICKS(40));
    }

    if (!fb) {
      Serial.println("[STREAM_ERROR] Camera capture failed after 5 attempts!");
      res = ESP_FAIL;
      break;
    }

    if (fb->format != PIXFORMAT_JPEG) {
      bool jpeg_converted = frame2jpg(fb, 80, &_jpg_buf, &_jpg_buf_len);
      esp_camera_fb_return(fb);
      fb = NULL;
      if (!jpeg_converted) {
        Serial.println("[STREAM_ERROR] Soft-JPEG conversion failed!");
        if (_jpg_buf) {
          free(_jpg_buf);
          _jpg_buf = NULL;
        }
        res = ESP_FAIL;
        break;
      }
    } else {
      _jpg_buf_len = fb->len;
      _jpg_buf = fb->buf;
    }

    res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
    if (res == ESP_OK) {
      size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART, _jpg_buf_len);
      res = httpd_resp_send_chunk(req, part_buf, hlen);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }

    // CRITICAL MEMORY MANAGEMENT: Always release frame buffer or converted JPEG buffer
    if (fb) {
      esp_camera_fb_return(fb);
      fb = NULL;
      _jpg_buf = NULL;
    } else if (_jpg_buf) {
      free(_jpg_buf);
      _jpg_buf = NULL;
    }

    if (res != ESP_OK) {
      // Client disconnected or socket error occurred
      break;
    }

    // Yield execution time to FreeRTOS scheduler to avoid TWDT (Task Watchdog) reset
    vTaskDelay(pdMS_TO_TICKS(10));
  }

  Serial.println("[STREAM] MJPEG stream client disconnected.");
  return res;
}

// ------------------------------------------------------------------
// Camera Initialization (Self-Testing Multi-Profile Fallback)
// Pre-allocates PSRAM UXGA buffer with reduced XCLK and Soft-JPEG fallback
// ------------------------------------------------------------------
bool initCamera() {
  // Temporarily enable driver debug output to expose cam_hal diagnostics (FB-OVF, NO-EOI, VSYNC-OVF)
  Serial.setDebugOutput(true);

  bool hasPsram = psramFound();
  Serial.printf("[CAMERA] PSRAM found=%s | Total=%u bytes | Free=%u bytes\n",
                hasPsram ? "YES" : "NO", (unsigned)ESP.getPsramSize(), (unsigned)ESP.getFreePsram());

  const CameraProfile profiles[] = {
    {
      "PSRAM UXGA-buf -> VGA JPEG 10MHz",
      PIXFORMAT_JPEG, FRAMESIZE_UXGA, FRAMESIZE_VGA,
      10000000, 10, 2, CAMERA_GRAB_WHEN_EMPTY
    },
    {
      "PSRAM UXGA-buf -> QVGA JPEG 10MHz",
      PIXFORMAT_JPEG, FRAMESIZE_UXGA, FRAMESIZE_QVGA,
      10000000, 10, 2, CAMERA_GRAB_WHEN_EMPTY
    },
    {
      "PSRAM UXGA-buf -> QVGA JPEG 6MHz",
      PIXFORMAT_JPEG, FRAMESIZE_UXGA, FRAMESIZE_QVGA,
      6000000, 12, 1, CAMERA_GRAB_WHEN_EMPTY
    },
    {
      "PSRAM QVGA RGB565 10MHz + SoftJPEG",
      PIXFORMAT_RGB565, FRAMESIZE_QVGA, FRAMESIZE_QVGA,
      10000000, 12, 2, CAMERA_GRAB_WHEN_EMPTY
    },
    {
      "PSRAM 240x240 RGB565 8MHz + SoftJPEG",
      PIXFORMAT_RGB565, FRAMESIZE_240X240, FRAMESIZE_240X240,
      8000000, 12, 2, CAMERA_GRAB_WHEN_EMPTY
    }
  };

  const int numProfiles = sizeof(profiles) / sizeof(profiles[0]);

  for (int i = 0; i < numProfiles; i++) {
    const CameraProfile& p = profiles[i];
    Serial.printf("[CAMERA] Testing profile [%d/%d]: %s...\n", i + 1, numProfiles, p.name);

    camera_config_t config = {};
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = p.xclk_freq_hz;
    config.pixel_format = p.pixel_format;
    config.frame_size   = p.init_frame_size;
    config.jpeg_quality = p.jpeg_quality;
    config.fb_count     = p.fb_count;
    config.grab_mode    = p.grab_mode;
    config.fb_location  = hasPsram ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
      Serial.printf("[CAMERA_WARN] Profile \"%s\" init failed with error 0x%x\n", p.name, err);
      continue;
    }

    sensor_t * s = esp_camera_sensor_get();
    if (s != NULL) {
      Serial.printf("[CAMERA] Sensor PID: 0x%x\n", s->id.PID);
      if (s->id.PID == OV3660_PID) {
        s->set_vflip(s, 1);
        s->set_brightness(s, 1);
        s->set_saturation(s, -2);
      }
      if (p.pixel_format == PIXFORMAT_JPEG && p.target_frame_size != p.init_frame_size) {
        s->set_framesize(s, p.target_frame_size);
      }
    }

    // Wait for sensor PLL and VSYNC stabilization
    delay(300);

    // Self-test live frame capture
    camera_fb_t * fb = NULL;
    for (int retry = 0; retry < 4; retry++) {
      fb = esp_camera_fb_get();
      if (fb) {
        break;
      }
      delay(100);
    }

    if (fb) {
      size_t verified_bytes = fb->len;
      bool capture_ok = true;

      if (fb->format != PIXFORMAT_JPEG) {
        uint8_t * test_jpg_buf = NULL;
        size_t test_jpg_len = 0;
        capture_ok = frame2jpg(fb, 80, &test_jpg_buf, &test_jpg_len);
        verified_bytes = test_jpg_len;
        if (test_jpg_buf) {
          free(test_jpg_buf);
          test_jpg_buf = NULL;
        }
      }

      esp_camera_fb_return(fb);
      fb = NULL;

      if (capture_ok && verified_bytes > 0) {
        Serial.printf("[CAMERA] Active profile: \"%s\" | Frame verified: %u bytes\n",
                      p.name, (unsigned)verified_bytes);
        Serial.setDebugOutput(false);
        return true;
      }

      Serial.printf("[CAMERA_WARN] Profile \"%s\" Soft-JPEG verification failed.\n", p.name);
    } else {
      Serial.printf("[CAMERA_WARN] Profile \"%s\" live capture timed out.\n", p.name);
      logSyncPinDiagnostics();
    }

    esp_camera_deinit();
  }

  Serial.println("[CAMERA_ERROR] All camera initialization profiles failed!");
  Serial.setDebugOutput(false);
  return false;
}

// ------------------------------------------------------------------
// WiFi Connection Routine
// Connects immediately; runs diagnostic scan only if timeout occurs
// ------------------------------------------------------------------
void setupWiFi() {
  Serial.print("[WIFI] Connecting to network: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false); // Disable WiFi sleep to reduce streaming latency and packet drops
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long startAttemptTime = millis();

  while (WiFi.status() != WL_CONNECTED && (millis() - startAttemptTime < WIFI_TIMEOUT_MS)) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\n[WIFI_ERROR] WiFi connection failed or timed out!");
    Serial.println("[WIFI] Running diagnostic 2.4GHz scan...");
    int n = WiFi.scanNetworks();
    if (n == 0) {
      Serial.println("[WIFI] No 2.4GHz networks found! Check ESP32 antenna.");
    } else {
      Serial.printf("[WIFI] Found %d networks nearby:\n", n);
      bool foundTarget = false;
      for (int i = 0; i < n; ++i) {
        String ssid = WiFi.SSID(i);
        Serial.printf("  [%d] \"%s\" (Signal: %d dBm)\n", i + 1, ssid.c_str(), WiFi.RSSI(i));
        if (ssid == WIFI_SSID) {
          foundTarget = true;
        }
      }
      if (!foundTarget) {
        Serial.printf("[WIFI_WARNING] Target \"%s\" was NOT FOUND in 2.4GHz scan!\n", WIFI_SSID);
        Serial.println("[TIP] Turn ON 'Maximize Compatibility' (iPhone) or set AP Band to 2.4GHz (Android).");
      }
    }
    Serial.println("[SYSTEM] Rebooting ESP32 in 5 seconds...");
    delay(5000);
    ESP.restart();
  }

  Serial.println("\n[WIFI] Connected successfully!");
  Serial.print("[WIFI] IP Address: ");
  Serial.println(WiFi.localIP());
  Serial.printf("[WIFI] Stream Endpoint: http://%s:%d/stream\n", WiFi.localIP().toString().c_str(), STREAM_SERVER_PORT);
}

// ------------------------------------------------------------------
// Start HTTP Stream Server on Port 81
// ------------------------------------------------------------------
void startStreamServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = STREAM_SERVER_PORT;
  config.ctrl_port   = STREAM_SERVER_PORT + 32688; // 81 + 32688 = 32769 (prevents port 80 collisions)

  httpd_uri_t stream_uri = {
    .uri       = "/stream",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };

  Serial.printf("[SERVER] Starting HTTP Stream Server on port %d...\n", config.server_port);
  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    Serial.printf("[SERVER] Server started successfully. Stream available at http://%s:%d/stream\n",
                  WiFi.localIP().toString().c_str(), STREAM_SERVER_PORT);
  } else {
    Serial.println("[SERVER_ERROR] Error starting HTTP stream server!");
    Serial.println("[SYSTEM] Rebooting ESP32 in 5 seconds...");
    delay(5000);
    ESP.restart();
  }
}

// ------------------------------------------------------------------
// Main Setup Initialization
// ------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false); // Suppress low-level driver spam
  Serial.println("\n=== EcoZone AI ESP32-S3 Camera Server Starting ===");

  // 1. Initialize Camera FIRST while internal DMA memory is clean and unfragmented
  if (!initCamera()) {
    Serial.println("[CRITICAL] Camera initialization failed! Rebooting in 5 seconds...");
    delay(5000);
    ESP.restart();
  }

  // 2. Initialize WiFi
  setupWiFi();

  startStreamServer();
  udp.begin(UDP_PORT);

  Serial.println("=== Initialization Complete. System Running ===");
}

// ------------------------------------------------------------------
// Main Background Loop
// Monitors WiFi health and handles non-blocking reconnection if needed
// ------------------------------------------------------------------
void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WIFI_WARNING] WiFi lost connection! Attempting reconnection...");
    WiFi.reconnect();
    delay(5000);
  } else {
    // Broadcast IP every 2 seconds for Auto-Discovery
    if (millis() - lastBroadcastTime > 2000) {
      udp.beginPacket("255.255.255.255", UDP_PORT);
      udp.print("ECOZONE_CAM:");
      udp.print(WiFi.localIP());
      udp.endPacket();
      lastBroadcastTime = millis();
    }
  }
  delay(10); // Prevent watchdog timeout and reduce CPU load
}
