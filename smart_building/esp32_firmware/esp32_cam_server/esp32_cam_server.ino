#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_camera.h"
#include "esp_http_server.h"
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

// ------------------------------------------------------------------
// MJPEG Stream Handler
// Pushes multipart JPEG frames continuously over HTTP GET /stream
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
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("[STREAM_ERROR] Camera capture failed!");
      res = ESP_FAIL;
      break;
    }

    _jpg_buf_len = fb->len;
    _jpg_buf = fb->buf;

    res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
    if (res == ESP_OK) {
      size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART, _jpg_buf_len);
      res = httpd_resp_send_chunk(req, part_buf, hlen);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }

    // CRITICAL MEMORY MANAGEMENT: Always release frame buffer immediately to prevent PSRAM memory leaks
    esp_camera_fb_return(fb);
    fb = NULL;

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
// Camera Initialization
// Configures OV2640 sensor for VGA (640x480) with double-buffered PSRAM
// ------------------------------------------------------------------
bool initCamera() {
  camera_config_t config;
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
  config.xclk_freq_hz = 16500000; // 16.5MHz prevents FB-OVF under heavy Wi-Fi streaming
  config.pixel_format = PIXFORMAT_JPEG;

  // PSRAM check & performance allocation
  if (psramFound()) {
    Serial.println("[CAMERA] PSRAM detected: Enabling VGA mode with double buffering in PSRAM.");
    config.frame_size   = CAM_FRAMESIZE;    // 640x480 (VGA)
    config.jpeg_quality = CAM_JPEG_QUALITY; // 12
    config.fb_count     = CAM_FB_COUNT;     // 2
    config.grab_mode    = CAMERA_GRAB_LATEST;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
  } else {
    Serial.println("[CAMERA_WARNING] PSRAM not found! Falling back to QVGA to prevent FB-OVF in DRAM.");
    config.frame_size   = FRAMESIZE_QVGA;   // 320x240 fallback to fit safely in internal SRAM
    config.jpeg_quality = 14;
    config.fb_count     = 1;
    config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location  = CAMERA_FB_IN_DRAM;
  }

  // Camera hardware initialization
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[CAMERA_ERROR] Camera init failed with error code 0x%x\n", err);
    return false;
  }

  // Configure sensor options (upright image orientation)
  sensor_t * s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_vflip(s, 0);
    s->set_hmirror(s, 0);
  }

  Serial.println("[CAMERA] Camera initialization succeeded (OV2640 VGA 640x480).");
  return true;
}

// ------------------------------------------------------------------
// WiFi Connection Routine
// Connects to WiFi network with Serial logging and timeout error handling
// ------------------------------------------------------------------
void setupWiFi() {
  Serial.print("[WIFI] Target network: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false); // Disable WiFi sleep to reduce streaming latency and packet drops

  // Scan nearby 2.4GHz Wi-Fi networks to diagnose visibility
  Serial.println("[WIFI] Scanning nearby 2.4GHz Wi-Fi networks...");
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
    if (foundTarget) {
      Serial.printf("[WIFI] Target \"%s\" is VISIBLE! Attempting connection...\n", WIFI_SSID);
    } else {
      Serial.printf("[WIFI_WARNING] Target \"%s\" was NOT FOUND in 2.4GHz scan!\n", WIFI_SSID);
      Serial.println("[TIP] If using Phone Hotspot: Turn ON 'Maximize Compatibility' (iPhone) or set AP Band to 2.4GHz (Android).");
    }
  }

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long startAttemptTime = millis();

  while (WiFi.status() != WL_CONNECTED && (millis() - startAttemptTime < WIFI_TIMEOUT_MS)) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\n[WIFI_ERROR] WiFi connection failed or timed out!");
    Serial.println("[TIP] Make sure hotspot/Wi-Fi is 2.4GHz (not 5GHz) and password is correct.");
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
