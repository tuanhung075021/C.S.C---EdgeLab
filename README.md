# EcoZone AI — Smart Edge Energy Management System
### Hệ Thống Quản Lý Năng Lượng Thông Minh Trên Thiết Bị Biên (Edge AI)

[![Platform: Raspberry Pi 4](https://img.shields.io/badge/Platform-Raspberry%20Pi%204%20(4GB)-C51A4A?logo=raspberrypi&logoColor=white)](https://www.raspberrypi.com/)
[![OS: Debian Trixie 64-bit](https://img.shields.io/badge/OS-Debian%20Trixie%2064--bit-A80030?logo=debian&logoColor=white)](https://www.debian.org/)
[![AI Engine: OpenVINO YOLOv8](https://img.shields.io/badge/AI-OpenVINO%20YOLOv8-0068B5?logo=intel&logoColor=white)](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
[![Camera: ESP32-S3](https://img.shields.io/badge/Camera-ESP32--S3%20MJPEG-E7352C?logo=espressif&logoColor=white)](https://www.espressif.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## MỤC LỤC / TABLE OF CONTENTS
1. [Tổng quan dự án / Project Overview](#1-tổng-quan-dự-án--project-overview)
2. [Kiến trúc hệ thống & Sơ đồ chân / Architecture & Pinout](#2-kiến-trúc-hệ-thống--sơ-đồ-chân--architecture--pinout)
3. [Cài đặt nhanh trên máy tính / Quickstart on PC/Mac/Linux](#3-cài-đặt-nhanh-trên-máy-tính--quickstart-on-pcmaclinux)
4. [Nạp firmware ESP32-S3 Camera / ESP32-S3 Camera Firmware](#4-nạp-firmware-esp32-s3-camera--esp32-s3-camera-firmware)
5. [Cân chỉnh góc nhìn camera qua Web / Web Perspective Calibration](#5-cân-chỉnh-góc-nhìn-camera-qua-web--web-perspective-calibration)
6. [Triển khai lên Raspberry Pi 4 / Raspberry Pi 4 Deployment](#6-triển-khai-lên-raspberry-pi-4--raspberry-pi-4-deployment)
7. [Tài liệu API & Giao thức SSE / API & SSE Documentation](#7-tài-liệu-api--giao-thức-sse--api--sse-documentation)

---

## 1. TỔNG QUAN DỰ ÁN / PROJECT OVERVIEW

### Tiếng Việt
**EcoZone AI** là giải pháp quản lý năng lượng thông minh ứng dụng Trí tuệ Nhân tạo tại biên (Edge AI) và Kỹ thuật Thị giác Không gian (Spatial Computer Vision), vận hành trên máy tính nhúng **Raspberry Pi 4 Model B (4GB RAM)** kết hợp camera góc rộng **ESP32-S3**.

Hệ thống theo dõi một sa bàn không gian thực tế kích thước **$30\text{ cm} \times 40\text{ cm}$** được chia làm **6 phân vùng (3 hàng $\times$ 2 cột)**. Tùy thuộc vào sự hiện diện của người hoặc vật thể mô phỏng trong từng phân vùng và điều kiện nhiệt độ môi trường (DHT11), hệ thống sẽ:
- **Tự động đóng ngắt chiếu sáng phân vùng:** Kích hoạt cụm đèn LED chiếu sáng độc lập tương ứng với khu vực có người thông qua module relay 6 kênh (hỗ trợ cả chuẩn Active-LOW và Active-HIGH).
- **Điều hòa vi khí hậu bằng quạt điều tốc PWM:** Tự động điều chỉnh xung PWM (1000Hz) qua driver cầu H L298N để tăng/giảm tốc độ quạt theo công thức tối ưu hóa nhiệt độ và mật độ người trong phòng.
- **Giám sát thời gian thực:** Cung cấp Web Dashboard hiển thị luồng video MJPEG, khung nhận diện HUD, bản đồ nhiệt phân vùng và thông số môi trường qua giao thức Server-Sent Events (SSE) với độ trễ $< 20\text{ ms}$.

### English
**EcoZone AI** is an intelligent Edge Computing energy management platform utilizing Deep Learning computer vision and spatial 2D homography mapping, deployed on a **Raspberry Pi 4 Model B (4GB RAM)** paired with an overhead **ESP32-S3** camera.

The system monitors a physical model table measuring **$30\text{ cm} \times 40\text{ cm}$** organized into a **3-row $\times$ 2-column grid (6 spatial zones)**. Based on occupancy status and environmental telemetry (DHT11), EcoZone AI delivers:
- **Zonal Illumination Automation:** Dynamically powers LED lighting fixtures for occupied zones via a 6-channel relay board with configurable polarity (`RELAY_ACTIVE_LOW`).
- **Dynamic PWM Microclimate Cooling:** Modulates DC cooling fan speed via hardware PWM (1000Hz) on an L298N driver based on ambient temperature and headcount.
- **Real-Time Telemetry & Web Dashboard:** Streams live video, spatial HUD overlays, occupancy grids, and climate data over Server-Sent Events (SSE) with sub-20ms latency.

---

## 2. KIẾN TRÚC HỆ THỐNG & SƠ ĐỒ CHÂN / ARCHITECTURE & PINOUT

### 2.1. Phân Vùng Sa Bàn 3×2 / 3×2 Spatial Grid Model
Sa bàn được nắn phẳng qua phép biến đổi phối cảnh 2D Homography sang khung hình kỹ thuật số kích thước **$600\text{ px} \times 400\text{ px}$**. Mỗi ô có kích thước **$300\text{ px} \times 133.33\text{ px}$**:

```text
+-----------------------------------+-----------------------------------+
|              ZONE 0               |              ZONE 1               |
|      Hàng 0, Cột 0 (Top-Left)     |     Hàng 0, Cột 1 (Top-Right)     |
|      Canvas: [0..300, 0..133]     |    Canvas: [300..600, 0..133]     |
+-----------------------------------+-----------------------------------+
|              ZONE 2               |              ZONE 3               |
|      Hàng 1, Cột 0 (Mid-Left)     |     Hàng 1, Cột 1 (Mid-Right)     |
|    Canvas: [0..300, 133..266]     |   Canvas: [300..600, 133..266]    |
+-----------------------------------+-----------------------------------+
|              ZONE 4               |              ZONE 5               |
|    Hàng 2, Cột 0 (Bottom-Left)    |    Hàng 2, Cột 1 (Bottom-Right)   |
|    Canvas: [0..300, 266..400]     |   Canvas: [300..600, 266..400]    |
+-----------------------------------+-----------------------------------+
```

**Công thức xác định Zone từ tọa độ pixel $(x, y)$:**
$$\text{Col} = \max\left(0, \min\left(1, \left\lfloor \frac{x}{300} \right\rfloor\right)\right)$$
$$\text{Row} = \max\left(0, \min\left(2, \left\lfloor \frac{y}{133.33} \right\rfloor\right)\right)$$
$$\text{Zone ID} = \text{Row} \times 2 + \text{Col} \quad (\text{giới hạn trong } [0, 5])$$

### 2.2. Cơ Chế Nhận Diện Kép (Dual Detection Pipeline)
- **Color Mode (`ColorDetector` - Mặc định cho Sa bàn):** Lọc màu HSV (`[20, 100, 100]` $\to$ `[35, 255, 255]`) trên khung hình đã nắn phẳng ($600 \times 400$) để phát hiện cốc vàng/sticker đại diện cho vật thể trong mô hình thu nhỏ.
- **AI Mode (`YOLOv8OpenVINODetector` - Dành cho Người thật):** Chạy mô hình YOLOv8 nano tối ưu hóa OpenVINO FP16 trên khung ảnh gốc ($640 \times 480$) nhận diện class 0 (`person`, `conf >= 0.35`). Điểm đáy bounding box $(x_{\text{center}}, y_{\text{max}})$ được chiếu qua ma trận Homography $H$ sang tọa độ sa bàn.

### 2.3. Bảng Sơ Đồ Chân Phần Cứng (Master Hardware Pinout)

| Thiết bị / Module | Tín hiệu | Chân BCM Raspberry Pi | Vị trí chân vật lý | Cấu hình & Mức Logic | Nguồn cấp |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Relay Kênh 1** | IN1 (Zone 0) | `GPIO 17` | Header Pin 11 | Digital OUT (Active-LOW / HIGH) | Nguồn ngoài 5V 10A |
| **Relay Kênh 2** | IN2 (Zone 1) | `GPIO 27` | Header Pin 13 | Digital OUT (Active-LOW / HIGH) | Nguồn ngoài 5V 10A |
| **Relay Kênh 3** | IN3 (Zone 2) | `GPIO 22` | Header Pin 15 | Digital OUT (Active-LOW / HIGH) | Nguồn ngoài 5V 10A |
| **Relay Kênh 4** | IN4 (Zone 3) | `GPIO 23` | Header Pin 16 | Digital OUT (Active-LOW / HIGH) | Nguồn ngoài 5V 10A |
| **Relay Kênh 5** | IN5 (Zone 4) | `GPIO 24` | Header Pin 18 | Digital OUT (Active-LOW / HIGH) | Nguồn ngoài 5V 10A |
| **Relay Kênh 6** | IN6 (Zone 5) | `GPIO 25` | Header Pin 22 | Digital OUT (Active-LOW / HIGH) | Nguồn ngoài 5V 10A |
| **L298N Quạt PWM** | ENA | `GPIO 18` | Header Pin 12 | Hardware PWM ($1000\text{ Hz}$) | Nguồn ngoài 5V-12V |
| **L298N Hướng 1** | IN1 | `GPIO 5` | Header Pin 29 | Digital OUT (`HIGH` - Quay thuận) | Nguồn Logic Pi |
| **L298N Hướng 2** | IN2 | `GPIO 6` | Header Pin 31 | Digital OUT (`LOW`) | Nguồn Logic Pi |
| **DHT11 Cảm biến** | DATA | `GPIO 4` | Header Pin 7 | 1-Wire Digital (Pull-up nội) | Nguồn Pi 3.3V |
| **Tiếp địa chung** | GND | `GND` | Header Pin 6 / 9 | Mass chung toàn hệ thống | Common Ground |

---

## 3. CÀI ĐẶT NHANH TRÊN MÁY TÍNH / QUICKSTART ON PC/MAC/LINUX

Bạn hoàn toàn có thể chạy, lập trình và kiểm thử toàn bộ hệ thống ngay trên máy tính cá nhân mà không cần gắn phần cứng vật lý (hệ thống tích hợp sẵn chế độ giả lập camera mô phỏng và hardware mock).

### Bước 1: Khởi tạo Môi trường Ảo (Virtual Environment)
```bash
# Clone kho mã nguồn
git clone https://github.com/your-org/CSC_EdgeLab.git
cd CSC_EdgeLab

# Tạo môi trường ảo Python 3.10+
python -m venv .venv

# Kích hoạt môi trường ảo:
# Trên Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Trên Linux / macOS:
source .venv/bin/activate
```

### Bước 2: Cài đặt Thư viện & Chuẩn bị Mô hình AI
```bash
# Cài đặt gói phụ thuộc cho môi trường dev
pip install --upgrade pip
pip install -r smart_building/requirements.txt

# Tải và xuất mô hình YOLOv8 sang định dạng OpenVINO IR
python smart_building/setup_model.py
```

### Bước 3: Chạy Bộ Kiểm Thử Tự Động (Run Test Suite)
```bash
# Chạy toàn bộ 227+ test cases
python smart_building/run_tests.py

# Hoặc chạy kiểm thử theo từng phân tầng:
python smart_building/run_tests.py --tier 1
python smart_building/run_tests.py --tier 2
```

### Bước 4: Khởi chạy Ứng dụng Web
```bash
cd smart_building
python app.py
```
Mở trình duyệt truy cập: **`http://localhost:5000`** để xem giao diện điều khiển trung tâm.

---

## 4. NẠP FIRMWARE ESP32-S3 CAMERA / ESP32-S3 CAMERA FIRMWARE

Camera ESP32-S3 đóng vai trò phát luồng video MJPEG và tự động gửi gói tin phát hiện (UDP broadcast beacon) về Raspberry Pi.

### 4.1. Thiết lập Cấu hình WiFi An toàn
Kho mã nguồn đã tách biệt mật khẩu WiFi nhằm bảo mật thông tin cá nhân:
1. Điều hướng vào thư mục firmware:
   `smart_building/esp32_firmware/esp32_cam_server/`
2. Sao chép file mẫu `wifi_secrets.example.h` thành `wifi_secrets.h`:
   ```bash
   cp wifi_secrets.example.h wifi_secrets.h
   ```
3. Mở `wifi_secrets.h` và điền thông tin WiFi của bạn:
   ```c
   #pragma once
   #define WIFI_SSID     "Ten_WiFi_Cua_Ban"
   #define WIFI_PASSWORD "Mat_Khau_WiFi"
   ```
*(File `wifi_secrets.h` đã được liệt kê trong `.gitignore`, tuyệt đối không commit lên GitHub).*

### 4.2. Biên dịch & Nạp Code bằng Arduino IDE
1. Cài đặt board package **ESP32 by Espressif Systems** (phiên bản `>= 2.0.11`).
2. Mở file `esp32_cam_server.ino` trong Arduino IDE.
3. Cấu hình thông số nạp trong menu **Tools**:
   - **Board:** `ESP32S3 Dev Module` (hoặc `Freenove ESP32S3 CAM`).
   - **PSRAM:** `OPI PSRAM` (Bắt buộc để xử lý khung ảnh phân giải cao).
   - **Flash Size:** `8MB` hoặc `16MB`.
   - **Upload Speed:** `921600`.
   - **USB CDC On Boot:** `Enabled`.
4. Nhấn **Upload** để nạp firmware.
5. Mở **Serial Monitor** ở baudrate `115200`. Khi khởi động thành công, ESP32 sẽ in địa chỉ IP (ví dụ `192.168.137.45`), khởi tạo luồng MJPEG tại cổng `81` (`/stream`) và tự động phát UDP beacon `ECOZONE_CAM:<IP>` trên cổng `12345`.

---

## 5. CÂN CHỈNH GÓC NHÌN CAMERA QUA WEB / WEB PERSPECTIVE CALIBRATION

Không cần sử dụng màn hình ngoài hay phần mềm đồ họa trên Raspberry Pi, hệ thống cung cấp giao diện cân chỉnh 4 góc bàn trực tiếp trên trình duyệt Web:

1. Đặt sa bàn vào vị trí cố định dưới camera ESP32-S3.
2. Truy cập vào đường dẫn cân chỉnh:
   **`http://<IP_RASPBERRY_PI>:5000/calibrate`** (hoặc `http://localhost:5000/calibrate`).
3. Giao diện Web hiển thị khung ảnh thô trực tiếp từ camera.
4. **Nhấp chuột lần lượt vào 4 góc của sa bàn theo chiều kim đồng hồ:**
   - **Điểm 1:** Góc Trên - Trái (Top-Left)
   - **Điểm 2:** Góc Trên - Phải (Top-Right)
   - **Điểm 3:** Góc Dưới - Phải (Bottom-Right)
   - **Điểm 4:** Góc Dưới - Trái (Bottom-Left)
5. Nhấn nút **"Save Calibration"** (Lưu Cân Chỉnh).
6. Hệ thống sẽ tự động tính toán ma trận Homography $3 \times 3$, lưu vào file `perspective_matrix.npy` và cập nhật tức thì luồng xử lý mà không cần khởi động lại server.
7. Bạn có thể nhấn **"Reset"** nếu muốn quay về trạng thái mặc định.

---

## 6. TRIỂN KHAI LÊN RASPBERRY PI 4 / RASPBERRY PI 4 DEPLOYMENT

### Thông số Phần cứng & Môi trường Đích (Target Setup)
- **Thiết bị:** Raspberry Pi 4 Model B (4GB RAM + 2GB Swap)
- **Hệ điều hành:** Raspberry Pi OS 64-bit (Debian Trixie, Linux Kernel 6.18+, Console/Headless)
- **Tài khoản mặc định:** `csc_admin` (Hostname: `CSC-EdgeLab`)
- **Địa chỉ IP cố định:** `192.168.137.31`
- **Đường dẫn thư mục dự án:** `/home/csc_admin/CSC_EdgeLab/smart_building`

### 6.1. Sao chép và Chạy Script Cài Đặt Tự Động
Đăng nhập SSH vào Raspberry Pi:
```bash
ssh csc_admin@192.168.137.31
```

Điều hướng vào thư mục mã nguồn và thực thi script triển khai:
```bash
cd /home/csc_admin/CSC_EdgeLab/smart_building
bash scripts/setup_pi.sh
```
*Script `setup_pi.sh` sẽ tự động cập nhật APT, cài đặt `libgpiod2`, tạo môi trường ảo `.venv`, cài đặt các gói chuyên biệt cho Pi trong `requirements-pi.txt` (`rpi-lgpio`, `opencv-python-headless`, `adafruit-circuitpython-dht`), và kiểm tra file mô hình OpenVINO IR.*

### 6.2. Kích hoạt Dịch Vụ Nền Hệ Thống (Systemd Service)
Để ứng dụng tự động chạy khi bật nguồn Raspberry Pi và tự phục hồi khi gặp lỗi:

```bash
# Sao chép file service vào thư mục cấu hình hệ thống
sudo cp scripts/ecozone.service /etc/systemd/system/ecozone.service

# Tải lại cấu hình systemd
sudo systemctl daemon-reload

# Kích hoạt service tự khởi động cùng hệ điều hành
sudo systemctl enable ecozone.service

# Khởi động dịch vụ ngay lập tức
sudo systemctl restart ecozone.service

# Kiểm tra trạng thái hoạt động của dịch vụ
sudo systemctl status ecozone.service
```

### 6.3. Quản lý và Xem Nhật Ký Vận Hành (Logs)
```bash
# Xem nhật ký thời gian thực
journalctl -u ecozone.service -f

# Khởi động lại dịch vụ sau khi cập nhật code
sudo systemctl restart ecozone.service

# Dừng dịch vụ
sudo systemctl stop ecozone.service
```

> [!TIP]
> Nếu bạn sử dụng mạch Relay công nghiệp cách ly quang Active-LOW (mặc định đóng khi GPIO ở mức 0V), hãy mở file `scripts/ecozone.service` và đảm bảo biến môi trường:
> ```ini
> Environment=RELAY_ACTIVE_LOW=1
> ```
> Sau đó chạy `sudo systemctl daemon-reload && sudo systemctl restart ecozone.service`.

---

## 7. TÀI LIỆU API & GIAO THỨC SSE / API & SSE DOCUMENTATION

Hệ thống cung cấp đầy đủ các cổng giao tiếp RESTful API và luồng truyền dữ liệu thời gian thực cho ứng dụng ngoài hoặc tích hợp hệ thống quản lý tòa nhà (BMS):

| Phương thức | Đường dẫn (Endpoint) | Định dạng dữ liệu | Mô tả chức năng |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | HTML | Giao diện Web Dashboard chính |
| `GET` | `/stream` | `text/event-stream` | Luồng dữ liệu thời gian thực Server-Sent Events (SSE) |
| `GET` | `/video_feed` | `multipart/x-mixed-replace` | Luồng video MJPEG có lớp phủ nhận diện HUD và lưới Zone |
| `GET` | `/raw_feed` | `multipart/x-mixed-replace` | Luồng video MJPEG thô từ camera phục vụ căn chỉnh góc nhìn |
| `GET` | `/debug_feed` | `multipart/x-mixed-replace` | Khung hình chẩn đoán (mặt nạ nhị phân HSV hoặc OpenVINO) |
| `POST` | `/set_mode` | JSON `{"mode": "color"\|"ai"}`| Chuyển đổi qua lại giữa chế độ Màu Sa Bàn và AI Người Thật |
| `GET` | `/calibrate` | HTML | Giao diện trực quan cân chỉnh 4 góc mặt phẳng phối cảnh |
| `GET` | `/calibrate/snapshot` | `image/jpeg` | Chụp khung ảnh thô hiện tại độ phân giải $640 \times 480$ |
| `POST` | `/calibrate/save` | JSON `{"points": [[x,y],...]}`| Lưu 4 tọa độ góc bàn, tính ma trận $H$ và cập nhật |
| `POST` | `/calibrate/reset` | JSON | Đặt lại ma trận biến đổi phối cảnh về mặc định |

### Cấu trúc Bản tin Telemetry SSE (`GET /stream`):
```json
{
  "zones": [1, 0, 0, 1, 0, 0],
  "temp": 28.5,
  "humidity": 62.0,
  "fan_speed": 70,
  "total_people": 2,
  "mode": "color",
  "timestamp": 1775822400
}
```

---

## BẢN QUYỀN & ĐỘI NGŨ PHÁT TRIỂN / LICENSE & TEAM
Dự án được phát hành theo giấy phép **MIT License**.  
Phát triển bởi đội ngũ **CSC EdgeLab** - Nghiên cứu và Ứng dụng Trí tuệ Nhân tạo Biên cho Tòa nhà Xanh Thông minh.
