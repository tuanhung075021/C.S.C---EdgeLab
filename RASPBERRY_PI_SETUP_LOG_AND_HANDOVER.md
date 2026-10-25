# RASPBERRY_PI_SETUP_LOG_AND_HANDOVER.md

**Project:** EcoZone AI - Smart Energy Management via Edge AI & Spatial Mapping  
**Target Board:** Raspberry Pi 4 Model B  
**Target Agent:** AntiGravity  
**Date:** August 2026  

---

## 1. THÔNG SỐ PHẦN CỨNG THỰC TẾ (VERIFIED HARDWARE SPECS)

Cấu hình thực tế đã được kiểm tra trực tiếp trên thiết bị (khác với giả định ban đầu là 1GB RAM):

* **Board Model:** Raspberry Pi 4 Model B Rev 1.5
* **Dung lượng RAM:** **4GB RAM** (Thực tế hệ thống nhận `3.7Gi`, khả dụng `3.5Gi` RAM trống)
* **Bộ nhớ ảo (Swap):** **2GB** (`2.0Gi` Swap active)
* **Dung lượng thẻ nhớ:** ~14GB khả dụng
* **Hệ điều hành:** Raspberry Pi OS 64-bit (Debian Trixie Kernel 6.18.x)
* **Thông số kết nối:**
  * **User:** `csc_admin`
  * **Hostname:** `CSC-EdgeLab`
  * **IP Address:** `192.168.137.31` (SSH enabled)

---

## 2. NHẬT KÝ CẤU HÌNH ĐÃ THỰC HIỆN (SYSTEM CONFIG LOG)

1. **Cập nhật hệ thống (System Upgrade):**
   * Đã chạy hoàn tất `sudo apt update && sudo apt upgrade -y`.
   * Toàn bộ gói hệ thống, Linux Kernel, driver GPU/Mesa và bộ thư viện nền tảng đã được nâng cấp lên bản mới nhất.

2. **Tối ưu hóa Boot Mode (Console Mode):**
   * Đã cấu hình qua `sudo raspi-config` chuyển boot mode sang **`B1 Console`** (`multi-user.target`).
   * Giao diện đồ họa (Desktop GUI) đã được tắt hoàn toàn để tiết kiệm tài nguyên.
   * Lượng RAM ngốn cho OS lúc khởi động cực thấp (~150MB - 200MB), dành trọn >3.3GB RAM cho tiến trình AI & Web Server.

3. **Bộ nhớ tạm (Swap Memory):**
   * Hệ thống đã có sẵn **2GB Swap**, đủ an toàn để ngăn ngừa các sự cố OOM (Out Of Memory) trong quá trình biên dịch hoặc chạy AI.

---

## 3. THÔNG ĐIỆP & YÊU CẦU DÀNH CHO AGENT ANTIGRAVITY

> **Gửi Agent AntiGravity:**
> 
> Thiết bị Raspberry Pi 4 hiện tại sở hữu **4GB RAM + 2GB Swap**, không phải bản 1GB RAM như giả định thiết kế ban đầu. Rào cản nghẽn bộ nhớ đã được giải tỏa hoàn toàn. Hãy tái cấu trúc lại kế hoạch phát triển và kiến trúc phần mềm theo các định hướng sau:

### 1. Không cần gò ép / thắt chặt công nghệ quá mức
* **Mô hình AI:** Có thể nâng cấp chất lượng đầu vào của luồng video (MJPEG Stream từ ESP32-S3) lên $640 \times 480$ px hoặc $1280 \times 720$ px mà không sợ tràn RAM.
* **OpenVINO Inference:** Có thể linh hoạt thử nghiệm cả phiên bản YOLOv8n (FP16/INT8) lẫn YOLOv8s nếu cần độ chính xác cao hơn cho 6 vị trí Zone.
* **Xử lý đa luồng (Multi-threading):** Thoải mái tách các tiến trình (Stream Decoding, Perspective Transformation, OpenVINO Inference, Flask SSE Engine) chạy song song trên các nhân CPU riêng biệt để đạt FPS tối đa.

### 2. Tải trọng hệ thống cho phép
* RAM trống thực tế $\sim 3.5\text{ GB}$ dư sức duy trì bộ đệm khung hình (Frame Buffer), lưu lịch sử trạng thái các Zone ngầm trên RAM và phục vụ dashboard realtime mượt mà.
* Không cần tốn thời gian tìm cách cắt giảm thư viện phụ thuộc quá mức. Tập trung vào tính chuẩn hóa, dễ bảo trì và độ tin cậy của mã nguồn.

### 3. Nhiệm vụ tiếp theo dành cho Agent AntiGravity
1. **Cập nhật lại `ARCHITECTURE.md`:** Đột phá các giới hạn xử lý, nâng chỉ tiêu FPS kỳ vọng và độ phân giải xử lý ảnh.
2. **Khởi tạo môi trường lập trình:**
   * Tạo thư mục dự án `~/smart_building` và môi trường ảo `venv`.
   * Xây dựng `requirements.txt` chuẩn cho Python 3.10+ (gồm `openvino`, `opencv-python-headless`, `flask`, `gpiod`).
3. **Triển khai Mã nguồn:** Viết cấu trúc modular cho `core/` (Vision Stream, Transform, Detector) và `hardware/` (GPIO Controller, DHT11 Driver) để sẵn sàng chạy thử nghiệm.