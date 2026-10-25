# Hướng dẫn đấu nối phần cứng EcoZone AI (Hardware Wiring & Pinout Guide)

Tài liệu hướng dẫn chi tiết cách đấu nối hệ thống phần cứng cho dự án **EcoZone AI** trên bo mạch **Raspberry Pi 4 Model B (4GB RAM)**, bao gồm hai phương án: triển khai thực tế với **Module Relay 6 kênh cho cụm đèn 5V-10W** và mô hình thử nghiệm nhanh với **LED 5mm trên breadboard**, cùng mạch điều khiển quạt **L298N PWM** và cảm biến môi trường **DHT11**.

---

## 1. QUY TẮC AN TOÀN & NGUYÊN LÝ NGUỒN ĐIỆN (SAFETY & POWER RULES)

> [!CAUTION]
> **Quy tắc tiếp địa chung (Common Ground - GND):**  
> Bắt buộc phải nối chung tất cả các chân **GND** của Raspberry Pi, Module Relay 6 kênh, Driver động cơ L298N và các Bộ nguồn rời (External DC Power Supply). Nếu không có chung GND, mức điện thế tham chiếu giữa các thiết bị sẽ sai lệch, tín hiệu logic GPIO không nhận diện được và có nguy cơ xuất hiện điện áp ngược phá hỏng chân GPIO của Raspberry Pi.

> [!WARNING]
> **Tách biệt Nguồn Điều Khiển (Logic) và Nguồn Tải (Power):**  
> - Chân 5V trên header của Raspberry Pi chỉ cấp được dòng tối đa khoảng 1.2A - 1.5A cho toàn bộ ngoại vi.
> - Tuyệt đối **KHÔNG** dùng nguồn 5V của Raspberry Pi để cấp cho cuộn hút của đồng thời 6 relay và tải đèn công suất 5V-10W hoặc động cơ quạt. Dòng khởi động (inrush current) sẽ làm sụt áp gây sập nguồn (Brownout Reset) Raspberry Pi.
> - Luôn sử dụng bộ nguồn rời DC $5\text{V} - 10\text{A}$ cho dàn đèn LED và nguồn $5\text{V} - 12\text{V}$ cho động cơ quạt.
> - Mọi thao tác cắm dây chỉ thực hiện khi **ĐÃ NGẮT HOÀN TOÀN NGUỒN ĐIỆN**.

---

## 2. PHƯƠNG ÁN 1: ĐẤU NỐI RELAY 6 KÊNH CHO DÀN ĐÈN 5V-10W (SA BÀN THỰC TẾ)

Phương án này sử dụng module relay 6 kênh cách ly quang (optocoupler) để đóng ngắt 6 cụm đèn LED chiếu sáng độc lập cho 6 vùng (Zone 0 đến Zone 5).

### 2.1. Đấu nối Nguồn & Tín hiệu Logic từ Raspberry Pi sang Board Relay
- **VCC (Relay Board):** Nối vào nguồn ngoài $5\text{V}$ (hoặc chân $5\text{V}$ nếu dùng cấu hình jumper cách ly JD-VCC).
- **GND (Relay Board):** Nối chung vào chân GND của nguồn ngoài và chân GND Raspberry Pi (Chân số 6, 9, 14, 20...).
- **Các chân điều khiển IN1 .. IN6 nối vào GPIO Raspberry Pi (chuẩn BCM):**

| Kênh Relay | Khu vực (Zone) | Chân BCM Raspberry Pi | Vị trí chân vật lý (Header Pin) | Màu dây khuyến nghị |
| :--- | :--- | :--- | :--- | :--- |
| **IN1** | **Zone 0** (Hàng 0 - Trái) | `GPIO 17` | Chân số 11 | Vàng |
| **IN2** | **Zone 1** (Hàng 0 - Phải) | `GPIO 27` | Chân số 13 | Vàng |
| **IN3** | **Zone 2** (Hàng 1 - Trái) | `GPIO 22` | Chân số 15 | Cam |
| **IN4** | **Zone 3** (Hàng 1 - Phải) | `GPIO 23` | Chân số 16 | Cam |
| **IN5** | **Zone 4** (Hàng 2 - Trái) | `GPIO 24` | Chân số 18 | Xanh lá |
| **IN6** | **Zone 5** (Hàng 2 - Phải) | `GPIO 25` | Chân số 22 | Xanh lá |

### 2.2. Đấu nối Tải Đèn LED 5V-10W qua Tiếp Điểm Relay
Mỗi kênh trên board relay có 3 cọc vít: **NO** (Normally Open - Thường Mở), **COM** (Common - Chân Chung), và **NC** (Normally Closed - Thường Đóng).
* **Cực Dương (+5V) từ Nguồn Ngoài:** Nối vào cọc **COM** của cả 6 kênh relay.
* **Cọc NO của từng kênh:** Nối vào cực Dương (+) của đèn LED vùng tương ứng.
* **Cực Âm (-) của toàn bộ 6 đèn LED:** Nối thẳng về cực Âm (GND) của Bộ nguồn ngoài $5\text{V}$.

*(Khi relay được kích hoạt, tiếp điểm COM sẽ đóng sang NO, khép kín mạch cấp điện cho đèn sáng)*.

### 2.3. Cấu hình Cực tính Relay: Active-LOW vs Active-HIGH

* **Relay quang Active-LOW (Mặc định cho hầu hết module relay thương mại):**
  - Mạch relay được kích đóng (Relay ON, đèn sáng) khi chân GPIO xuất mức điện thế **LOW** ($0\text{V}$).
  - Mạch relay ngắt (Relay OFF, đèn tắt) khi chân GPIO xuất mức điện thế **HIGH** ($3.3\text{V}$).
  - Khi khởi động Raspberry Pi, nếu chưa được cấu hình đúng, trạng thái mặc định của chân GPIO có thể vô tình làm đóng toàn bộ 6 relay.
  - **Cách kích hoạt:** Đặt biến môi trường hệ thống hoặc trong service:
    ```bash
    export RELAY_ACTIVE_LOW=1
    ```
    Hệ thống `gpio_controller.py` sẽ tự động khởi tạo các chân ở mức `HIGH` (ngắt tiếp điểm) và chỉ kéo xuống `LOW` khi phát hiện người trong zone.

* **Relay Active-HIGH:**
  - Kích đóng khi GPIO ở mức `HIGH` ($3.3\text{V}$), ngắt khi ở mức `LOW` ($0\text{V}$).
  - Đặt biến môi trường: `RELAY_ACTIVE_LOW=0` (mặc định nếu không set).

---

## 3. PHƯƠNG ÁN 2: ĐẤU NỐI TRỰC TIẾP 6 LED 5MM BREADBOARD (TESTING NHANH)

Dành cho lập trình viên thử nghiệm thuật toán trên bàn làm việc mà không cần đấu nối dàn relay công suất.

* **Chân Âm (Cathode - chân ngắn):** Của cả 6 LED chụm lại cắm vào rail GND trên breadboard, nối về chân **GND** của Raspberry Pi (Header Pin 6).
* **Chân Dương (Anode - chân dài):** Mỗi LED nối tiếp qua 01 điện trở $220\,\Omega$ (hoặc $330\,\Omega$) để hạn dòng, sau đó cắm vào các chân GPIO tương ứng:
  - LED Zone 0 $\to$ `GPIO 17` (Pin 11)
  - LED Zone 1 $\to$ `GPIO 27` (Pin 13)
  - LED Zone 2 $\to$ `GPIO 22` (Pin 15)
  - LED Zone 3 $\to$ `GPIO 23` (Pin 16)
  - LED Zone 4 $\to$ `GPIO 24` (Pin 18)
  - LED Zone 5 $\to$ `GPIO 25` (Pin 22)
* **Cấu hình phần mềm:** Sử dụng chế độ mặc định `RELAY_ACTIVE_LOW=0` (khi có người: xuất `HIGH` 3.3V làm sáng LED).

---

## 4. ĐẤU NỐI ĐỘNG CƠ QUẠT TẢN NHIỆT (DC FAN) QUA DRIVER L298N

Module cầu H L298N nhận tín hiệu điều xung phần cứng (Hardware PWM) từ Raspberry Pi để tự động điều chỉnh tốc độ quạt theo nhiệt độ phòng và mật độ người.

### 4.1. Nguồn Cấp cho L298N
- **+12V (hoặc VCC trên L298N):** Nối vào cực Dương của nguồn ngoài ($5\text{V} - 12\text{V}$ tùy loại quạt).
- **GND (L298N):** Nối vào cực Âm nguồn ngoài **VÀ** nối chung vào chân **GND** của Raspberry Pi.
- **OUT1 & OUT2:** Nối trực tiếp 2 dây của động cơ quạt DC.

### 4.2. Tín hiệu Điều khiển từ Raspberry Pi
> [!IMPORTANT]
> Hãy **rút thanh jumper kim loại ngắn mạch** (jumper cap) trên chân **ENA** của board L298N ra trước khi cắm dây.

- **ENA (Enable A - Tín hiệu PWM):** Nối vào **`GPIO 18`** (Chân vật lý 12). Đây là chân PWM phần cứng chuyên dụng trên Raspberry Pi 4, chạy tần số $1000\text{ Hz}$.
- **IN1 (Direction 1):** Nối vào **`GPIO 5`** (Chân vật lý 29) - Điều khiển logic quay thuận (`HIGH`).
- **IN2 (Direction 2):** Nối vào **`GPIO 6`** (Chân vật lý 31) - Điều khiển logic quay (`LOW`).

---

## 5. ĐẤU NỐI CẢM BIẾN NHIỆT ĐỘ & ĐỘ ẨM DHT11

Module cảm biến DHT11 thu thập dữ liệu nhiệt độ và độ ẩm môi trường theo giao thức 1-Wire số.
- **VCC:** Nối vào chân **`3.3V`** (Chân vật lý 1). *(Lưu ý: Không cắm vào chân 5V để tránh mức logic 5V làm hỏng chân GPIO Pi).*
- **GND:** Nối vào chân **`GND`** (Chân vật lý 6 hoặc 9).
- **DATA (Signal):** Nối vào **`GPIO 4`** (Chân vật lý 7).
  *(Phần mềm đã tích hợp cơ chế chống nghẽn non-blocking: nếu đọc lỗi sẽ lập tức trả về giá trị bộ nhớ đệm trong $< 0.05\text{s}$, không làm treo luồng camera).*

---

## 6. BẢNG TỔNG HỢP SƠ ĐỒ CHÂN (MASTER PINOUT TABLE)

| Thiết bị / Thành phần | Tín hiệu | Chân BCM Pi | Vị trí Header Pin | Mức Logic / Chức năng | Nguồn cấp điện |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Zone 0 LED / Relay** | IN1 | `GPIO 17` | Pin 11 | Digital OUT (Active-LOW / HIGH) | External 5V 10A |
| **Zone 1 LED / Relay** | IN2 | `GPIO 27` | Pin 13 | Digital OUT (Active-LOW / HIGH) | External 5V 10A |
| **Zone 2 LED / Relay** | IN3 | `GPIO 22` | Pin 15 | Digital OUT (Active-LOW / HIGH) | External 5V 10A |
| **Zone 3 LED / Relay** | IN4 | `GPIO 23` | Pin 16 | Digital OUT (Active-LOW / HIGH) | External 5V 10A |
| **Zone 4 LED / Relay** | IN5 | `GPIO 24` | Pin 18 | Digital OUT (Active-LOW / HIGH) | External 5V 10A |
| **Zone 5 LED / Relay** | IN6 | `GPIO 25` | Pin 22 | Digital OUT (Active-LOW / HIGH) | External 5V 10A |
| **L298N Quạt PWM** | ENA | `GPIO 18` | Pin 12 | Hardware PWM ($1000\text{ Hz}$) | External 5V-12V |
| **L298N Hướng quay 1**| IN1 | `GPIO 5` | Pin 29 | Digital OUT (`HIGH`) | Logic 3.3V |
| **L298N Hướng quay 2**| IN2 | `GPIO 6` | Pin 31 | Digital OUT (`LOW`) | Logic 3.3V |
| **DHT11 Cảm biến** | DATA | `GPIO 4` | Pin 7 | Digital IN (1-Wire Pull-up) | Pi 3.3V Rail |
| **DHT11 Nguồn** | VCC | $3.3\text{V}$ | Pin 1 | DC Power Output | Pi 3.3V Rail |
| **Ground chung** | GND | `GND` | Pin 6, 9, 14 | Reference Mass | Common GND |
