"""
tests/test_hardware.py
Unit test suite for Milestone 3 Hardware & Sensor Controller.
"""

import io
import os
import sys
import time
import py_compile
import unittest
from unittest.mock import MagicMock, PropertyMock, patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hardware import GPIOController, calculate_fan_duty, DHTSensor


class TestHardwarePackage(unittest.TestCase):

    def test_py_compile(self):
        """Verify py_compile completes with zero errors on all hardware source files."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        hardware_files = [
            os.path.join(base_dir, "hardware", "__init__.py"),
            os.path.join(base_dir, "hardware", "gpio_controller.py"),
            os.path.join(base_dir, "hardware", "dht_sensor.py"),
        ]
        for file_path in hardware_files:
            compiled = py_compile.compile(file_path, doraise=True)
            self.assertIsNotNone(compiled)

    def test_package_exports(self):
        """Verify package exports from hardware/__init__.py."""
        self.assertIsNotNone(GPIOController)
        self.assertIsNotNone(calculate_fan_duty)
        self.assertIsNotNone(DHTSensor)

    def test_calculate_fan_duty_vectors(self):
        """Verify mandatory test vectors for calculate_fan_duty."""
        # set_fan(temp=25, total_people=1) -> duty = 50%
        self.assertEqual(calculate_fan_duty(25.0, 1), 50)
        # set_fan(temp=30, total_people=2) -> duty = 85%
        self.assertEqual(calculate_fan_duty(30.0, 2), 85)
        # set_fan(temp=20, total_people=0) -> duty = 0%
        self.assertEqual(calculate_fan_duty(20.0, 0), 0)
        # set_fan(temp=40, total_people=5) -> duty = 100% (clamped)
        self.assertEqual(calculate_fan_duty(40.0, 5), 100)

    def test_calculate_fan_duty_edge_cases(self):
        """Verify edge cases for calculate_fan_duty."""
        # Cold temperature with occupants (raw duty < 0, clamped to 0)
        self.assertEqual(calculate_fan_duty(10.0, 1), 0)
        # High temperature without occupants (occupancy 0 rule overrides high temp)
        self.assertEqual(calculate_fan_duty(45.0, 0), 0)
        # Fractional temperature rounding: 40 + (28.5-25)*5 + 2*10 = 40 + 17.5 + 20 = 77.5 -> 78
        self.assertEqual(calculate_fan_duty(28.5, 2), 78)
        # Negative people input
        self.assertEqual(calculate_fan_duty(25.0, -2), 0)
        # Extremely high people count clamped to 100
        self.assertEqual(calculate_fan_duty(25.0, 20), 100)

    def test_gpio_controller_simulation_logging(self):
        """Verify GPIOController in simulation mode console logging and state management."""
        with patch('sys.stdout', new_callable=io.StringIO) as fake_out:
            controller = GPIOController(is_simulation=True)
            self.assertTrue(controller.is_simulation)
            self.assertIn("[SIMULATION]", fake_out.getvalue())

        with patch('sys.stdout', new_callable=io.StringIO) as fake_out:
            zones = [1, 0, 1, 0, 0, 1]
            controller.update_zones(zones)
            self.assertEqual(controller.current_zones, zones)
            output = fake_out.getvalue()
            self.assertIn("[SIMULATION]", output)
            self.assertIn("Zone LEDs updated", output)

        with patch('sys.stdout', new_callable=io.StringIO) as fake_out:
            duty = controller.set_fan(30.0, 2)
            self.assertEqual(duty, 85)
            self.assertEqual(controller.current_fan_duty, 85)
            output = fake_out.getvalue()
            self.assertIn("[SIMULATION]", output)
            self.assertIn("Fan duty set to 85%", output)

        with patch('sys.stdout', new_callable=io.StringIO) as fake_out:
            controller.cleanup()
            output = fake_out.getvalue()
            self.assertIn("[SIMULATION]", output)
            self.assertIn("cleaned up", output)

    def test_dht_sensor_simulation(self):
        """Verify DHTSensor in simulation mode returns static (28.0, 65.0)."""
        with patch('sys.stdout', new_callable=io.StringIO) as fake_out:
            sensor = DHTSensor(pin=4, is_simulation=True)
            self.assertTrue(sensor.is_simulation)
            reading = sensor.read()
            self.assertEqual(reading, (28.0, 65.0))
            output = fake_out.getvalue()
            self.assertIn("[SIMULATION]", output)

        with patch('sys.stdout', new_callable=io.StringIO) as fake_out:
            sensor.cleanup()
            output = fake_out.getvalue()
            self.assertIn("[SIMULATION]", output)

    def test_dht_sensor_hardware_retry_success(self):
        """Verify DHTSensor retries hardware read on transient failure and returns valid reading."""
        sensor = DHTSensor(pin=4, is_simulation=True)
        sensor.is_simulation = False
        mock_device = MagicMock()
        type(mock_device).temperature = PropertyMock(side_effect=[Exception("Timing error"), 31.5])
        type(mock_device).humidity = PropertyMock(side_effect=[60.0])

        sensor._dht_device = mock_device
        with patch('time.sleep', return_value=None):
            reading = sensor.read()
            self.assertEqual(reading, (31.5, 60.0))
            self.assertEqual(sensor.last_valid_reading, (31.5, 60.0))

    def test_dht_sensor_hardware_retry_fallback(self):
        """Verify DHTSensor falls back to last_valid_reading after 3 failed hardware read retries."""
        sensor = DHTSensor(pin=4, is_simulation=True)
        sensor.last_valid_reading = (29.0, 62.0)
        sensor.is_simulation = False

        mock_device = MagicMock()
        type(mock_device).temperature = PropertyMock(side_effect=RuntimeError("Read error"))
        type(mock_device).humidity = PropertyMock(side_effect=RuntimeError("Read error"))
        sensor._dht_device = mock_device

        with patch('time.sleep', return_value=None):
            reading = sensor.read()
            self.assertEqual(reading, (29.0, 62.0))

    def test_gpio_controller_relay_active_low(self):
        """Verify GPIOController supports relay_active_low=True (Active-LOW: 0=HIGH, 1=LOW)."""
        mock_gpio = MagicMock()
        mock_gpio.BCM = 11
        mock_gpio.OUT = 0
        mock_gpio.LOW = 0
        mock_gpio.HIGH = 1

        mock_rpi = MagicMock()
        mock_rpi.GPIO = mock_gpio

        with patch.dict('sys.modules', {'RPi': mock_rpi, 'RPi.GPIO': mock_gpio}):
            controller = GPIOController(is_simulation=False, relay_active_low=True)
            self.assertFalse(controller.is_simulation)
            self.assertTrue(controller.relay_active_low)

            # Pin setup should be initialized with HIGH (OFF for active-LOW relay)
            for call_args in mock_gpio.setup.call_args_list:
                args, kwargs = call_args
                if args and args[0] in [17, 27, 22, 23, 24, 25]:
                    self.assertEqual(kwargs.get('initial'), mock_gpio.HIGH)

            # update_zones with zone 0 occupied (1) and zones 1..5 empty (0)
            mock_gpio.output.reset_mock()
            controller.update_zones([1, 0, 0, 0, 0, 0])
            # Pin 17 (zone 0) should receive LOW (energized/ON)
            # Other pins should receive HIGH (de-energized/OFF)
            mock_gpio.output.assert_any_call(17, mock_gpio.LOW)
            mock_gpio.output.assert_any_call(27, mock_gpio.HIGH)

    def test_dht_sensor_caching_and_non_blocking_timing(self):
        """Verify DHTSensor caching within min_interval and non-blocking failure return (<0.05s)."""
        sensor = DHTSensor(pin=4, is_simulation=True, min_interval=2.5)
        sensor.last_valid_reading = (25.5, 55.0)
        sensor.is_simulation = False

        mock_device = MagicMock()
        temp_prop = PropertyMock(side_effect=RuntimeError("Bus jitter"))
        hum_prop = PropertyMock(side_effect=RuntimeError("Bus jitter"))
        type(mock_device).temperature = temp_prop
        type(mock_device).humidity = hum_prop
        sensor._dht_device = mock_device

        # First read: failure should complete rapidly without sleeping
        start = time.time()
        reading = sensor.read()
        elapsed = time.time() - start
        self.assertLess(elapsed, 0.05, f"Hardware retry stalled for {elapsed}s")
        self.assertEqual(reading, (25.5, 55.0))

        # Second read immediately: should be cached (0 calls to _dht_device)
        temp_prop.reset_mock()
        reading_cached = sensor.read()
        self.assertEqual(reading_cached, (25.5, 55.0))
        temp_prop.assert_not_called()

    def test_vision_stream_fallback_cascade(self):
        """Verify VisionStream cascades to synthetic when ESP32 and webcam fail."""
        from core.vision_stream import VisionStream

        # With FORCE_ESP32_ONLY=1, should not fall back
        with patch.dict(os.environ, {"FORCE_ESP32_ONLY": "1"}):
            with patch('cv2.VideoCapture') as mock_cv:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = False
                mock_cv.return_value = mock_cap
                vs = VisionStream(stream_url="http://invalid-ip:81/stream", is_simulation=False)
                ret = vs._init_source()
                self.assertFalse(ret)
                self.assertEqual(vs.active_source_type, "none")

        # Without FORCE_ESP32_ONLY, should cascade to synthetic when webcam is unavailable
        with patch.dict(os.environ, {"FORCE_ESP32_ONLY": "0"}):
            with patch('cv2.VideoCapture') as mock_cv:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = False
                mock_cv.return_value = mock_cap
                vs = VisionStream(stream_url="http://invalid-ip:81/stream", is_simulation=False)
                ret = vs._init_source()
                self.assertTrue(ret)
                self.assertEqual(vs.active_source_type, "synthetic")


if __name__ == "__main__":
    unittest.main()
