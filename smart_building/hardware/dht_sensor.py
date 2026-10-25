"""
hardware/dht_sensor.py
Interface for DHT11 Temperature and Humidity Sensor.
Provides hardware readings on Raspberry Pi GPIO or static simulated readings on non-ARM/Windows platforms.
"""

import time
import logging

try:
    from config import DHT_PIN, IS_SIMULATION
except ImportError:
    DHT_PIN = 4
    IS_SIMULATION = True

try:
    import board
    import adafruit_dht
    HAS_ADAFRUIT_DHT = True
except (ImportError, NotImplementedError, RuntimeError):
    HAS_ADAFRUIT_DHT = False

logger = logging.getLogger(__name__)


class DHTSensor:
    """
    Interface for DHT11 Temperature and Humidity Sensor.
    Supports hardware readings via adafruit_dht with 3 retries and fallback to last valid or simulated reading.
    """

    def __init__(
        self,
        pin: int = DHT_PIN,
        is_simulation: bool = IS_SIMULATION,
        min_interval: float = 2.5,
    ) -> None:
        """
        Initialize DHTSensor.

        :param pin: BCM GPIO pin number for DHT11 data line.
        :param is_simulation: If True, uses static simulated readings.
        :param min_interval: Minimum interval in seconds between physical sensor reads.
        """
        self.pin = pin
        self.is_simulation = is_simulation or not HAS_ADAFRUIT_DHT
        self.min_interval = float(min_interval)
        self.last_poll_time: float = 0.0
        self.last_valid_reading: tuple[float, float] = (28.0, 65.0)
        self._dht_device = None

        if not self.is_simulation:
            try:
                pin_attr = getattr(board, f"D{self.pin}")
                self._dht_device = adafruit_dht.DHT11(pin_attr)
                logger.info(f"Initialized DHT11 sensor on GPIO pin {self.pin}")
            except Exception as exc:
                logger.warning(
                    f"Failed to initialize adafruit_dht on pin {self.pin}: {exc}. "
                    "Falling back to simulation mode."
                )
                self.is_simulation = True

        if self.is_simulation:
            print("[SIMULATION] DHTSensor initialized in simulation mode.")

    def read(self) -> tuple[float, float]:
        """
        Read temperature (°C) and humidity (%) from sensor.

        Returns cached reading immediately if called within min_interval seconds.
        In simulation mode, returns static (28.0, 65.0).
        In hardware mode, retries up to 3 times on hardware read errors without blocking sleeps,
        returning last_valid_reading immediately on failure.

        :return: Tuple of (temperature, humidity) floats.
        """
        now = time.time()
        if (now - self.last_poll_time) < self.min_interval:
            return self.last_valid_reading

        self.last_poll_time = now

        if self.is_simulation or self._dht_device is None:
            print("[SIMULATION] DHT11 read -> (28.0, 65.0)")
            return self.last_valid_reading

        for attempt in range(1, 4):
            try:
                temp = self._dht_device.temperature
                hum = self._dht_device.humidity
                if temp is not None and hum is not None:
                    self.last_valid_reading = (float(temp), float(hum))
                    return self.last_valid_reading
            except Exception as exc:
                logger.debug(f"DHT read attempt {attempt}/3 failed: {exc}")

        logger.debug("DHT11 sensor read failed. Returning last valid reading immediately.")
        return self.last_valid_reading

    def cleanup(self) -> None:
        """
        Release underlying hardware sensor resources.
        """
        if self._dht_device is not None:
            try:
                self._dht_device.exit()
            except Exception as exc:
                logger.debug(f"Error during DHT device exit: {exc}")
            self._dht_device = None
        if self.is_simulation:
            print("[SIMULATION] DHTSensor cleaned up.")
