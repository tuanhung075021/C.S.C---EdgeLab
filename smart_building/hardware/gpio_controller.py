"""
hardware/gpio_controller.py
BCM GPIO controller for 6 Zone LEDs and L298N Fan PWM Speed Control.
Supports hardware control on Raspberry Pi and console simulation on non-ARM/Windows platforms.
"""

from typing import List

try:
    from config import IS_SIMULATION, ZONE_LEDS, FAN_ENA_PIN, FAN_IN1_PIN, FAN_IN2_PIN, RELAY_ACTIVE_LOW
except ImportError:
    IS_SIMULATION = True
    ZONE_LEDS = {0: 17, 1: 27, 2: 22, 3: 23, 4: 24, 5: 25}
    FAN_ENA_PIN = 18
    FAN_IN1_PIN = 5
    FAN_IN2_PIN = 6
    RELAY_ACTIVE_LOW = False


def calculate_fan_duty(temp: float, total_people: int) -> int:
    """
    Calculate L298N DC fan PWM duty cycle percentage based on temperature and total people count.

    Formula:
      0 if total_people <= 0 else min(100, max(0, int(round(40 + (temp - 25) * 5 + total_people * 10))))

    :param temp: Ambient temperature in degrees Celsius.
    :param total_people: Number of detected people in all zones.
    :return: Integer PWM duty cycle percentage clamped to [0, 100].
    """
    if total_people <= 0:
        return 0
    raw_duty = 40.0 + (temp - 25.0) * 5.0 + total_people * 10.0
    return min(100, max(0, int(round(raw_duty))))


class GPIOController:
    """
    GPIO Controller for 6 zone LEDs and PWM fan driver.
    """

    def __init__(
        self,
        is_simulation: bool = IS_SIMULATION,
        relay_active_low: bool = RELAY_ACTIVE_LOW,
    ) -> None:
        """
        Initialize GPIO Controller.

        :param is_simulation: If True, operates in simulation mode with console logging.
        :param relay_active_low: If True, uses active-LOW logic (0V/LOW = ON, 3.3V/HIGH = OFF).
        """
        self.is_simulation = is_simulation
        self.relay_active_low = relay_active_low
        self.current_zones: List[int] = [0] * 6
        self.current_fan_duty: int = 0
        self.pwm = None
        self.GPIO = None

        if not self.is_simulation:
            try:
                import RPi.GPIO as GPIO
                self.GPIO = GPIO
                self.GPIO.setmode(self.GPIO.BCM)
                self.GPIO.setwarnings(False)

                # Setup LED pins with initial state based on polarity
                initial_led_state = self.GPIO.HIGH if self.relay_active_low else self.GPIO.LOW
                for pin in ZONE_LEDS.values():
                    self.GPIO.setup(pin, self.GPIO.OUT, initial=initial_led_state)

                # Setup Fan driver pins
                self.GPIO.setup(FAN_ENA_PIN, self.GPIO.OUT)
                self.GPIO.setup(FAN_IN1_PIN, self.GPIO.OUT)
                self.GPIO.setup(FAN_IN2_PIN, self.GPIO.OUT)

                # Set Fan direction forward
                self.GPIO.output(FAN_IN1_PIN, self.GPIO.HIGH)
                self.GPIO.output(FAN_IN2_PIN, self.GPIO.LOW)

                # Initialize 1000Hz PWM on ENA pin
                self.pwm = self.GPIO.PWM(FAN_ENA_PIN, 1000)
                self.pwm.start(0)
            except (ImportError, RuntimeError, Exception) as exc:
                self.is_simulation = True
                print(f"[SIMULATION] Hardware GPIO driver unavailable ({exc}). Operating in simulation mode.")

        if self.is_simulation:
            print("[SIMULATION] GPIOController initialized in simulation mode.")

    def update_zones(self, zones: List[int]) -> None:
        """
        Update LED outputs for 6 zones.
        Active-HIGH: occupied (1) -> GPIO.HIGH (3.3V), empty (0) -> GPIO.LOW (0V).
        Active-LOW: occupied (1) -> GPIO.LOW (0V), empty (0) -> GPIO.HIGH (3.3V).

        :param zones: List of 6 integers (0 or 1) representing zone occupancy.
        """
        self.current_zones = list(zones)
        if not self.is_simulation and self.GPIO is not None:
            for idx, state in enumerate(zones):
                if idx in ZONE_LEDS:
                    pin = ZONE_LEDS[idx]
                    if self.relay_active_low:
                        pin_state = self.GPIO.LOW if state else self.GPIO.HIGH
                    else:
                        pin_state = self.GPIO.HIGH if state else self.GPIO.LOW
                    self.GPIO.output(pin, pin_state)
        else:
            status_str = ", ".join([f"Zone {i}: {'ON' if s else 'OFF'}" for i, s in enumerate(zones)])
            print(f"[SIMULATION] Zone LEDs updated -> {status_str}")

    def set_fan(self, temp: float, total_people: int) -> int:
        """
        Calculate and set DC fan PWM duty cycle based on temperature and total people count.

        :param temp: Ambient temperature in °C.
        :param total_people: Total count of detected people.
        :return: Calculated integer PWM duty cycle percentage.
        """
        duty = calculate_fan_duty(temp, total_people)
        self.current_fan_duty = duty
        if not self.is_simulation and self.pwm is not None:
            self.pwm.ChangeDutyCycle(duty)
        else:
            print(f"[SIMULATION] Fan duty set to {duty}% (Temp: {temp}°C, People: {total_people})")
        return duty

    def cleanup(self) -> None:
        """
        Stop PWM output and clean up GPIO configuration.
        """
        if not self.is_simulation and self.GPIO is not None:
            if self.pwm is not None:
                try:
                    self.pwm.stop()
                except Exception:
                    pass
                self.pwm = None
            try:
                self.GPIO.cleanup()
            except Exception:
                pass
        else:
            print("[SIMULATION] GPIOController cleaned up.")
