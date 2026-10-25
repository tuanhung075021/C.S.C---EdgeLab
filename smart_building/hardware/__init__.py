"""
Hardware package for EcoZone AI Smart Energy Management System.
Exports GPIOController, calculate_fan_duty, and DHTSensor.
"""

from .gpio_controller import GPIOController, calculate_fan_duty
from .dht_sensor import DHTSensor

__all__ = [
    "GPIOController",
    "calculate_fan_duty",
    "DHTSensor",
]
