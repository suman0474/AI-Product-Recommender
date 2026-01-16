# utils/unit_converter.py
# Industrial Unit Conversion Library
# Handles pressure, temperature, flow rate, and range conversions for product matching

import re
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UnitValue:
    """Represents a value with its unit"""
    value: float
    unit: str
    original_string: str

    def __str__(self):
        return f"{self.value} {self.unit}"


@dataclass
class Range:
    """Represents a range with min/max values and unit"""
    min_value: float
    max_value: float
    unit: str
    original_string: str

    def __str__(self):
        return f"{self.min_value}-{self.max_value} {self.unit}"

    def contains(self, other: 'Range') -> bool:
        """Check if this range contains another range (after unit normalization)"""
        return self.min_value <= other.min_value and self.max_value >= other.max_value

    def overlaps(self, other: 'Range') -> bool:
        """Check if this range overlaps with another range"""
        return not (self.max_value < other.min_value or self.min_value > other.max_value)


class IndustrialUnitConverter:
    """
    Converts between common industrial units for process instrumentation.

    Supports:
    - Pressure: bar, psi, kPa, MPa, Pa
    - Temperature: °C, °F, K
    - Flow: m3/h, L/min, GPM, m3/s, L/s
    - Length: mm, cm, m, in, ft
    """

    # Pressure conversion factors (to bar as base unit)
    PRESSURE_UNITS = {
        "bar": 1.0,
        "psi": 0.0689476,
        "kpa": 0.01,
        "mpa": 10.0,
        "pa": 0.00001,
        "barg": 1.0,  # Gauge pressure, same conversion
        "psig": 0.0689476,  # Gauge pressure
    }

    # Flow rate conversion factors (to m3/h as base unit)
    FLOW_UNITS = {
        "m3/h": 1.0,
        "m³/h": 1.0,
        "l/min": 0.06,
        "l/h": 0.001,
        "gpm": 0.227125,
        "m3/s": 3600.0,
        "m³/s": 3600.0,
        "l/s": 3.6,
        "cfm": 1.699,  # Cubic feet per minute
        "ft3/h": 0.0283168,
        "ft³/h": 0.0283168,
    }

    # Length conversion factors (to mm as base unit)
    LENGTH_UNITS = {
        "mm": 1.0,
        "cm": 10.0,
        "m": 1000.0,
        "in": 25.4,
        "inch": 25.4,
        "ft": 304.8,
        "feet": 304.8,
    }

    # Common unit aliases
    UNIT_ALIASES = {
        "°c": "c",
        "deg c": "c",
        "degree c": "c",
        "celsius": "c",
        "°f": "f",
        "deg f": "f",
        "degree f": "f",
        "fahrenheit": "f",
        "kelvin": "k",
        "liters/min": "l/min",
        "liters/hour": "l/h",
        "gallons/min": "gpm",
        "cubic meters/hour": "m3/h",
        "kilopa": "kpa",
        "kilopascal": "kpa",
        "megapa": "mpa",
        "megapascal": "mpa",
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def normalize_unit_string(self, unit_str: str) -> str:
        """Normalize unit string to standard form"""
        unit_lower = unit_str.lower().strip()

        # Remove extra spaces
        unit_lower = re.sub(r'\s+', '', unit_lower)

        # Apply aliases
        if unit_lower in self.UNIT_ALIASES:
            return self.UNIT_ALIASES[unit_lower]

        return unit_lower

    def parse_pressure(self, value_str: str) -> Optional[UnitValue]:
        """
        Parse pressure string and extract value + unit.

        Examples:
            "10 bar" → UnitValue(10.0, "bar")
            "145 psi" → UnitValue(145.0, "psi")
            "1.5 MPa" → UnitValue(1.5, "MPa")
        """
        # Pattern: number (with optional decimal) + optional space + unit
        pattern = r'([-+]?\d+\.?\d*)\s*(bar|psi|kpa|mpa|pa|barg|psig)'

        match = re.search(pattern, value_str.lower())
        if not match:
            return None

        value = float(match.group(1))
        unit = self.normalize_unit_string(match.group(2))

        return UnitValue(value, unit, value_str)

    def parse_temperature(self, value_str: str) -> Optional[UnitValue]:
        """
        Parse temperature string and extract value + unit.

        Examples:
            "100 °C" → UnitValue(100.0, "c")
            "212 °F" → UnitValue(212.0, "f")
            "373 K" → UnitValue(373.0, "k")
        """
        # Pattern for temperature with various formats
        pattern = r'([-+]?\d+\.?\d*)\s*(?:°|deg|degree)?\s*([cfk])\b'

        match = re.search(pattern, value_str.lower())
        if not match:
            return None

        value = float(match.group(1))
        unit = match.group(2).lower()

        return UnitValue(value, unit, value_str)

    def parse_flow(self, value_str: str) -> Optional[UnitValue]:
        """
        Parse flow rate string and extract value + unit.

        Examples:
            "100 m3/h" → UnitValue(100.0, "m3/h")
            "10 GPM" → UnitValue(10.0, "gpm")
        """
        # Pattern for flow rates
        pattern = r'([-+]?\d+\.?\d*)\s*([a-z³3]+/[a-z]+)'

        match = re.search(pattern, value_str.lower())
        if not match:
            # Try without slash (like "gpm", "cfm")
            pattern2 = r'([-+]?\d+\.?\d*)\s*(gpm|cfm|lpm)'
            match = re.search(pattern2, value_str.lower())
            if not match:
                return None

        value = float(match.group(1))
        unit = self.normalize_unit_string(match.group(2))

        return UnitValue(value, unit, value_str)

    def convert_pressure(self, from_value: float, from_unit: str, to_unit: str = "bar") -> Optional[float]:
        """
        Convert pressure from one unit to another.

        Args:
            from_value: Pressure value
            from_unit: Source unit (bar, psi, kPa, etc.)
            to_unit: Target unit (default: bar)

        Returns:
            Converted value, or None if units not supported
        """
        from_unit_norm = self.normalize_unit_string(from_unit)
        to_unit_norm = self.normalize_unit_string(to_unit)

        if from_unit_norm not in self.PRESSURE_UNITS or to_unit_norm not in self.PRESSURE_UNITS:
            return None

        # Convert to bar first, then to target unit
        in_bar = from_value * self.PRESSURE_UNITS[from_unit_norm]
        result = in_bar / self.PRESSURE_UNITS[to_unit_norm]

        return result

    def convert_temperature(self, from_value: float, from_unit: str, to_unit: str = "c") -> Optional[float]:
        """
        Convert temperature from one unit to another.

        Args:
            from_value: Temperature value
            from_unit: Source unit (C, F, K)
            to_unit: Target unit (default: C)

        Returns:
            Converted value, or None if units not supported
        """
        from_unit_norm = self.normalize_unit_string(from_unit)
        to_unit_norm = self.normalize_unit_string(to_unit)

        # Convert to Celsius first
        if from_unit_norm == "c":
            in_celsius = from_value
        elif from_unit_norm == "f":
            in_celsius = (from_value - 32) * 5/9
        elif from_unit_norm == "k":
            in_celsius = from_value - 273.15
        else:
            return None

        # Convert from Celsius to target unit
        if to_unit_norm == "c":
            return in_celsius
        elif to_unit_norm == "f":
            return (in_celsius * 9/5) + 32
        elif to_unit_norm == "k":
            return in_celsius + 273.15
        else:
            return None

    def convert_flow(self, from_value: float, from_unit: str, to_unit: str = "m3/h") -> Optional[float]:
        """
        Convert flow rate from one unit to another.

        Args:
            from_value: Flow rate value
            from_unit: Source unit (m3/h, GPM, L/min, etc.)
            to_unit: Target unit (default: m3/h)

        Returns:
            Converted value, or None if units not supported
        """
        from_unit_norm = self.normalize_unit_string(from_unit)
        to_unit_norm = self.normalize_unit_string(to_unit)

        if from_unit_norm not in self.FLOW_UNITS or to_unit_norm not in self.FLOW_UNITS:
            return None

        # Convert to m3/h first, then to target unit
        in_m3h = from_value * self.FLOW_UNITS[from_unit_norm]
        result = in_m3h / self.FLOW_UNITS[to_unit_norm]

        return result

    def parse_range(self, range_str: str, value_type: str = "auto") -> Optional[Range]:
        """
        Parse a range string into a Range object.

        Examples:
            "0-100 °C" → Range(0, 100, "c")
            "10 to 200 bar" → Range(10, 200, "bar")
            "-20...150 °C" → Range(-20, 150, "c")

        Args:
            range_str: Range string to parse
            value_type: Type of value ("pressure", "temperature", "flow", "auto")

        Returns:
            Range object or None if parsing fails
        """
        # Pattern for ranges: min-max or min to max or min...max
        pattern = r'([-+]?\d+\.?\d*)\s*(?:-|to|\.\.\.)\s*([-+]?\d+\.?\d*)\s*(.+)'

        match = re.search(pattern, range_str.lower())
        if not match:
            return None

        min_val = float(match.group(1))
        max_val = float(match.group(2))
        unit_str = match.group(3).strip()

        # Normalize unit
        unit = self.normalize_unit_string(unit_str)

        return Range(min_val, max_val, unit, range_str)

    def ranges_compatible(
        self,
        req_range_str: str,
        product_range_str: str,
        value_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Check if product range is compatible with required range.

        Compatible means product range contains or overlaps with required range.

        Args:
            req_range_str: Required range string (e.g., "0-100 °C")
            product_range_str: Product range string (e.g., "-20 to 150 °C")
            value_type: Type of value for unit conversion

        Returns:
            Dict with compatibility info:
            {
                "compatible": bool,
                "reason": str,
                "req_range": Range,
                "product_range": Range,
                "contains": bool  # True if product range fully contains req range
            }
        """
        # Parse both ranges
        req_range = self.parse_range(req_range_str, value_type)
        product_range = self.parse_range(product_range_str, value_type)

        if not req_range or not product_range:
            return {
                "compatible": False,
                "reason": "Could not parse range strings",
                "req_range": req_range_str,
                "product_range": product_range_str
            }

        # Try to normalize units if different
        if req_range.unit != product_range.unit:
            # Detect value type and convert
            if req_range.unit in self.PRESSURE_UNITS and product_range.unit in self.PRESSURE_UNITS:
                # Convert product range to req range units
                product_range.min_value = self.convert_pressure(
                    product_range.min_value, product_range.unit, req_range.unit
                )
                product_range.max_value = self.convert_pressure(
                    product_range.max_value, product_range.unit, req_range.unit
                )
                product_range.unit = req_range.unit
            elif req_range.unit in ["c", "f", "k"] and product_range.unit in ["c", "f", "k"]:
                # Convert product range to req range units
                product_range.min_value = self.convert_temperature(
                    product_range.min_value, product_range.unit, req_range.unit
                )
                product_range.max_value = self.convert_temperature(
                    product_range.max_value, product_range.unit, req_range.unit
                )
                product_range.unit = req_range.unit
            elif req_range.unit in self.FLOW_UNITS and product_range.unit in self.FLOW_UNITS:
                # Convert product range to req range units
                product_range.min_value = self.convert_flow(
                    product_range.min_value, product_range.unit, req_range.unit
                )
                product_range.max_value = self.convert_flow(
                    product_range.max_value, product_range.unit, req_range.unit
                )
                product_range.unit = req_range.unit

        # Check compatibility
        contains = product_range.contains(req_range)
        overlaps = product_range.overlaps(req_range)

        if contains:
            return {
                "compatible": True,
                "reason": f"Product range fully contains required range",
                "req_range": str(req_range),
                "product_range": str(product_range),
                "contains": True,
                "overlap_percentage": 100.0
            }
        elif overlaps:
            # Calculate overlap percentage
            overlap_min = max(req_range.min_value, product_range.min_value)
            overlap_max = min(req_range.max_value, product_range.max_value)
            overlap_size = overlap_max - overlap_min
            req_size = req_range.max_value - req_range.min_value

            overlap_pct = (overlap_size / req_size * 100) if req_size > 0 else 0

            return {
                "compatible": True,
                "reason": f"Ranges overlap by {overlap_pct:.0f}%",
                "req_range": str(req_range),
                "product_range": str(product_range),
                "contains": False,
                "overlap_percentage": overlap_pct
            }
        else:
            return {
                "compatible": False,
                "reason": "Ranges do not overlap",
                "req_range": str(req_range),
                "product_range": str(product_range),
                "contains": False,
                "overlap_percentage": 0.0
            }

    def values_equivalent(
        self,
        value1_str: str,
        value2_str: str,
        value_type: str = "auto",
        tolerance: float = 0.01
    ) -> Dict[str, Any]:
        """
        Check if two values are equivalent (accounting for unit conversion).

        Args:
            value1_str: First value string (e.g., "10 bar")
            value2_str: Second value string (e.g., "145 psi")
            value_type: Type of value ("pressure", "temperature", "flow", "auto")
            tolerance: Tolerance for comparison (relative error, default 1%)

        Returns:
            Dict with equivalence info:
            {
                "equivalent": bool,
                "value1": UnitValue,
                "value2": UnitValue,
                "difference_pct": float
            }
        """
        # Auto-detect value type if not specified
        if value_type == "auto":
            if any(u in value1_str.lower() for u in ["bar", "psi", "pa", "kpa", "mpa"]):
                value_type = "pressure"
            elif any(u in value1_str.lower() for u in ["°c", "°f", "deg c", "deg f", " c", " f", " k", "celsius", "fahrenheit"]):
                value_type = "temperature"
            elif any(u in value1_str.lower() for u in ["m3/h", "gpm", "l/min", "cfm"]):
                value_type = "flow"

        # Parse values based on type
        if value_type == "pressure":
            val1 = self.parse_pressure(value1_str)
            val2 = self.parse_pressure(value2_str)

            if not val1 or not val2:
                return {"equivalent": False, "reason": "Could not parse pressure values"}

            # Convert to same unit
            val2_converted = self.convert_pressure(val2.value, val2.unit, val1.unit)

        elif value_type == "temperature":
            val1 = self.parse_temperature(value1_str)
            val2 = self.parse_temperature(value2_str)

            if not val1 or not val2:
                return {"equivalent": False, "reason": "Could not parse temperature values"}

            # Convert to same unit
            val2_converted = self.convert_temperature(val2.value, val2.unit, val1.unit)

        elif value_type == "flow":
            val1 = self.parse_flow(value1_str)
            val2 = self.parse_flow(value2_str)

            if not val1 or not val2:
                return {"equivalent": False, "reason": "Could not parse flow values"}

            # Convert to same unit
            val2_converted = self.convert_flow(val2.value, val2.unit, val1.unit)

        else:
            return {"equivalent": False, "reason": f"Unknown value type: {value_type}"}

        # Calculate difference percentage
        diff_pct = abs(val1.value - val2_converted) / val1.value if val1.value != 0 else 0

        equivalent = diff_pct <= tolerance

        return {
            "equivalent": equivalent,
            "value1": str(val1),
            "value2": str(val2),
            "difference_pct": diff_pct * 100,
            "tolerance_pct": tolerance * 100,
            "reason": f"Values are {'equivalent' if equivalent else 'different'} (diff: {diff_pct*100:.1f}%)"
        }


# Singleton instance
_converter = None

def get_unit_converter() -> IndustrialUnitConverter:
    """Get singleton unit converter instance"""
    global _converter
    if _converter is None:
        _converter = IndustrialUnitConverter()
    return _converter
