"""Stdlib port of the Metaculus template's percentile → CDF conversion and standardization.

Source: Metaculus/metac-bot-template main_with_no_framework.py (NumericDistribution), commit fetched 2026-10-06.
Constraints enforced by the Metaculus API on a submitted CDF (cdf_size = 201 for numeric):
- increases by at least 5e-05 at every step
- max PMF step = 0.2 * 200 / (cdf_size - 1) (0.95x wiggle room when building)
- closed bounds put no mass outside the range; open bounds keep a minimum mass outside
"""
import math

DEFAULT_CDF_SIZE = 201
MAX_PMF = 0.2
MIN_STEP = 5e-05


def max_pmf(cdf_size: int, wiggle_room: bool = True) -> float:
    cap = MAX_PMF * ((DEFAULT_CDF_SIZE - 1) / (cdf_size - 1))
    return cap * 0.95 if wiggle_room else cap


class Scale:
    def __init__(self, lower: float, upper: float, zero_point: float | None, open_lower: bool, open_upper: bool):
        if zero_point is not None and lower <= zero_point:
            raise ValueError(f"Lower bound {lower} must be greater than zero point {zero_point}")
        self.lower, self.upper, self.zero_point = lower, upper, zero_point
        self.open_lower, self.open_upper = open_lower, open_upper

    def to_location(self, value: float) -> float:
        """Real-world value → cdf x-axis location (0 at lower bound, 1 at upper bound)."""
        if self.zero_point is None:
            return (value - self.lower) / (self.upper - self.lower)
        ratio = (self.upper - self.zero_point) / (self.lower - self.zero_point)
        if value == self.zero_point:
            value += 1e-10
        return (math.log((value - self.lower) * (ratio - 1) + (self.upper - self.lower))
                - math.log(self.upper - self.lower)) / math.log(ratio)

    def to_value(self, location: float) -> float:
        if self.zero_point is None:
            return self.lower + (self.upper - self.lower) * location
        ratio = (self.upper - self.zero_point) / (self.lower - self.zero_point)
        return self.lower + (self.upper - self.lower) * (ratio ** location - 1) / (ratio - 1)


def _with_bound_anchors(percentiles: dict[float, float], scale: Scale) -> list[tuple[float, float]]:
    """percentiles: {percent (0-100): value}. Returns sorted [(percent, value)] incl. anchors at the bounds."""
    anchored = dict(percentiles)
    top, bottom = max(anchored), min(anchored)
    span = abs(scale.upper - scale.lower)
    buffer = 1 if span > 100 else 0.01 * span
    for percent, value in list(anchored.items()):
        if not scale.open_lower and value <= scale.lower + buffer:
            anchored[percent] = scale.lower + buffer
        if not scale.open_upper and value >= scale.upper - buffer:
            anchored[percent] = scale.upper - buffer
    if scale.open_upper:
        if scale.upper > anchored[top]:
            anchored[100 - 0.5 * (100 - top)] = scale.upper
    else:
        anchored[100] = scale.upper
    if scale.open_lower:
        if scale.lower < anchored[bottom]:
            anchored[0.5 * bottom] = scale.lower
    else:
        anchored[0] = scale.lower
    return sorted(anchored.items())


def _dedupe_values(points: list[tuple[float, float]], scale: Scale) -> list[tuple[float, float]]:
    counts: dict[float, int] = {}
    for _, value in points:
        counts[value] = counts.get(value, 0) + 1
    result = []
    for percent, value in points:
        fraction = percent / 100
        if counts[value] == 1:
            result.append((percent, value))
        elif scale.lower < value < scale.upper:
            result.append((percent, value - (1 - fraction) * 1e-6))
        elif value >= scale.upper:
            result.append((percent, scale.upper + 1e-10 * fraction))
        else:
            result.append((percent, scale.lower - 1e-10 * (1 - fraction)))
    return result


def _height_at(location: float, mapping: list[tuple[float, float]]) -> float:
    previous = mapping[0]
    for current in mapping[1:]:
        if previous[0] - 1e-10 <= location <= current[0] + 1e-10:
            if current[0] == previous[0]:
                return current[1]
            return previous[1] + (current[1] - previous[1]) * (location - previous[0]) / (current[0] - previous[0])
        previous = current
    raise ValueError(f"CDF location {location} not covered by declared percentiles")


def _standardize(cdf: list[float], scale: Scale) -> list[float]:
    lower_to = 0.0 if scale.open_lower else cdf[0]
    upper_to = 1.0 if scale.open_upper else cdf[-1]
    inbound = upper_to - lower_to
    size = len(cdf)
    adjusted = []
    for index, height in enumerate(cdf):
        rescaled = (height - lower_to) / inbound
        location = index / (size - 1)
        if scale.open_lower and scale.open_upper:
            adjusted.append(0.988 * rescaled + 0.01 * location + 0.001)
        elif scale.open_lower:
            adjusted.append(0.989 * rescaled + 0.01 * location + 0.001)
        elif scale.open_upper:
            adjusted.append(0.989 * rescaled + 0.01 * location)
        else:
            adjusted.append(0.99 * rescaled + 0.01 * location)
    # PMF space: [mass below range, inbound steps..., mass above range]
    pmf = [adjusted[0]] + [adjusted[i + 1] - adjusted[i] for i in range(size - 1)] + [1 - adjusted[-1]]
    cap = max_pmf(size)

    def capped(factor: float) -> list[float]:
        return [pmf[0]] + [min(cap, factor * mass) for mass in pmf[1:-1]] + [pmf[-1]]

    low = high = factor = 1.0
    while sum(capped(high)) < 1.0:
        high *= 1.2
    for _ in range(100):
        factor = 0.5 * (low + high)
        total = sum(capped(factor))
        if total < 1.0:
            low = factor
        else:
            high = factor
        if total == 1.0 or high - low < 2e-5:
            break
    pmf = capped(factor)
    inner = sum(pmf[1:-1])
    target = adjusted[-1] - adjusted[0]
    pmf = [pmf[0]] + [mass * target / inner for mass in pmf[1:-1]] + [pmf[-1]]
    running, result = 0.0, []
    for mass in pmf[:-1]:
        running += mass
        result.append(round(running, 10))
    return result


def percentiles_to_cdf(percentiles: dict[float, float], scale: Scale, cdf_size: int = DEFAULT_CDF_SIZE) -> list[float]:
    """{10: v10, 20: v20, ...} → standardized CDF of `cdf_size` points, validated."""
    ordered = sorted(percentiles.items())
    if len(ordered) < 2:
        raise ValueError("Need at least 2 percentiles")
    for (p1, v1), (p2, v2) in zip(ordered, ordered[1:]):
        if p1 >= p2 or v1 > v2:
            raise ValueError("Percentiles and values must be increasing")
    if scale.zero_point is not None and any(value < scale.zero_point for _, value in ordered):
        raise ValueError("Percentile value below the zero point of a log-scaled question")
    span = scale.upper - scale.lower
    if not any(scale.lower - 0.25 * span <= value <= scale.upper + 0.25 * span for _, value in ordered):
        raise ValueError("No declared percentile within the question range (+/-25%)")
    if any(value < scale.lower - 2 * span or value > scale.upper + 2 * span for _, value in ordered):
        raise ValueError("Declared percentiles far exceed the question bounds")
    points = _dedupe_values(_with_bound_anchors(dict(ordered), scale), scale)
    mapping = [(scale.to_location(value), percent / 100) for percent, value in points]
    raw = [_height_at(i / (cdf_size - 1), mapping) for i in range(cdf_size)]
    cdf = _standardize(raw, scale)
    validate(cdf, scale)
    return cdf


def validate(cdf: list[float], scale: Scale) -> None:
    steps = [b - a for a, b in zip(cdf, cdf[1:])]
    if min(steps) < MIN_STEP - 1e-12:
        raise ValueError(f"CDF step below {MIN_STEP}: {min(steps)}")
    if max(steps) > max_pmf(len(cdf), wiggle_room=False) + 1e-9:
        raise ValueError(f"CDF too concentrated: step {max(steps)}")
    if not scale.open_lower and abs(cdf[0]) > 1e-9:
        raise ValueError("Closed lower bound must start at 0")
    if not scale.open_upper and abs(cdf[-1] - 1) > 1e-9:
        raise ValueError("Closed upper bound must end at 1")
    if scale.open_lower and cdf[0] < 0.001 - 1e-9:
        raise ValueError("Open lower bound needs >= 0.001 mass below")
    if scale.open_upper and cdf[-1] > 0.999 + 1e-9:
        raise ValueError("Open upper bound needs >= 0.001 mass above")


def pool(cdfs: list[list[float]]) -> list[float]:
    """Linear pool (mean). Unlike the elementwise median, a mean of valid CDFs keeps both step constraints."""
    return [sum(column) / len(column) for column in zip(*cdfs)]
