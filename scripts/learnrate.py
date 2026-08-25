from math import floor
from typing import Optional


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def lr_clamped_linear(
    error: float,
    *,
    lr_min: float,
    lr_max: float,
    slope: float,
    error_scale: Optional[float] = None,
) -> float:
    """
    Error-dependent learning rate:

        lr = lr_min + slope * abs(error)

    The result is always limited to [lr_min, lr_max].

    If error_scale is provided, the error is first normalized:
        normalized_error = abs(error) / error_scale
    """
    magnitude = abs(error)

    if error_scale is not None:
        if error_scale <= 0.0:
            raise ValueError("error_scale must be positive")
        magnitude /= error_scale

    return clamp(
        lr_min + slope * magnitude,
        lr_min,
        lr_max,
    )


def lr_exponential(
    step: int,
    *,
    lr_initial: float,
    gamma: float,
    lr_min: float = 0.0,
) -> float:
    """
    Smooth exponential decay:

        lr = lr_initial * gamma ** step

    gamma = 0.99 means a 1% reduction per step.
    """
    if step < 0:
        raise ValueError("step must be non-negative")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in the interval (0, 1]")
    if lr_initial < 0.0 or lr_min < 0.0:
        raise ValueError("learning rates must be non-negative")

    return max(lr_min, lr_initial * gamma ** step)


def lr_inverse_time(
    step: int,
    *,
    lr_initial: float,
    decay: float,
    lr_min: float = 0.0,
) -> float:
    """
    Inverse-time decay:

        lr = lr_initial / (1 + decay * step)
    """
    if step < 0:
        raise ValueError("step must be non-negative")
    if decay < 0.0:
        raise ValueError("decay must be non-negative")
    if lr_initial < 0.0 or lr_min < 0.0:
        raise ValueError("learning rates must be non-negative")

    return max(lr_min, lr_initial / (1.0 + decay * step))


def lr_step(
    step: int,
    *,
    lr_initial: float,
    step_size: int,
    gamma: float,
    lr_min: float = 0.0,
) -> float:
    """
    Staircase decay:

        lr = lr_initial * gamma ** floor(step / step_size)

    The learning rate is reduced every step_size iterations.
    """
    if step < 0:
        raise ValueError("step must be non-negative")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in the interval (0, 1]")
    if lr_initial < 0.0 or lr_min < 0.0:
        raise ValueError("learning rates must be non-negative")

    decay_count = step // step_size
    return max(lr_min, lr_initial * gamma ** decay_count)