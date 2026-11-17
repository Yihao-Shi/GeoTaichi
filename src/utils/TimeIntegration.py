"""Three-parameter integration convention shared with IGA and Direct MPM."""

import math

import taichi as ti


def newmark_parameters(values):
    try:
        if isinstance(values, (str, bytes)):
            raise TypeError("newmark requires a parameter sequence")
        parameters = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("newmark must contain finite positive [alpha, beta, gamma]") from error
    if len(parameters) != 3 or any(not math.isfinite(v) or v <= 0 for v in parameters):
        raise ValueError("newmark must contain finite positive [alpha, beta, gamma]")
    return parameters


@ti.func
def newmark_prediction(position, velocity, acceleration, dt, integration):
    return position + dt * velocity + dt * dt * integration[0] * (1.0 - 2.0 * integration[1]) * acceleration


@ti.func
def newmark_acceleration(position, prediction, dt, integration):
    return (position - prediction) / (2.0 * integration[0] * integration[1] * dt * dt)


@ti.func
def newmark_velocity(velocity, acceleration, next_acceleration, dt, integration):
    return velocity + dt * ((1.0 - integration[2]) * acceleration + integration[2] * next_acceleration)
