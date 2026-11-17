"""Scalar time-integration configuration shared by IGA--MPM engines."""

import math


def newmark_endpoint_velocity_coefficients(integration, timestep):
    """Return endpoint-velocity coefficients for ``[alpha, beta, gamma]``."""

    if len(integration) < 3:
        raise ValueError("fully implicit friction requires Newmark [alpha, beta, gamma]")
    alpha = float(integration[0])
    beta = float(integration[1])
    gamma = float(integration[2])
    timestep = float(timestep)
    if (
        not all(math.isfinite(value) for value in (alpha, beta, gamma, timestep))
        or alpha <= 0.0
        or beta <= 0.0
        or timestep <= 0.0
    ):
        raise ValueError(
            "fully implicit friction requires positive finite Newmark " "[alpha, beta, gamma] and timestep"
        )
    displacement = 0.5 * gamma / (alpha * beta * timestep)
    previous_velocity = -(0.5 * gamma / (alpha * beta) - 1.0)
    previous_acceleration = -0.5 * timestep * (gamma / beta - 2.0)
    return displacement, previous_velocity, previous_acceleration


__all__ = ["newmark_endpoint_velocity_coefficients"]
