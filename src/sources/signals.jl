@doc raw"""
    GaussianPulse(width; delay = 0)

Time signal ``s(t) = \exp(-((t - t_0) / w)^2)`` with `width` ``w`` and `delay`
``t_0``. Callable as `signal(t)`; [`signal_derivative`](@ref) and
[`signal_second_derivative`](@ref) give ``\dot s`` and ``\ddot s``.
"""
struct GaussianPulse{RealT <: Real}
    width::RealT
    delay::RealT
end

GaussianPulse(width; delay = zero(width)) = GaussianPulse(promote(width, delay)...)

@inline function (signal::GaussianPulse)(t)
    tau = (t - signal.delay) / signal.width
    return exp(-tau^2)
end

@inline function signal_derivative(signal::GaussianPulse, t)
    tau = (t - signal.delay) / signal.width
    return -2 * tau / signal.width * exp(-tau^2)
end

@inline function signal_second_derivative(signal::GaussianPulse, t)
    tau = (t - signal.delay) / signal.width
    return (4 * tau^2 - 2) / signal.width^2 * exp(-tau^2)
end

@doc raw"""
    ModulatedGaussianPulse(frequency, width; delay = 0, phase = 0)

Time signal ``s(t) = \exp(-((t - t_0) / w)^2) \sin(2 \pi f (t - t_0) + \phi)``.
Two signals with phases ``0`` and ``\pi / 2`` on orthogonal polarizations give a
circularly polarized [`PlaneWave`](@ref).
"""
struct ModulatedGaussianPulse{RealT <: Real}
    frequency::RealT
    width::RealT
    delay::RealT
    phase::RealT
end

function ModulatedGaussianPulse(frequency, width; delay = zero(width), phase = zero(width))
    return ModulatedGaussianPulse(promote(frequency, width, delay, phase)...)
end

@inline angular_frequency(signal::ModulatedGaussianPulse) = 2 *
                                                            oftype(signal.frequency, pi) *
                                                            signal.frequency

@inline function (signal::ModulatedGaussianPulse)(t)
    tau = (t - signal.delay) / signal.width
    theta = angular_frequency(signal) * (t - signal.delay) + signal.phase
    return exp(-tau^2) * sin(theta)
end

@inline function signal_derivative(signal::ModulatedGaussianPulse, t)
    tau = (t - signal.delay) / signal.width
    omega = angular_frequency(signal)
    theta = omega * (t - signal.delay) + signal.phase
    s, c = sincos(theta)
    return exp(-tau^2) * (omega * c - 2 * tau / signal.width * s)
end

@inline function signal_second_derivative(signal::ModulatedGaussianPulse, t)
    tau = (t - signal.delay) / signal.width
    omega = angular_frequency(signal)
    theta = omega * (t - signal.delay) + signal.phase
    s, c = sincos(theta)
    return exp(-tau^2) *
           (((4 * tau^2 - 2) / signal.width^2 - omega^2) * s -
            4 * tau * omega / signal.width * c)
end

"""
    signal_derivative(signal, t)
    signal_second_derivative(signal, t)

First and second time derivative of a signal.
"""
signal_derivative, signal_second_derivative
