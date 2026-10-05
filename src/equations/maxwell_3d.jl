struct Homogeneous end
@doc raw"""
    Maxwell 3D:
     ∂t E = curl H / epsilon
     ∂t H = -curl E / mu
"""
struct MaxwellEquations3D{Material, RealT <: Real} <: Trixi.AbstractMaxwellEquations{3, 6}
    epsilon::RealT
    mu::RealT
end

function MaxwellEquations3D(; epsilon = 1.0, mu = 1.0)
    epsilon, mu = promote(epsilon, mu)
    return MaxwellEquations3D{Homogeneous, typeof(epsilon)}(epsilon, mu)
end

function Base.similar(equations::MaxwellEquations3D, ::Type{RealT}) where {RealT}
    return MaxwellEquations3D(epsilon = convert(RealT, equations.epsilon),
                              mu = convert(RealT, equations.mu))
end

permittivity(u, eq::MaxwellEquations3D{Homogeneous}) = eq.epsilon
permittivity(eq::MaxwellEquations3D{Homogeneous}) = eq.epsilon
permeability(u, eq::MaxwellEquations3D{Homogeneous}) = eq.mu
permeability(eq::MaxwellEquations3D{Homogeneous}) = eq.mu

impedance(u, eq) = sqrt(permeability(u, eq) / permittivity(u, eq))
impedance(eq) = sqrt(permeability(eq) / permittivity(eq))
speed_of_light(u, eq) = inv(sqrt(permittivity(u, eq) * permeability(u, eq)))
speed_of_light(eq) = inv(sqrt(permittivity(eq) * permeability(eq)))

function Trixi.varnames(::typeof(Trixi.cons2cons), ::MaxwellEquations3D)
    return ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
end

function Trixi.varnames(::typeof(Trixi.cons2prim), ::MaxwellEquations3D)
    return ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
end

@inline electric_field(u) = SVector(u[1], u[2], u[3])
@inline magnetic_field(u) = SVector(u[4], u[5], u[6])

@inline function Trixi.flux(u, normal_direction::AbstractVector,
                            equations::MaxwellEquations3D)
    E = electric_field(u)
    H = magnetic_field(u)
    eps = permittivity(u, equations)
    mu = permeability(u, equations)

    f_E = -cross(normal_direction, H) / eps
    f_H = cross(normal_direction, E) / mu

    return vcat(f_E, f_H)
end

@inline function unit_normal(orientation::Integer, ::Type{RealT}) where {RealT}
    if orientation == 1
        return SVector(one(RealT), zero(RealT), zero(RealT))
    elseif orientation == 2
        return SVector(zero(RealT), one(RealT), zero(RealT))
    else
        return SVector(zero(RealT), zero(RealT), one(RealT))
    end
end

@inline function Trixi.flux(u, orientation::Integer, equations::MaxwellEquations3D)
    return Trixi.flux(u, unit_normal(orientation, eltype(u)), equations)
end

struct FluxUpwindPenalty{RealT <: Real}
    alpha::RealT
end

const flux_upwind = FluxUpwindPenalty(1.0)

@inline function (numerical_flux::FluxUpwindPenalty)(u_ll, u_rr,
                                                     normal_direction::AbstractVector,
                                                     equations::MaxwellEquations3D)
    RealT = eltype(u_ll)
    alpha = convert(RealT, numerical_flux.alpha)

    E_ll = electric_field(u_ll)
    H_ll = magnetic_field(u_ll)
    E_rr = electric_field(u_rr)
    H_rr = magnetic_field(u_rr)

    eps = permittivity(u_ll, equations)
    mu = permeability(u_ll, equations)
    c = speed_of_light(u_ll, equations)

    norm_ = norm(normal_direction)
    n_hat = normal_direction / norm_

    E_avg = 0.5f0 * (E_ll + E_rr)
    H_avg = 0.5f0 * (H_ll + H_rr)

    dE = E_rr - E_ll
    dH = H_rr - H_ll
    dE_t = dE - dot(dE, n_hat) * n_hat
    dH_t = dH - dot(dH, n_hat) * n_hat

    penalty = 0.5f0 * alpha * c * norm_
    f_E = -cross(normal_direction, H_avg) / eps - penalty * dE_t
    f_H = cross(normal_direction, E_avg) / mu - penalty * dH_t

    return vcat(f_E, f_H)
end

@inline function (numerical_flux::FluxUpwindPenalty)(u_ll, u_rr, orientation::Integer,
                                                     equations::MaxwellEquations3D)
    return numerical_flux(u_ll, u_rr, unit_normal(orientation, eltype(u_ll)), equations)
end

function Base.show(io::IO, numerical_flux::FluxUpwindPenalty)
    print(io, "FluxUpwindPenalty(alpha=", numerical_flux.alpha, ")")
end

"""
    initial_condition_convergence_test(x, t, equations::MaxwellEquations3D)

Plane wave travelling in the positive x direction with unit period.

"""
function Trixi.initial_condition_convergence_test(x, t, equations::MaxwellEquations3D)
    c = speed_of_light(equations)
    Z = impedance(equations)
    g = sinpi(2 * (x[1] - c * t))
    z = zero(g)

    return SVector(z, g, z, z, z, g / Z)
end

@inline Trixi.cons2prim(u, ::MaxwellEquations3D) = u
@inline Trixi.cons2entropy(u, equations::MaxwellEquations3D) = vcat(permittivity(u,
                                                                                 equations) *
                                                                    electric_field(u),
                                                                    permeability(u,
                                                                                 equations) *
                                                                    magnetic_field(u))

function Trixi.energy_total(u, equations::MaxwellEquations3D)
    E = electric_field(u)
    H = magnetic_field(u)

    return 0.5f0 *
           (permittivity(u, equations) * dot(E, E) + permeability(u, equations) * dot(H, H))
end

@inline function Trixi.max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                           equations::MaxwellEquations3D)
    return max(speed_of_light(u_ll, equations), speed_of_light(u_rr, equations))
end

@inline function Trixi.max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                           equations::MaxwellEquations3D)
    return Trixi.max_abs_speed_naive(u_ll, u_rr, 1, equations) * norm(normal_direction)
end

@inline function Trixi.max_abs_speeds(u, equations::MaxwellEquations3D{Homogeneous})
    c = speed_of_light(u, equations)
    return c, c, c
end

@inline function Trixi.max_abs_speeds(equations::MaxwellEquations3D{Homogeneous})
    c = speed_of_light(equations)
    return c, c, c
end

@inline Trixi.have_constant_speed(::MaxwellEquations3D{Homogeneous}) = Trixi.True()

struct BoundaryConditionPerfectElectricConductor end
"""
    boundary_condition_perfect_electric_conductor = BoundaryConditionPerfectElectricConductor()

Perfect electric conductor. Mirrors the electric field using the surface flux.
"""
const boundary_condition_perfect_electric_conductor = BoundaryConditionPerfectElectricConductor()

@inline function (::BoundaryConditionPerfectElectricConductor)(u_inner,
                                                               normal_direction::AbstractVector,
                                                               x, t, surface_flux,
                                                               equations::MaxwellEquations3D)
    u_outer = vcat(-electric_field(u_inner), magnetic_field(u_inner))
    return surface_flux(u_inner, u_outer, normal_direction, equations)
end

struct BoundaryConditionPerfectMagneticConductor end
"""
    boundary_condition_perfect_magnetic_conductor = BoundaryConditionPerfectMagneticConductor()

Perfect magnetic conductor. Mirrors the magnetic field using the surface flux.
"""
const boundary_condition_perfect_magnetic_conductor = BoundaryConditionPerfectMagneticConductor()

@inline function (::BoundaryConditionPerfectMagneticConductor)(u_inner,
                                                               normal_direction::AbstractVector,
                                                               x, t, surface_flux,
                                                               equations::MaxwellEquations3D)
    u_outer = vcat(electric_field(u_inner), -magnetic_field(u_inner))
    return surface_flux(u_inner, u_outer, normal_direction, equations)
end

struct BoundaryConditionSilverMueller end
"""
    boundary_condition_silver_mueller = BoundaryConditionSilverMueller()

First-order absorption using Silver-Mueller boundary condition.
Implemented with a full upwind flux against a zero exterior state.
"""
const boundary_condition_silver_mueller = BoundaryConditionSilverMueller()

@inline function (::BoundaryConditionSilverMueller)(u_inner,
                                                    normal_direction::AbstractVector,
                                                    x, t, surface_flux,
                                                    equations::MaxwellEquations3D)
    return flux_upwind(u_inner, zero(u_inner), normal_direction, equations)
end

"""
    BoundaryConditionIncidentField(incident_field)

Describes the state from the outside on a boundary face with the function `incident_field(x, t, equations)`.
"""
struct BoundaryConditionIncidentField{F}
    incident_field::F
end

@inline function (boundary_condition::BoundaryConditionIncidentField)(u_inner,
                                                                      normal_direction::AbstractVector,
                                                                      x, t, surface_flux,
                                                                      equations::MaxwellEquations3D)
    u_outer = boundary_condition.incident_field(x, t, equations)
    return surface_flux(u_inner, u_outer, normal_direction, equations)
end
