module TestUnit

using Test
using Trixi
using TrixiMaxwell
using StaticArrays: SVector
using LinearAlgebra: norm, dot, cross

include("test_trixi.jl")

@testset "Unit tests" begin
#! format: noindent

@timed_testset "MaxwellEquations3D" begin
    equations = MaxwellEquations3D()

    @test equations isa MaxwellEquations3D{Homogeneous, Float64}
    @test equations isa Trixi.AbstractMaxwellEquations{3, 6}
    @test ndims(equations) == 3
    @test Trixi.nvariables(equations) == 6
    @test equations.epsilon == 1.0
    @test equations.mu == 1.0

    equations = MaxwellEquations3D(epsilon = 4, mu = 1.0)
    @test equations isa MaxwellEquations3D{Homogeneous, Float64}
    @test permittivity(equations) == 4.0
    @test permeability(equations) == 1.0
    @test impedance(equations) == 0.5
    @test TrixiMaxwell.speed_of_light(equations) == 0.5

    u = SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    @test permittivity(u, equations) == 4.0
    @test permeability(u, equations) == 1.0
    @test impedance(u, equations) == 0.5
    @test TrixiMaxwell.speed_of_light(u, equations) == 0.5

    equations32 = similar(equations, Float32)
    @test equations32 isa MaxwellEquations3D{Homogeneous, Float32}
    @test equations32.epsilon == 4.0f0

    expected_names = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    @test Trixi.varnames(Trixi.cons2cons, equations) == expected_names
    @test Trixi.varnames(Trixi.cons2prim, equations) == expected_names

    @test Trixi.have_constant_speed(equations) === Trixi.True()
end

@timed_testset "Physical flux" begin
    equations = MaxwellEquations3D(epsilon = 4.0, mu = 1.0)
    u = SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    @test flux(u, 1, equations) == SVector(0.0, 1.5, -1.25, 0.0, -3.0, 2.0)
    @test flux(u, 2, equations) == SVector(-1.5, 0.0, 1.0, 3.0, 0.0, -1.0)
    @test flux(u, 3, equations) == SVector(1.25, -1.0, 0.0, -2.0, 1.0, 0.0)

    for orientation in 1:3
        normal = SVector(ntuple(i -> i == orientation ? 1.0 : 0.0, 3))
        @test flux(u, normal, equations) == flux(u, orientation, equations)
    end

    normal = SVector(2.0, -1.0, 0.5)
    @test flux(u, normal, equations) == SVector(2.125, 2.5, -3.5, -4.0, -5.5, 5.0)

    # linear in the normal direction
    @test flux(u, 3 * normal, equations) ≈ 3 * flux(u, normal, equations)
end

@timed_testset "Wave speeds" begin
    equations = MaxwellEquations3D(epsilon = 4.0, mu = 1.0)
    u_ll = SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    u_rr = -u_ll

    for orientation in 1:3
        @test max_abs_speed_naive(u_ll, u_rr, orientation, equations) == 0.5
    end

    # norm(normal) = 3
    normal = SVector(2.0, -1.0, 2.0)
    @test max_abs_speed_naive(u_ll, u_rr, normal, equations) == 1.5
    @test Trixi.max_abs_speed(u_ll, u_rr, normal, equations) == 1.5

    @test Trixi.max_abs_speeds(equations) == (0.5, 0.5, 0.5)
    @test Trixi.max_abs_speeds(u_ll, equations) == (0.5, 0.5, 0.5)
end

@timed_testset "Upwind flux" begin
    equations = MaxwellEquations3D(epsilon = 4.0, mu = 1.0)
    c = TrixiMaxwell.speed_of_light(equations)
    flux_central_penalty = FluxUpwindPenalty(0.0)

    u_ll = SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    u_rr = SVector(-2.0, 0.5, 4.0, -1.0, 3.0, -0.5)
    normal = SVector(2.0, -1.0, 2.0)
    n_hat = normal / norm(normal)

    @test flux_upwind isa FluxUpwindPenalty
    @test flux_upwind.alpha == 1.0

    # consistency
    @test flux_upwind(u_ll, u_ll, normal, equations) == flux(u_ll, normal, equations)
    for orientation in 1:3
        @test flux_upwind(u_ll, u_ll, orientation, equations) ==
              flux(u_ll, orientation, equations)
        @test flux_upwind(u_ll, u_rr, orientation, equations) ==
              flux_upwind(u_ll, u_rr,
                          SVector(ntuple(i -> i == orientation ? 1.0 : 0.0, 3)),
                          equations)
    end

    # symmetry
    @test flux_upwind(u_ll, u_rr, normal, equations) ≈
          -flux_upwind(u_rr, u_ll, -normal, equations)

    # homogeneous of degree one in the normal direction
    @test flux_upwind(u_ll, u_rr, 3 * normal, equations) ≈
          3 * flux_upwind(u_ll, u_rr, normal, equations)

    # alpha = 0 is the central flux
    @test flux_central_penalty(u_ll, u_rr, normal, equations) ≈
          0.5 * (flux(u_ll, normal, equations) + flux(u_rr, normal, equations))

    # jumps in the normal components carry no penalty
    normal_jump = vcat(0.7 * n_hat, -1.3 * n_hat)
    @test flux_upwind(u_ll, u_ll + normal_jump, normal, equations) ≈
          flux_central_penalty(u_ll, u_ll + normal_jump, normal, equations)

    # tangential jumps are penalized with speed c, like Lax-Friedrichs
    tangent = cross(n_hat, SVector(0.0, 0.0, 1.0))
    tangential_jump = vcat(0.7 * tangent, -1.3 * tangent)
    @test flux_upwind(u_ll, u_ll + tangential_jump, normal, equations) ≈
          flux_lax_friedrichs(u_ll, u_ll + tangential_jump, normal, equations)

    # general jump: penalty acts on the tangential projection only
    dE = SVector(u_rr[1] - u_ll[1], u_rr[2] - u_ll[2], u_rr[3] - u_ll[3])
    dH = SVector(u_rr[4] - u_ll[4], u_rr[5] - u_ll[5], u_rr[6] - u_ll[6])
    dE_t = dE - dot(dE, n_hat) * n_hat
    dH_t = dH - dot(dH, n_hat) * n_hat
    @test flux_upwind(u_ll, u_rr, normal, equations) -
          flux_central_penalty(u_ll, u_rr, normal, equations) ≈
          -0.5 * c * norm(normal) * vcat(dE_t, dH_t)

    # alpha scales the penalty linearly
    @test FluxUpwindPenalty(0.5)(u_ll, u_rr, normal, equations) ≈
          0.5 * (flux_upwind(u_ll, u_rr, normal, equations) +
           flux_central_penalty(u_ll, u_rr, normal, equations))

    @test sprint(show, FluxUpwindPenalty(0.25)) == "FluxUpwindPenalty(alpha=0.25)"
end

@timed_testset "Energy" begin
    equations = MaxwellEquations3D(epsilon = 4.0, mu = 1.0)
    u = SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    @test cons2prim(u, equations) == u
    @test cons2entropy(u, equations) == SVector(4.0, 8.0, 12.0, 4.0, 5.0, 6.0)
    @test energy_total(u, equations) == 66.5
    @test dot(cons2entropy(u, equations), u) ≈ 2 * energy_total(u, equations)
end

@timed_testset "Plane-wave initial condition" begin
    equations = MaxwellEquations3D(epsilon = 4.0, mu = 1.0)
    c = TrixiMaxwell.speed_of_light(equations)
    Z = impedance(equations)
    initial_condition = initial_condition_convergence_test

    # quarter period: sin(2 pi x) = 1
    u_peak = initial_condition(SVector(0.25, 0.0, 0.0), 0.0, equations)
    @test u_peak ≈ SVector(0.0, 1.0, 0.0, 0.0, 0.0, 2.0)

    x = SVector(1 / 8, 1 / 3, 2 / 5)
    t = 0.125
    u = initial_condition(x, t, equations)
    x_shifted = SVector(x[1] - c * t, x[2], x[3])
    @test u ≈ initial_condition(x_shifted, 0.0, equations)

    E = SVector(u[1], u[2], u[3])
    H = SVector(u[4], u[5], u[6])
    @test norm(E) ≈ Z * norm(H)
    @test cross(E, H)[1] > 0
    @test cross(E, H)[2] == 0
    @test cross(E, H)[3] == 0

    # unit period in x
    @test initial_condition(SVector(0.3, 0.0, 0.0), 0.0, equations) ≈
          initial_condition(SVector(1.3, 0.0, 0.0), 0.0, equations)
end
end

end # module
