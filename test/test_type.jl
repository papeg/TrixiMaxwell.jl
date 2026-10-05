module TestType

using Test
using Trixi
using TrixiMaxwell
using StaticArrays: SVector

include("test_trixi.jl")

@testset "Test Type Stability" begin
    @timed_testset "Maxwell 3D" begin
        for RealT in (Float32, Float64)
            equations = @inferred MaxwellEquations3D(epsilon = one(RealT), mu = one(RealT))
            @test equations isa MaxwellEquations3D{Homogeneous, RealT}
            @test (@inferred similar(equations, RealT)) isa
                  MaxwellEquations3D{Homogeneous, RealT}

            x = SVector(zero(RealT), zero(RealT), zero(RealT))
            t = zero(RealT)
            u = u_ll = u_rr = SVector(ntuple(_ -> one(RealT), 6))
            normal_direction = SVector(one(RealT), one(RealT), zero(RealT))

            @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
                  RealT

            for orientation in 1:3
                @test eltype(@inferred flux(u, orientation, equations)) == RealT
                @test eltype(@inferred flux_upwind(u_ll, u_rr, orientation, equations)) ==
                      RealT
                @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                           equations)) == RealT
            end
            @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
            @test eltype(@inferred flux_upwind(u_ll, u_rr, normal_direction, equations)) ==
                  RealT
            @test eltype(@inferred FluxUpwindPenalty(0.0)(u_ll, u_rr, normal_direction,
                                                          equations)) == RealT
            @test eltype(@inferred flux_lax_friedrichs(u_ll, u_rr, normal_direction,
                                                       equations)) == RealT
            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
                                                       equations)) == RealT
            @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
            @test eltype(@inferred Trixi.max_abs_speeds(equations)) == RealT

            @test typeof(@inferred permittivity(u, equations)) == RealT
            @test typeof(@inferred permeability(u, equations)) == RealT
            @test typeof(@inferred impedance(u, equations)) == RealT
            @test typeof(@inferred TrixiMaxwell.speed_of_light(u, equations)) == RealT

            for boundary_condition in (boundary_condition_perfect_electric_conductor,
                                       boundary_condition_perfect_magnetic_conductor,
                                       boundary_condition_silver_mueller,
                                       BoundaryConditionIncidentField(initial_condition_convergence_test))
                @test eltype(@inferred boundary_condition(u, normal_direction, x, t,
                                                          flux_upwind, equations)) == RealT
            end

            @test eltype(@inferred cons2prim(u, equations)) == RealT
            @test eltype(@inferred cons2entropy(u, equations)) == RealT
            @test typeof(@inferred energy_total(u, equations)) == RealT
        end
    end
end

end # module
