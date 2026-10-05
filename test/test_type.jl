module TestType

using Test
using Trixi
using TrixiMaxwell
using StaticArrays: SVector

include("test_trixi.jl")

@testset "Test Type Stability" begin
    @timed_testset "Maxwell 3D" begin
        for RealT in (Float32, Float64)
            equations = @inferred MaxwellEquations3D(one(RealT))

            x = SVector(zero(RealT), zero(RealT), zero(RealT))
            t = zero(RealT)
            u = u_ll = u_rr = SVector(ntuple(_ -> one(RealT), 6))
            normal_direction = SVector(one(RealT), zero(RealT), zero(RealT))

            @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
                  RealT
            for orientation in 1:3
                @test eltype(@inferred flux(u, orientation, equations)) == RealT
                @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                           equations)) == RealT
            end
            @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
                                                       equations)) == RealT
            @test eltype(@inferred cons2prim(u, equations)) == RealT
            @test eltype(@inferred cons2entropy(u, equations)) == RealT
        end
    end
end

end # module
