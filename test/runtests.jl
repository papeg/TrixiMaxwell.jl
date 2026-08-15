using Test
using Trixi
using TrixiMaxwell

@testset "TrixiMaxwell" begin
    @test TrixiMaxwell isa Module
    @test isdefined(TrixiMaxwell, :Trixi)
    include("test_maxwell_3d.jl")
end
