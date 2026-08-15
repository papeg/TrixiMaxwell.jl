using Test
using LinearAlgebra: norm
using Trixi
using TrixiMaxwell

@testset "TrixiMaxwell" begin
    @test TrixiMaxwell isa Module
    @test isdefined(TrixiMaxwell, :Trixi)
end

include("test_maxwell_3d.jl")
