module TestExamplesDGMulti3D

using Test
using Trixi
using TrixiMaxwell

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(TrixiMaxwell, "examples", "dgmulti_3d")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "DGMulti 3D" begin
#! format: noindent

@trixi_testset "elixir_maxwell_3d_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_periodic.jl"),
                        tspan=(0.0, 0.1))
    @test maximum(analysis_callback(sol).linf) < 0.1
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true, force = true)

end # module
