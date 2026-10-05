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
                        tspan=(0.0, 0.1),
                        l2=[
                            0.0028552004428660524,
                            0.014404338850014405,
                            0.0006299587093177779,
                            0.003258927909210615,
                            0.0005939131970414426,
                            0.014325582509961483
                        ],
                        linf=[
                            0.013776115061273903,
                            0.06310569285709777,
                            0.003609935919698594,
                            0.01512872608359634,
                            0.002450788404007798,
                            0.05902331747400852
                        ])
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true, force = true)

end # module
