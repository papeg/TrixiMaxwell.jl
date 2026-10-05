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

@trixi_testset "elixir_maxwell_3d_cavity.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_cavity.jl"),
                        tspan=(0.0, 0.2))
    @test maximum(analysis_callback(sol).l2) < 5e-3
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "elixir_maxwell_3d_cavity.jl (central flux, energy conservation)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_cavity.jl"),
                        surface_flux=FluxUpwindPenalty(0.0), tspan=(0.0, 0.5))

    # semidiscrete energy derivative vanishes to rounding
    u = sol.u[end]
    du = similar(u)
    Trixi.rhs_hyperbolic!(du, u, semi, sol.t[end])
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    energy_rate = Trixi.analyze(Trixi.entropy_timederivative,
                                Trixi.wrap_array(du, semi), Trixi.wrap_array(u, semi),
                                sol.t[end], mesh, equations, solver, cache)
    energy = Trixi.integrate(energy_total, u, semi, normalize = false)
    @test abs(energy_rate) < 1e-12 * energy

    # fully discrete drift is the RK error and shrinks with the time step
    energy_start = Trixi.integrate(energy_total, sol.u[1], semi, normalize = false)
    drift_coarse = abs(energy - energy_start) / energy_start
    @test drift_coarse < 1e-6

    trixi_include(@__MODULE__, joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_cavity.jl"),
                  surface_flux = FluxUpwindPenalty(0.0), tspan = (0.0, 0.5), cfl = 0.25)
    energy_fine = Trixi.integrate(energy_total, sol.u[end], semi, normalize = false)
    drift_fine = abs(energy_fine - energy_start) / energy_start
    @test drift_fine < drift_coarse / 8
end

@trixi_testset "elixir_maxwell_3d_cavity.jl (convergence)" begin
    using Trixi, TrixiMaxwell
    # Ez, Hx, Hy carry the mode; Ex, Ey, Hz are zero in the exact solution
    mode_components = (3, 4, 5)
    for polydeg in (1, 2)
        eocs, _ = Trixi.convergence_test(@__MODULE__,
                                         joinpath(EXAMPLES_DIR,
                                                  "elixir_maxwell_3d_cavity.jl"),
                                         3; polydeg, cells_per_dimension = (2, 2, 2),
                                         cfl = 0.1, tspan = (0.0, 0.5))
        # the coarsest pair is pre-asymptotic, judge the finest refinement step
        eoc_finest = eocs[:l2][end, :]
        @test all(eoc_finest[c] > polydeg + 0.75 for c in mode_components)
    end
end
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true, force = true)

end # module
