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
                            0.004026282519168456,
                            0.013830965800949144,
                            0.0013561586607982563,
                            0.004587291244711452,
                            0.0013306309232592418,
                            0.013525193474118773
                        ],
                        linf=[
                            0.018962436875544983,
                            0.09160119756625695,
                            0.012413018884185844,
                            0.026963233136926276,
                            0.008512663708931854,
                            0.08438493308564704
                        ])
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "elixir_maxwell_3d_cavity.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_cavity.jl"),
                        tspan=(0.0, 0.2),
                        l2=[
                            0.0007793129472875638,
                            0.0007826285312641698,
                            0.0020419344903954604,
                            0.0015324399388593466,
                            0.001560580391670923,
                            0.0004480688735142423
                        ],
                        linf=[
                            0.007836853414515583,
                            0.011000334046308386,
                            0.02857835700820057,
                            0.007525831485687683,
                            0.011446976834034457,
                            0.00534208660038957
                        ])
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "elixir_maxwell_3d_cavity.jl (central flux, energy conservation)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_cavity.jl"),
                        surface_flux=FluxUpwindPenalty(0.0), tspan=(0.0, 0.5),
                        l2=[
                            0.0031108530820131036,
                            0.003106392979200707,
                            0.003633740045071118,
                            0.003699456386487258,
                            0.00377399785887293,
                            0.003049410562590979
                        ],
                        linf=[
                            0.05717224250663998,
                            0.04801641087014596,
                            0.04885040982352329,
                            0.0371122790946351,
                            0.039151855392283566,
                            0.03687370171164024
                        ])

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

@trixi_testset "elixir_maxwell_3d_silver_mueller.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_silver_mueller.jl"),
                        l2=[
                            0.00117439020493943,
                            0.001708558869573452,
                            0.0010304124230123169,
                            0.0015951257294803507,
                            0.0014479087538807533,
                            0.002192990311656334
                        ],
                        linf=[
                            0.006127871499208194,
                            0.012858598940748847,
                            0.0067441046663994884,
                            0.010557906689524062,
                            0.008339618562718715,
                            0.018482229839108192
                        ])
    # the pulse has left the box, what remains is projection error, not reflection
    energy_start = Trixi.integrate(energy_total, sol.u[1], semi)
    energy_end = Trixi.integrate(energy_total, sol.u[end], semi)
    @test energy_end < 2e-4 * energy_start
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true, force = true)

end # module
