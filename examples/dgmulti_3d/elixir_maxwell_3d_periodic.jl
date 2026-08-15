using OrdinaryDiffEqLowStorageRK
using Trixi
using TrixiMaxwell

equations = MaxwellEquations3D(1.0)
initial_condition = initial_condition_convergence_test

solver = DGMulti(polydeg = 3,
                 element_type = Tet(),
                 approximation_type = Polynomial(),
                 surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
                 volume_integral = VolumeIntegralWeakForm())

cells_per_dimension = (2, 2, 2)

mesh = DGMultiMesh(solver, cells_per_dimension;
                   coordinates_min = (0.0, 0.0, 0.0),
                   coordinates_max = (1.0, 1.0, 1.0),
                   periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)

tspan = (0.0, 0.25)
ode = semidiscretize(semi, tspan)

stepsize_callback = StepsizeCallback(cfl = 0.5)
callbacks = CallbackSet(
    SummaryCallback(),
    AnalysisCallback(semi, interval = 10),
    AliveCallback(alive_interval = 10),
    stepsize_callback
)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = stepsize_callback(ode),
            ode_default_options()...,
            callback = callbacks)
