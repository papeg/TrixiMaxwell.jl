using OrdinaryDiffEqLowStorageRK
using Trixi
using TrixiMaxwell

equations = MaxwellEquations3D()

function initial_condition_cavity(x, t, equations::MaxwellEquations3D)
    RealT = eltype(x)
    omega = convert(RealT, sqrt(2) * pi)
    amplitude = convert(RealT, pi) / omega
    
    Ez = sinpi(x[1]) * sinpi(x[2]) * cos(omega * t)
    Hx = -amplitude * sinpi(x[1]) * cospi(x[2]) * sin(omega * t)
    Hy = amplitude * cospi(x[1]) * sinpi(x[2]) * sin(omega * t)
    z = zero(Ez)

    return SVector(z, z, Ez, Hx, Hy, z)
end

initial_condition = initial_condition_cavity

boundary_conditions = (; entire_boundary = boundary_condition_perfect_electric_conductor)

polydeg = 3
surface_flux = flux_upwind

solver = DGMulti(polydeg = polydeg,
        element_type = Tet(),
        approximation_type = Polynomial(),
        surface_integral = SurfaceIntegralWeakForm(surface_flux),
        volume_integral = VolumeIntegralWeakForm())

cells_per_dimension = (4, 4, 4)
mesh = DGMultiMesh(solver, cells_per_dimension;
    coordinates_min = (-1.0, -1.0, -1.0),
    coordinates_max = (1.0, 1.0, 1.0))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver; boundary_conditions)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
alive_callback = AliveCallback(analysis_interval = analysis_interval)

cfl = 0.5
stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, stepsize_callback)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
    dt = 1.0, # overwritten by stepsize callback
    ode_default_options()..., callback = callbacks)

