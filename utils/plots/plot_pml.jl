# Reflections after the dipole pulse has left: PML versus Silver-Mueller only.
include("common.jl")

fig = Figure(size = (1000, 460))
sm_dipole = HertzianDipole((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.1,
                           GaussianPulse(0.4; delay = 1.4))
cases = (("uniaxial PML", (;)),
         ("Silver-Mueller only",
          (; equations = MaxwellEquations3D(), source_terms = sm_dipole)))
for (column, (title, kwargs)) in enumerate(cases)
    (; sol, semi) = run_elixir("pml"; kwargs...)
    ax = Axis(fig[1, column], aspect = DataAspect(),
              title = "$title, Ez at t = $(sol.t[end])",
              xlabel = "x", ylabel = "z")
    slice_heatmap!(ax, sol.u[end], semi, "Ez"; slice = :xz, point = (0.0, 0.0, 0.0),
                   colorrange = (-0.01, 0.01))
    inner = 1.5 - elixir_var(:pml_thickness)
    lines!(ax, [-inner, inner, inner, -inner, -inner],
           [-inner, -inner, inner, inner, -inner];
           color = :black, linestyle = :dash)
end
Colorbar(fig[1, 3], colormap = Reverse(:RdBu), colorrange = (-0.01, 0.01), label = "Ez")
save_figure("pml.png", fig)
