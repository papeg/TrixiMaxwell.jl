# Pulse scattering off a dielectric sphere on the OpenSEMBA sphere-in-box geometry.
include("common.jl")
using Gmsh

times = (1.5, 2.5, 3.5)
fig = Figure(size = (1200, 420))
for (column, t_end) in enumerate(times)
    (; sol, semi) = run_elixir("dielectric_sphere"; tspan = (0.0, t_end), size_factor = 2.0)
    sphere_radius = elixir_var(:sphere_radius)
    ax = Axis(fig[1, column], aspect = DataAspect(), title = "Ex at t = $t_end",
              xlabel = "x", ylabel = "z")
    slice_heatmap!(ax, sol.u[end], semi, "Ex"; slice = :xz, point = (0.0, 0.0, 0.0),
                   colorrange = (-1, 1))
    angles = range(0, 2pi, length = 200)
    lines!(ax, sphere_radius .* cos.(angles), sphere_radius .* sin.(angles);
           color = :black, linewidth = 1.5)
end
Colorbar(fig[1, 4], colormap = Reverse(:RdBu), colorrange = (-1, 1), label = "Ex")
save_figure("dielectric_sphere.png", fig)
