# Plane-wave pulse scattering off a PEC sphere on the imported OpenSEMBA mesh:
# TF/SF injection, PEC surface and absorbing outer sphere acting together.
include("common.jl")
using Gmsh

times = (3.5, 5.5)
fig = Figure(size = (1000, 480))
for (column, t_end) in enumerate(times)
    (; sol, semi) = run_elixir("pec_sphere"; tspan = (0.0, t_end))
    ax = Axis(fig[1, column], aspect = DataAspect(), title = "Ex at t = $t_end",
              xlabel = "x", ylabel = "z")
    slice_heatmap!(ax, sol.u[end], semi, "Ex"; slice = :xz, point = (0.0, 0.0, 0.0),
                   colorrange = (-1, 1))
    angles = range(0, 2pi, length = 200)
    lines!(ax, cos.(angles), sin.(angles); color = :black, linewidth = 1.5)
    lines!(ax, [-1.5, 1.5, 1.5, -1.5, -1.5], [-1.5, -1.5, 1.5, 1.5, -1.5];
           color = :black, linestyle = :dash)
end
Colorbar(fig[1, 3], colormap = Reverse(:RdBu), colorrange = (-1, 1), label = "Ex")
save_figure("pec_sphere.png", fig)
