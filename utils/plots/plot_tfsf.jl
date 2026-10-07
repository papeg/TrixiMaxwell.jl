# Total-field/scattered-field box in free space: the pulse exists only inside the box.
include("common.jl")

times = (1.2, 1.6, 2.0)
fig = Figure(size = (1200, 400))
for (column, t_end) in enumerate(times)
    (; sol, semi) = run_elixir("tfsf"; tspan = (0.0, t_end))
    ax = Axis(fig[1, column], aspect = DataAspect(), title = "Ez at t = $t_end",
              xlabel = "x", ylabel = "y")
    slice_heatmap!(ax, sol.u[end], semi, "Ez"; slice = :xy, point = (0.0, 0.0, 0.0),
                   colorrange = (-1, 1))
    lines!(ax, [-0.5, 0.5, 0.5, -0.5, -0.5], [-0.5, -0.5, 0.5, 0.5, -0.5];
           color = :black, linestyle = :dash)
end
Colorbar(fig[1, 4], colormap = Reverse(:RdBu), colorrange = (-1, 1), label = "Ez")
save_figure("tfsf.png", fig)
