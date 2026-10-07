# Gaussian pulse hitting a dielectric half space: reflected and transmitted parts.
include("common.jl")

times = (0.2, 0.5, 0.8)
fig = Figure(size = (1200, 260))
for (column, t_end) in enumerate(times)
    (; sol, semi) = run_elixir("fresnel"; tspan = (0.0, t_end))
    ax = Axis(fig[1, column], aspect = DataAspect(), title = "Ey at t = $t_end",
              xlabel = "x", ylabel = "z")
    slice_heatmap!(ax, sol.u[end], semi, "Ey"; slice = :xz, point = (0.0, 0.25, 0.0),
                   colorrange = (-1, 1))
    vlines!(ax, [0.0]; color = :black, linestyle = :dash)
    text!(ax, -0.9, 0.42; text = "ε = $(elixir_var(:epsilon_left))", fontsize = 14)
    text!(ax, 0.3, 0.42; text = "ε = $(elixir_var(:epsilon_right))", fontsize = 14)
end
Colorbar(fig[1, 4], colormap = Reverse(:RdBu), colorrange = (-1, 1), label = "Ey")
save_figure("fresnel.png", fig)
