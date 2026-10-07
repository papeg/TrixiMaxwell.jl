# PEC cavity mode on the structured tet cube: Ez and Hx on the midplane.
include("common.jl")

(; sol, semi) = run_elixir("cavity"; tspan = (0.0, 0.25))

fig = Figure(size = (900, 400))
for (column, variable) in enumerate(("Ez", "Hx"))
    ax = Axis(fig[1, column], aspect = DataAspect(),
              title = "$variable at t = $(sol.t[end])",
              xlabel = "x", ylabel = "y")
    plt = slice_heatmap!(ax, sol.u[end], semi, variable; slice = :xy,
                         point = (0.5, 0.5, 0.5),
                         plot_mesh = true)
    Colorbar(fig[2, column], plt, vertical = false)
end
save_figure("cavity_mode.png", fig)
