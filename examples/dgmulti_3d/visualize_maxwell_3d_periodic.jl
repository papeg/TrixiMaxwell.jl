using GLMakie
using Trixi

GLMakie.activate!()

frame_times = range(0.0, 1.0, length = 121)

trixi_include(
    @__MODULE__,
    joinpath(@__DIR__, "elixir_maxwell_3d_periodic.jl");
    tspan = (0.0, 1.0),
    saveat = frame_times
)

slice_z = 0.5
slice_point = (0.0, 0.0, slice_z)

function slice_plot_data(u_ode, semi, point)
    return PlotData2D(u_ode, semi;
                      slice = :xy,
                      point = point,
                      solution_variables = cons2cons)
end

function variable_values(plot_data, variable_id)
    return Float32[plot_data.data[index][variable_id]
                   for index in eachindex(plot_data.data)]
end

initial_plot_data = slice_plot_data(sol.u[1], semi, slice_point)
electric_y = initial_plot_data["Ey"]
time = Observable(sol.t[1])

title = lift(time) do t
    return "Periodic Maxwell plane wave E_y at z = $slice_z, t = $(round(t; digits = 3))"
end

figure = Figure(size = (900, 700))

axis = Axis(figure[1, 1];
            title = title,
            xlabel = "x",
            ylabel = "y",
            aspect = DataAspect(),
            limits = (extrema(initial_plot_data.x)...,
                      extrema(initial_plot_data.y)...))

field_plot = trixiheatmap!(axis, electric_y;
                           plot_mesh = true,
                           colormap = :balance)
field_mesh = field_plot.plots[1]
field_mesh.colorrange[] = (-1.0f0, 1.0f0)

Colorbar(figure[1, 2], field_mesh;
         label = "E_y")

video_path = joinpath(@__DIR__, "maxwell_plane_wave.mp4")

record(figure, video_path, eachindex(sol.t);
       framerate = 30) do frame
    plot_data = slice_plot_data(sol.u[frame], semi, slice_point)
    field_mesh.color[] = variable_values(plot_data, electric_y.variable_id)
    time[] = sol.t[frame]
end

println("Video written to: ", video_path)
