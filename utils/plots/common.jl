# Shared setup for the figure scripts. Run from the package root with
#   julia --project=utils/plots utils/plots/<script>.jl
# after `Pkg.develop(path = ".")` in that environment.

using Trixi
using TrixiMaxwell
using CairoMakie
using StaticArrays: SVector
using LinearAlgebra: norm

const EXAMPLES_DIR = pkgdir(TrixiMaxwell, "examples", "dgmulti_3d")
const FIGURES_DIR = pkgdir(TrixiMaxwell, "docs", "figures")

elixir(name) = joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_$(name).jl")

# Runs an elixir in `Main` and returns its solution and semidiscretization;
# other elixir variables are read with `elixir_var(:name)`. The lookup goes
# through `invokelatest` because the bindings are created after this code was
# compiled.
function run_elixir(name; kwargs...)
    trixi_include(elixir(name); kwargs...)
    return (; sol = elixir_var(:sol), semi = elixir_var(:semi))
end

elixir_var(name) = Base.invokelatest(getglobal, Main, name)

# Heatmap of one variable on a plane through a 3D DGMulti solution. Uses the
# triangulation of Trixi's Makie extension directly so that the color range is
# shared between panels.
function slice_heatmap!(ax, u, semi, variable; slice = :xz, point = (0.0, 0.0, 0.0),
                        colorrange = nothing, colormap = Reverse(:RdBu),
                        plot_mesh = false)
    ext = Base.get_extension(Trixi, :TrixiMakieExt)
    pds = PlotData2D(u, semi; slice, point)[variable]
    values = getindex.(ext.global_plotting_triangulation_makie(pds).position, 3)
    triangulation = ext.global_plotting_triangulation_makie(pds;
                                                            set_z_coordinate_zero = true)
    if colorrange === nothing
        limit = maximum(abs, values)
        colorrange = (-limit, limit)
    end
    plt = mesh!(ax, triangulation; color = values, colormap, colorrange,
                shading = NoShading)
    if plot_mesh
        lines!(ax, ext.convert_PlotData2D_to_mesh_Points(pds; set_z_coordinate_zero = true);
               color = :lightgrey)
    end
    return plt
end

function save_figure(name, fig)
    path = joinpath(FIGURES_DIR, name)
    save(path, fig; px_per_unit = 2)
    println("saved ", path)
    return path
end
