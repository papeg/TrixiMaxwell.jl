"""
    SaveVtkCallback(; interval = 0, dt = nothing,
                    save_initial_solution = true, save_final_solution = true,
                    output_directory = "out", filename = "solution",
                    solution_variables = cons2prim)

Write the solution as VTK files during the simulation, every `interval`
accepted time steps or every `dt` in simulation time, through
[`write_solution_vtk`](@ref). The files `<filename>_<step>.vtu` and the ParaView
collection `<filename>.pvd` are written to `output_directory`; the collection is
updated after every output, so a running simulation can be inspected.
"""
mutable struct SaveVtkCallback{IntervalType, SolutionVariablesType}
    interval_or_dt::IntervalType
    save_initial_solution::Bool
    save_final_solution::Bool
    output_directory::String
    filename::String
    solution_variables::SolutionVariablesType
    entries::Vector{Tuple{Float64, String}}
end

function SaveVtkCallback(; interval::Integer = 0, dt = nothing,
                         save_initial_solution = true, save_final_solution = true,
                         output_directory = "out", filename = "solution",
                         solution_variables = Trixi.cons2prim)
    if !isnothing(dt) && interval > 0
        throw(ArgumentError("You can either set the number of steps between output (using `interval`) or the time between outputs (using `dt`) but not both simultaneously"))
    end
    interval_or_dt = isnothing(dt) ? interval : dt
    vtk_callback = SaveVtkCallback(interval_or_dt, save_initial_solution,
                                   save_final_solution, String(output_directory),
                                   String(filename), solution_variables,
                                   Tuple{Float64, String}[])

    if isnothing(dt)
        return Trixi.DiscreteCallback(vtk_callback, vtk_callback,
                                      save_positions = (false, false),
                                      initialize = initialize_save_vtk!)
    else
        return Trixi.PeriodicCallback(vtk_callback, dt,
                                      save_positions = (false, false),
                                      initialize = initialize_save_vtk!,
                                      final_affect = save_final_solution)
    end
end

function initialize_save_vtk!(cb, u, t, integrator)
    return initialize_save_vtk!(cb.affect!, u, t, integrator)
end

function initialize_save_vtk!(vtk_callback::SaveVtkCallback, u, t, integrator)
    mkpath(vtk_callback.output_directory)
    empty!(vtk_callback.entries)
    if vtk_callback.save_initial_solution
        vtk_callback(integrator)
    end
    return nothing
end

# condition
function (vtk_callback::SaveVtkCallback)(u, t, integrator)
    (; interval_or_dt, save_final_solution) = vtk_callback
    return interval_or_dt > 0 && (integrator.stats.naccept % interval_or_dt == 0 ||
            (save_final_solution && Trixi.isfinished(integrator)))
end

# affect!
function (vtk_callback::SaveVtkCallback)(integrator)
    (; output_directory, filename, solution_variables, entries) = vtk_callback
    step = integrator.stats.naccept
    base = joinpath(output_directory, filename * "_" * lpad(step, 9, '0'))
    files = write_solution_vtk(integrator.u, integrator.p, base; solution_variables)
    push!(entries, (Float64(integrator.t), basename(first(files))))
    write_pvd(joinpath(output_directory, filename * ".pvd"), entries)
    Trixi.derivative_discontinuity!(integrator, false)
    return nothing
end

function Base.show(io::IO, cb::Trixi.DiscreteCallback{<:Any, <:SaveVtkCallback})
    @nospecialize cb
    print(io, "SaveVtkCallback(interval=", cb.affect!.interval_or_dt, ")")
    return nothing
end

function Base.show(io::IO,
                   cb::Trixi.DiscreteCallback{<:Any,
                                              <:Trixi.PeriodicCallbackAffect{<:SaveVtkCallback}})
    @nospecialize cb
    print(io, "SaveVtkCallback(dt=", cb.affect!.affect!.interval_or_dt, ")")
    return nothing
end

function summary_setup(vtk_callback::SaveVtkCallback, key)
    return [
            key => vtk_callback.interval_or_dt,
            "solution variables" => vtk_callback.solution_variables,
            "save initial solution" => vtk_callback.save_initial_solution ? "yes" : "no",
            "save final solution" => vtk_callback.save_final_solution ? "yes" : "no",
            "file name" => vtk_callback.filename,
            "output directory" => abspath(normpath(vtk_callback.output_directory))
            ]
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::Trixi.DiscreteCallback{<:Any, <:SaveVtkCallback})
    @nospecialize cb
    if get(io, :compact, false)
        show(io, cb)
    else
        Trixi.summary_box(io, "SaveVtkCallback", summary_setup(cb.affect!, "interval"))
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::Trixi.DiscreteCallback{<:Any,
                                              <:Trixi.PeriodicCallbackAffect{<:SaveVtkCallback}})
    @nospecialize cb
    if get(io, :compact, false)
        show(io, cb)
    else
        Trixi.summary_box(io, "SaveVtkCallback", summary_setup(cb.affect!.affect!, "dt"))
    end
    return nothing
end
