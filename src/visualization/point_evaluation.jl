"""
    PointEvaluator(points, semi)

Interpolation of a DGMulti solution on straight-sided tetrahedra to `points`.
Each point is located in its element once; `evaluator(u_ode, semi)` then returns
the state at every point.
"""
struct PointEvaluator{RealT <: Real}
    elements::Vector{Int}
    interpolation::Matrix{RealT}
end

function PointEvaluator(points, semi)
    mesh, _, dg, _ = Trixi.mesh_equations_solver_cache(semi)
    rd = dg.basis
    md = mesh.md
    RealT = eltype(md.xyz[1])

    reference_vertices = ([-1, 1, -1, -1], [-1, -1, 1, -1], [-1, -1, -1, 1])
    to_vertices = StartUpDG.vandermonde(rd.element_type, rd.N, reference_vertices...) /
                  rd.VDM

    elements = Int[]
    interpolation = zeros(RealT, length(points), size(rd.VDM, 1))
    for (p, point) in enumerate(points)
        element, reference_coordinates = locate_point(SVector{3, RealT}(point), md,
                                                      to_vertices)
        push!(elements, element)
        interpolation[p, :] = StartUpDG.vandermonde(rd.element_type, rd.N,
                                                    map(c -> [c],
                                                        reference_coordinates)...) /
                              rd.VDM
    end
    return PointEvaluator(elements, interpolation)
end

function locate_point(point, md, to_vertices; tolerance = 1.0e-10)
    for element in Base.OneTo(md.num_elements)
        coordinates = ntuple(d -> to_vertices * view(md.xyz[d], :, element), 3)
        vertex(v) = SVector(coordinates[1][v], coordinates[2][v], coordinates[3][v])
        v1 = vertex(1)
        edges = hcat(vertex(2) - v1, vertex(3) - v1, vertex(4) - v1)
        lambda = edges \ (point - v1)
        if all(lambda .>= -tolerance) && sum(lambda) <= 1 + tolerance
            return element, 2 * lambda .- 1
        end
    end
    throw(ArgumentError("point $point lies outside the mesh"))
end

function (evaluator::PointEvaluator)(u_ode, semi)
    u = Trixi.wrap_array(u_ode, semi)
    return map(eachindex(evaluator.elements)) do p
        element = evaluator.elements[p]
        sum(evaluator.interpolation[p, i] * u[i, element] for i in axes(u, 1))
    end
end
