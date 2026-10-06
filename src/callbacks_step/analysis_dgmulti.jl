function Trixi.default_analysis_integrals(::MaxwellEquations3D)
    return (entropy_timederivative, energy_total,
            Val(:l2_dive), Val(:linf_dive), Val(:l2_divh), Val(:linf_divh))
end

function l2_divergence(components, mesh::DGMultiMesh, dg::DGMulti, cache)
    (; md) = mesh
    rd = dg.basis
    uEltype = eltype(first(components))
    local_divergence = zeros(uEltype, size(first(components), 1))
    l2norm_squared = zero(uEltype)
    for element in Trixi.eachelement(mesh, dg, cache)
        Trixi.compute_local_divergence!(local_divergence, element,
                                        view.(components, :, element), mesh, dg, cache)
        l2norm_squared += md.J[1, element] * dot(local_divergence, rd.M, local_divergence)
    end
    return sqrt(l2norm_squared)
end

function linf_divergence(components, mesh::DGMultiMesh, dg::DGMulti, cache)
    uEltype = eltype(first(components))
    local_divergence = zeros(uEltype, size(first(components), 1))
    linfnorm = zero(uEltype)
    for element in Trixi.eachelement(mesh, dg, cache)
        Trixi.compute_local_divergence!(local_divergence, element,
                                        view.(components, :, element), mesh, dg, cache)
        linfnorm = max(linfnorm, maximum(abs, local_divergence))
    end
    return linfnorm
end

function field_components(u, indices::NTuple{N, Int}) where {N}
    return ntuple(i -> Trixi.get_component(u, indices[i]), Val(N))
end

const electric_field_indices = (1, 2, 3)
const magnetic_field_indices = (4, 5, 6)

function Trixi.analyze(::Val{:l2_dive}, du, u, t, mesh::DGMultiMesh,
                       equations::MaxwellEquations3D, dg::DGMulti, cache)
    return l2_divergence(field_components(u, electric_field_indices), mesh, dg, cache)
end

function Trixi.analyze(::Val{:linf_dive}, du, u, t, mesh::DGMultiMesh,
                       equations::MaxwellEquations3D, dg::DGMulti, cache)
    return linf_divergence(field_components(u, electric_field_indices), mesh, dg, cache)
end

function Trixi.analyze(::Val{:l2_divh}, du, u, t, mesh::DGMultiMesh,
                       equations::MaxwellEquations3D, dg::DGMulti, cache)
    return l2_divergence(field_components(u, magnetic_field_indices), mesh, dg, cache)
end

function Trixi.analyze(::Val{:linf_divh}, du, u, t, mesh::DGMultiMesh,
                       equations::MaxwellEquations3D, dg::DGMulti, cache)
    return linf_divergence(field_components(u, magnetic_field_indices), mesh, dg, cache)
end

Trixi.pretty_form_utf(::Val{:l2_dive}) = "L2 ∇⋅E"
Trixi.pretty_form_ascii(::Val{:l2_dive}) = "l2_dive"
Trixi.pretty_form_utf(::Val{:linf_dive}) = "L∞ ∇⋅E"
Trixi.pretty_form_ascii(::Val{:linf_dive}) = "linf_dive"
Trixi.pretty_form_utf(::Val{:l2_divh}) = "L2 ∇⋅H"
Trixi.pretty_form_ascii(::Val{:l2_divh}) = "l2_divh"
Trixi.pretty_form_utf(::Val{:linf_divh}) = "L∞ ∇⋅H"
Trixi.pretty_form_ascii(::Val{:linf_divh}) = "linf_divh"

# The passive material components are set per element, not by the initial condition,
# so their error against it carries no information: report zero for them.
function Trixi.calc_error_norms(func, u, t, analyzer, mesh::DGMultiMesh,
                                equations::MaxwellEquations3D{Heterogeneous},
                                initial_condition, dg::DGMulti, cache, cache_analysis)
    l2_error, linf_error = invoke(Trixi.calc_error_norms,
                                  Tuple{Any, Any, Any, Any, DGMultiMesh{3}, Any, Any,
                                        DGMulti{3}, Any, Any},
                                  func, u, t, analyzer, mesh, equations, initial_condition,
                                  dg, cache, cache_analysis)
    mask = SVector(ntuple(i -> i <= 6 ? one(eltype(l2_error)) : zero(eltype(l2_error)),
                          Val(length(l2_error))))
    return l2_error .* mask, linf_error .* mask
end
