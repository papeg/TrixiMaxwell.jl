"""
    Material(; epsilon = 1.0, mu = one(epsilon), sigma = zero(epsilon))

Relative permittivity, relative permeability and normalized conductivity of a
linear medium, assigned per element with [`set_materials!`](@ref).
"""
struct Material{RealT <: Real}
    epsilon::RealT
    mu::RealT
    sigma::RealT
end

function Material(; epsilon = 1.0, mu = one(epsilon), sigma = zero(epsilon))
    return Material(promote(float(epsilon), float(mu), float(sigma))...)
end

@inline material_components(material::Material) = SVector(material.epsilon, material.mu,
                                                          material.sigma)

function element_centroid(md, element)
    return SVector(ntuple(d -> sum(view(md.xyz[d], :, element)) / size(md.xyz[d], 1), 3))
end

"""
    set_materials!(u_ode, semi, material_at)
    set_materials!(u_ode, semi, element_groups, materials::AbstractDict)

Write the material into the passive components of every node of a
[`MaxwellEquations3D`](@ref)`{Heterogeneous}` state, element by element. The
first form evaluates `material_at(x)` at the element centroid, the second looks
up `materials[element_groups[element]]`, for example with the groups of an
[`ImportedMesh`](@ref). Both return a [`Material`](@ref) per element. Call after
`semidiscretize` on `ode.u0`.
"""
function set_materials!(u_ode, semi, material_at)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    if !(equations isa MaxwellEquations3D{Heterogeneous})
        throw(ArgumentError("set_materials! needs MaxwellEquations3D(Heterogeneous()), got $(typeof(equations).name.wrapper) with material $(typeof(equations).parameters[1])"))
    end
    md = mesh.md
    return set_materials_by_element!(u_ode, semi,
                                     element -> material_at(element_centroid(md, element)))
end

function set_materials!(u_ode, semi, element_groups::AbstractVector{<:Integer},
                        materials::AbstractDict)
    mesh, = Trixi.mesh_equations_solver_cache(semi)
    num_elements = mesh.md.num_elements
    length(element_groups) == num_elements ||
        throw(ArgumentError("got $(length(element_groups)) element groups for $num_elements elements"))
    for group in unique(element_groups)
        haskey(materials, group) ||
            throw(ArgumentError("no material given for element group $group"))
    end
    return set_materials_by_element!(u_ode, semi,
                                     element -> materials[element_groups[element]])
end

function set_materials_by_element!(u_ode, semi, material_of_element)
    u = Trixi.wrap_array(u_ode, semi)
    for element in axes(u, 2)
        components = material_components(material_of_element(element))
        for node in axes(u, 1)
            u_node = u[node, element]
            u[node, element] = vcat(electric_field(u_node), magnetic_field(u_node),
                                    components)
        end
    end
    return u_ode
end
