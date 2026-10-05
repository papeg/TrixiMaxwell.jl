module TrixiMaxwellGmshExt

using TrixiMaxwell: TrixiMaxwell, ImportedMesh
using Gmsh: gmsh

function TrixiMaxwell._read_gmsh(path::AbstractString; verbose = false)
    isfile(path) || throw(ArgumentError("file $path does not exist"))
    initialized_here = !Bool(gmsh.isInitialized())
    initialized_here && gmsh.initialize()
    terminal = gmsh.option.getNumber("General.Terminal")
    gmsh.option.setNumber("General.Terminal", verbose ? 1 : 0)
    gmsh.open(path)
    try
        return ImportedMesh(gmsh_mesh_data(path))
    finally
        gmsh.model.remove()
        gmsh.option.setNumber("General.Terminal", terminal)
        initialized_here && gmsh.finalize()
    end
end

# Extract the current Gmsh model into plain arrays and dictionaries: corner
# vertices of all tetrahedra, physical tag per element, and sorted vertex triples
# of all triangles in physical surface groups.
function gmsh_mesh_data(path)
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    node_index = Dict{Int, Int}(Int(tag) => i for (i, tag) in enumerate(node_tags))
    VX = coordinates[1:3:end]
    VY = coordinates[2:3:end]
    VZ = coordinates[3:3:end]

    EToV = Vector{NTuple{4, Int}}()
    element_groups = Int[]
    for (_, entity) in gmsh.model.getEntities(3)
        physical = gmsh.model.getPhysicalGroupsForEntity(3, entity)
        length(physical) <= 1 ||
            throw(ArgumentError("volume $entity of $path belongs to $(length(physical)) physical groups, expected at most one"))
        group = isempty(physical) ? 0 : Int(first(physical))
        types, _, nodes = gmsh.model.mesh.getElements(3, entity)
        for (element_type, element_nodes) in zip(types, nodes)
            name, _, _, nodes_per_element, _, _ = gmsh.model.mesh.getElementProperties(element_type)
            startswith(name, "Tetrahedron") ||
                throw(ArgumentError("$path contains $name elements, only tetrahedra are supported"))
            for offset in 0:nodes_per_element:(length(element_nodes) - 1)
                push!(EToV, ntuple(i -> node_index[Int(element_nodes[offset + i])], 4))
                push!(element_groups, group)
            end
        end
    end
    isempty(EToV) && throw(ArgumentError("$path contains no tetrahedra"))

    face_sets = Dict{Int, Vector{NTuple{3, Int}}}()
    for (_, entity) in gmsh.model.getEntities(2)
        physical = gmsh.model.getPhysicalGroupsForEntity(2, entity)
        isempty(physical) && continue
        types, _, nodes = gmsh.model.mesh.getElements(2, entity)
        for (element_type, element_nodes) in zip(types, nodes)
            name, _, _, nodes_per_element, _, _ = gmsh.model.mesh.getElementProperties(element_type)
            startswith(name, "Triangle") ||
                throw(ArgumentError("$path contains $name surface elements, only triangles are supported"))
            for offset in 0:nodes_per_element:(length(element_nodes) - 1)
                triple = ntuple(i -> node_index[Int(element_nodes[offset + i])], 3)
                for tag in physical
                    push!(get!(face_sets, Int(tag), NTuple{3, Int}[]), triple)
                end
            end
        end
    end

    group_names = Dict{Int, String}()
    face_set_names = Dict{Int, String}()
    for (dim, tag) in gmsh.model.getPhysicalGroups()
        name = gmsh.model.getPhysicalName(dim, tag)
        isempty(name) && continue
        if dim == 3 && Int(tag) in element_groups
            group_names[Int(tag)] = name
        elseif dim == 2 && haskey(face_sets, Int(tag))
            face_set_names[Int(tag)] = name
        end
    end

    connectivity = permutedims(reduce(hcat, collect.(EToV)))
    return (; vertex_coordinates = (VX, VY, VZ), EToV = connectivity, element_groups,
            group_names, face_sets, face_set_names)
end

end # module
