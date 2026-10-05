# Face vertices of a tetrahedron in the Gambit neutral file format, face 1 to 4.
const GAMBIT_TET_FACES = ((1, 2, 3), (1, 2, 4), (2, 3, 4), (1, 3, 4))

function gambit_section(lines, name, start = 1)
    index = findnext(line -> occursin(name, line), lines, start)
    index === nothing && throw(ArgumentError("section $name not found"))
    return index
end

"""
    read_gambit(path)

Read a linear tetrahedral mesh from a Gambit neutral file (`.neu`) into an
[`ImportedMesh`](@ref). Element groups become element tags with the group
name, boundary condition sets become face sets numbered in file order with the
set name.
"""
read_gambit(path::AbstractString) = ImportedMesh(gambit_mesh_data(path))

# Parse a Gambit neutral file into plain arrays and dictionaries. Boundary faces
# are returned as sorted vertex triples, independent of any face numbering.
function gambit_mesh_data(path::AbstractString)
    isfile(path) || throw(ArgumentError("file $path does not exist"))
    lines = readlines(path)

    header = gambit_section(lines, "NUMNP")
    counts = parse.(Int, split(lines[header + 1]))
    num_vertices, num_elements, num_groups, num_sets, num_dimensions = counts[1:5]
    num_dimensions == 3 ||
        throw(ArgumentError("$path is a $(num_dimensions)D mesh, only 3D meshes are supported"))

    VX, VY, VZ = ntuple(_ -> zeros(num_vertices), 3)
    start = gambit_section(lines, "NODAL COORDINATES")
    for k in 1:num_vertices
        parts = split(lines[start + k])
        parse(Int, parts[1]) == k ||
            throw(ArgumentError("vertex $k of $path is numbered $(parts[1]), expected consecutive numbering"))
        VX[k] = parse(Float64, parts[2])
        VY[k] = parse(Float64, parts[3])
        VZ[k] = parse(Float64, parts[4])
    end

    EToV = zeros(Int, num_elements, 4)
    start = gambit_section(lines, "ELEMENTS/CELLS")
    for k in 1:num_elements
        parts = split(lines[start + k])
        id, element_type, nodes_per_element = parse.(Int, parts[1:3])
        id == k ||
            throw(ArgumentError("element $k of $path is numbered $id, expected consecutive numbering"))
        if element_type != 6 || nodes_per_element != 4
            throw(ArgumentError("element $k of $path is not a linear tetrahedron (type $element_type with $nodes_per_element nodes)"))
        end
        EToV[k, :] = parse.(Int, parts[4:7])
    end

    element_groups = zeros(Int, num_elements)
    group_names = Dict{Int, String}()
    start = 1
    for _ in 1:num_groups
        start = gambit_section(lines, "GROUP:", start)
        parts = split(lines[start])
        group = parse(Int, parts[2])
        group_size = parse(Int, parts[4])
        group_names[group] = strip(lines[start + 1])
        ids = Int[]
        line = start + 3
        while !occursin("ENDOFSECTION", lines[line])
            append!(ids, parse.(Int, split(lines[line])))
            line += 1
        end
        length(ids) == group_size ||
            throw(ArgumentError("group $group of $path lists $(length(ids)) elements, header says $group_size"))
        element_groups[ids] .= group
        start = line
    end
    if any(iszero, element_groups)
        throw(ArgumentError("$(count(iszero, element_groups)) elements of $path belong to no element group"))
    end

    face_sets = Dict{Int, Vector{NTuple{3, Int}}}()
    face_set_names = Dict{Int, String}()
    start = 1
    tag = 0
    while (index = findnext(line -> occursin("BOUNDARY CONDITIONS", line), lines, start)) !==
          nothing
        parts = split(lines[index + 1])
        name = String(parts[1])
        set_type = parse(Int, parts[2])
        set_size = parse(Int, parts[3])
        set_type == 1 ||
            throw(ArgumentError("boundary set $name of $path is node based, only element based sets are supported"))
        faces = Vector{NTuple{3, Int}}(undef, set_size)
        for i in 1:set_size
            element, element_type, face = parse.(Int, split(lines[index + 1 + i])[1:3])
            element_type == 6 ||
                throw(ArgumentError("boundary set $name of $path references element type $element_type, expected 6"))
            vertices = GAMBIT_TET_FACES[face]
            faces[i] = Tuple(sort([EToV[element, v] for v in vertices]))
        end
        tag += 1
        face_sets[tag] = faces
        face_set_names[tag] = name
        start = index + 1
    end
    tag == num_sets ||
        @warn "$path declares $num_sets boundary sets but contains $tag"

    return (; vertex_coordinates = (VX, VY, VZ), EToV, element_groups, group_names,
            face_sets, face_set_names)
end
