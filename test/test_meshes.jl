module TestMeshes

using Test
using Trixi
using TrixiMaxwell
using Gmsh

include("test_trixi.jl")

solver = DGMulti(polydeg = 2, element_type = Tet(), approximation_type = Polynomial(),
                 surface_integral = SurfaceIntegralWeakForm(flux_upwind),
                 volume_integral = VolumeIntegralWeakForm())

set_sizes(imported) = Dict(key => length(faces) for (key, faces) in imported.face_sets)
num_boundary_faces(mesh) = count(i -> mesh.md.FToF[i] == i, eachindex(mesh.md.FToF))

@testset "Mesh import" begin
#! format: noindent

@timed_testset "download_mesh" begin
    @test_throws ArgumentError download_mesh("does_not_exist.msh")
    path = download_mesh("F072.neu")
    @test isfile(path)
    @test download_mesh("F072.neu") == path
end

@timed_testset "read_gambit cube meshes" begin
    for (name, vertices, elements, set_name, set_size) in (("F072.neu", 32, 72, :Diri,
                                                            54),
                                                           ("cubeK268.neu", 94, 268,
                                                            :Wall, 156),
                                                           ("sphere.neu", 15, 24,
                                                            :sphere, 24))
        imported = read_gambit(download_mesh(name))
        @test TrixiMaxwell.num_vertices(imported) == vertices
        @test TrixiMaxwell.num_elements(imported) == elements
        @test all(==(1), imported.element_groups)
        @test set_sizes(imported) == Dict(1 => set_size)
        @test TrixiMaxwell.face_set_key(1, imported.face_set_names) == set_name
        mesh = DGMultiMesh(solver, imported)
        @test keys(mesh.boundary_faces) == (set_name,)
        @test length(mesh.boundary_faces[set_name]) == set_size
        @test num_boundary_faces(mesh) == set_size
    end
end

@timed_testset "read_gambit two boundary sets" begin
    imported = read_gambit(download_mesh("FS_K01022.neu"))
    @test TrixiMaxwell.num_vertices(imported) == 309
    @test TrixiMaxwell.num_elements(imported) == 1022
    @test set_sizes(imported) == Dict(1 => 70, 2 => 402)
    @test imported.face_set_names == Dict(1 => "Wall", 2 => "Far")
    mesh = DGMultiMesh(solver, imported)
    @test keys(mesh.boundary_faces) == (:Far, :Wall)
    @test length(mesh.boundary_faces.Wall) == 70
    @test length(mesh.boundary_faces.Far) == 402
end

@timed_testset "read_gambit without boundary sets" begin
    imported = @test_logs (:warn, r"declares 1 boundary sets but contains 0") read_gambit(download_mesh("cubeK5.neu"))
    @test TrixiMaxwell.num_elements(imported) == 5
    @test isempty(imported.face_sets)
    @test_throws ArgumentError DGMultiMesh(solver, imported)
    mesh = DGMultiMesh(solver, imported, allow_untagged_boundary = true)
    @test isempty(mesh.boundary_faces)

    imported = read_gambit(download_mesh("cubeK86.neu"))
    @test TrixiMaxwell.num_elements(imported) == 86
    mesh = DGMultiMesh(solver, imported,
                       is_on_boundary = (; walls = x -> true))
    @test length(mesh.boundary_faces.walls) == num_boundary_faces(mesh)
end

@timed_testset "read_gmsh tagged cube" begin
    imported = read_gmsh(download_mesh("3D_PEC.msh"))
    @test TrixiMaxwell.num_vertices(imported) == 401
    @test TrixiMaxwell.num_elements(imported) == 1227
    @test imported.group_names == Dict(1 => "Vacuum")
    @test set_sizes(imported) ==
          Dict(1 => 32, 2 => 32, 3 => 160, 4 => 160, 5 => 160, 6 => 160)
    mesh = DGMultiMesh(solver, imported)
    @test keys(mesh.boundary_faces) == (:tag_1, :tag_2, :tag_3, :tag_4, :tag_5, :tag_6)
    @test sum(length, mesh.boundary_faces) == num_boundary_faces(mesh)
    # every tagged face lies on the plane its tag describes
    md = mesh.md
    xf, yf, zf = md.xyzf
    for key in keys(mesh.boundary_faces)
        faces = mesh.boundary_faces[key]
        num_pts = size(xf, 1) ÷ 4
        coords = [vec(reshape(c, num_pts, :)[:, faces]) for c in (xf, yf, zf)]
        @test any(c -> all(≈(0, atol = 1e-12), c) || all(≈(1, atol = 1e-12), c), coords)
    end
end

@timed_testset "read_gmsh interior face sets and groups" begin
    imported = read_gmsh(download_mesh("3D_TFSF.msh"))
    @test TrixiMaxwell.num_elements(imported) == 248
    @test sort(unique(imported.element_groups)) == [1, 2]
    @test count(==(1), imported.element_groups) == 100
    @test length(imported.face_sets) == 11
    mesh = DGMultiMesh(solver, imported)
    # set 2 is the interface between the two volumes and is skipped
    @test :tag_2 ∉ keys(mesh.boundary_faces)
    @test length(mesh.boundary_faces) == 10
    @test sum(length, mesh.boundary_faces) == num_boundary_faces(mesh)
end

@timed_testset "read_gmsh sphere in box" begin
    imported = read_gmsh(download_mesh("3D_RCS_SGBC_Sphere_Box_G1.msh"))
    @test TrixiMaxwell.num_elements(imported) == 25084
    @test sort(unique(imported.element_groups)) == [1, 2, 3, 4]
    @test count(==(4), imported.element_groups) == 50
    @test count(==(3), imported.element_groups) == 644
    @test set_sizes(imported)[2] == 520
    mesh = DGMultiMesh(solver, imported)
    # only the outer sphere is a boundary; the file names it "tfsf"
    @test keys(mesh.boundary_faces) == (:tfsf,)
    @test num_boundary_faces(mesh) == 520
end

@timed_testset "read_gmsh geometry file" begin
    path = download_mesh("3D_RCS_SGBC_Sphere_Box.geo")
    coarse = read_gmsh(path; size_factor = 2.0)
    fine = read_gmsh(path; size_factor = 1.5)
    @test TrixiMaxwell.num_elements(fine) > 1.5 * TrixiMaxwell.num_elements(coarse)
    @test coarse.group_names == Dict(1 => "vacuum")
    @test coarse.face_set_names == Dict(2 => "SMA", 3 => "TFSF", 4 => "SGBC")
    VX, VY, VZ = coarse.vertex_coordinates
    radius(i) = sqrt(VX[i]^2 + VY[i]^2 + VZ[i]^2)
    for (tag, r) in ((2, 2.5), (4, 0.5))
        @test all(abs(radius(i) - r) < 1e-8 for face in coarse.face_sets[tag]
                  for i in face)
    end
    mesh = DGMultiMesh(solver, coarse)
    @test keys(mesh.boundary_faces) == (:SMA,)
    @test length(mesh.boundary_faces.SMA) == length(coarse.face_sets[2])
    @test length(mesh.boundary_faces.SMA) == num_boundary_faces(mesh)
    @test_throws ArgumentError read_gmsh(download_mesh("3D_PEC.msh"); size_factor = 2.0)
end

@timed_testset "read_gmsh quadratic tetrahedra" begin
    imported = read_gmsh(download_mesh("3D_Resonant_Sphere.msh"))
    @test TrixiMaxwell.num_elements(imported) == 271
    @test size(imported.EToV, 2) == 4
    @test set_sizes(imported) == Dict(1 => 172)
    mesh = DGMultiMesh(solver, imported)
    @test num_boundary_faces(mesh) == 172
end
end

end # module
