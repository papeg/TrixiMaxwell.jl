module TrixiMaxwell

using StaticArrays: SVector
using LinearAlgebra: norm, dot, cross

import Trixi

using Trixi: DGMulti, DGMultiMesh, entropy_timederivative, energy_total
using Trixi: StartUpDG
using WriteVTK: vtk_grid, vtk_save, MeshCell, VTKCellTypes, VTKCellData

include("equations/maxwell_3d.jl")
include("callbacks_step/analysis_dgmulti.jl")
include("meshes/imported_mesh.jl")
include("meshes/gambit.jl")
include("meshes/gmsh.jl")
include("meshes/download.jl")
include("visualization/vtk.jl")
include("callbacks_step/save_vtk.jl")

export MaxwellEquations3D, Homogeneous, Heterogeneous, FluxUpwindPenalty, flux_upwind,
       permittivity, permeability, conductivity, impedance, admittance, speed_of_light,
       source_terms_conductivity,
       boundary_condition_perfect_electric_conductor,
       boundary_condition_perfect_magnetic_conductor,
       boundary_condition_silver_mueller, BoundaryConditionIncidentField,
       initial_condition_cavity,
       ImportedMesh, read_gambit, read_gmsh, download_mesh,
       write_mesh_vtk, write_solution_vtk, SaveVtkCallback,
       Heterogeneous, conductivity, admittance, source_terms_conductivity

end
