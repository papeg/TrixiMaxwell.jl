module TrixiMaxwell

using StaticArrays: SVector
using LinearAlgebra: norm, dot, cross

import Trixi

using Trixi: DGMulti, DGMultiMesh, entropy_timederivative, energy_total
using Trixi: StartUpDG

include("equations/maxwell_3d.jl")
include("callbacks_step/analysis_dgmulti.jl")
include("meshes/imported_mesh.jl")
include("meshes/gambit.jl")
include("meshes/gmsh.jl")
include("meshes/download.jl")

export MaxwellEquations3D, Homogeneous, FluxUpwindPenalty, flux_upwind, permittivity,
       permeability, impedance, speed_of_light,
       boundary_condition_perfect_electric_conductor,
       boundary_condition_perfect_magnetic_conductor,
       boundary_condition_silver_mueller, BoundaryConditionIncidentField,
       initial_condition_cavity,
       ImportedMesh, read_gambit, read_gmsh, download_mesh

end
