module TrixiMaxwell

using StaticArrays: SVector
using LinearAlgebra: norm, dot, cross

import Trixi

include("equations/maxwell_3d.jl")

export MaxwellEquations3D, Homogeneous, FluxUpwindPenalty, flux_upwind, permittivity,
       permeability, impedance

end
