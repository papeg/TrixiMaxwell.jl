module TrixiMaxwell

using StaticArrays: SVector
using LinearAlgebra: norm

import Trixi

include("equations/maxwell_3d.jl")

export MaxwellEquations3D

end
