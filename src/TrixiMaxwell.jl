module TrixiMaxwell

using StaticArrays: SVector
import Trixi

include("equations/maxwell_3d.jl")

export MaxwellEquations3D

end
