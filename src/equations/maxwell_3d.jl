@doc raw"""
    vacuum Maxwell
"""
struct MaxwellEquations3D{RealT <: Real} <: Trixi.AbstractMaxwellEquations{3, 6}
    speed_of_light::RealT
    function MaxwellEquations3D(c::Real = 299_792_458.0)
        return new{typeof(c)}(c)
    end
end


function Base.similar(equations::MaxwellEquations3D, ::Type{NewRealT}) where {NewRealT}
    return MaxwellEquations3D(convert(NewRealT, equations.speed_of_light))
end

function Trixi.varnames(::typeof(Trixi.cons2cons), ::MaxwellEquations3D)
    return ("Ex", "Ey", "Ez", "Bx", "By", "Bz")
end

function Trixi.varnames(::typeof(Trixi.cons2prim), ::MaxwellEquations3D)
    return ("Ex", "Ey", "Ez", "Bx", "By", "Bz")
end

@inline Trixi.cons2prim(u, ::MaxwellEquations3D) = u
@inline Trixi.cons2entropy(u, ::MaxwellEquations3D) = u
