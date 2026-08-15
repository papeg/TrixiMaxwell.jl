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

@inline function Trixi.flux(u, orientation::Integer, equations::MaxwellEquations3D)
    Ex, Ey, Ez, Bx, By, Bz = u
    c2 = equations.speed_of_light^2
    RealT = eltype(u)

    if orientation == 1
        f1 = zero(RealT)
        f2 = c2 * Bz
        f3 = -c2 * By
        f4 = zero(RealT)
        f5 = -Ez
        f6 = Ey
    elseif orientation == 2
        f1 = -c2 * Bz
        f2 = zero(RealT)
        f3 = c2 * Bx
        f4 = Ez
        f5 = zero(RealT)
        f6 = -Ex
    else
        f1 = c2 * By
        f2 = -c2 * Bx
        f3 = zero(RealT)
        f4 = -Ey
        f5 = Ex
        f6 = zero(RealT)
    end

    return SVector(f1, f2, f3, f4, f5, f6)
end

@inline Trixi.cons2prim(u, ::MaxwellEquations3D) = u
@inline Trixi.cons2entropy(u, ::MaxwellEquations3D) = u
