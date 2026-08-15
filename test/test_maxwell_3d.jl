@testset "MaxwellEquations3D" begin
    equations = MaxwellEquations3D()

    @test equations.speed_of_light == 299_792_458.0
    @test equations isa Trixi.AbstractMaxwellEquations{3, 6}
    @test ndims(equations) == 3
    @test Trixi.nvariables(equations) == 6

    equations_normalized = MaxwellEquations3D(1.0)
    @test equations_normalized.speed_of_light == 1.0

    equations32 = similar(equations, Float32)
    @test equations32.speed_of_light isa Float32

    expected_names = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")
    @test Trixi.varnames(Trixi.cons2cons, equations) == expected_names
    @test Trixi.varnames(Trixi.cons2prim, equations) == expected_names
end
