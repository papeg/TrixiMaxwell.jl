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

@testset "Cartesian physical flux" begin
    equations = MaxwellEquations3D(2.0)
    u = SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    @test Trixi.flux(u, 1, equations) ==
        SVector(0.0, 24.0, -20.0, 0.0, -3.0, 2.0)

    @test Trixi.flux(u, 2, equations) ==
        SVector(-24.0, 0.0, 16.0, 3.0, 0.0, -1.0)

    @test Trixi.flux(u, 3, equations) ==
        SVector(20.0, -16.0, 0.0, -2.0, 1.0, 0.0)
end

@testset "Normal-direction physical flux" begin
    equations = MaxwellEquations3D(2.0)
    u = SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    @test Trixi.flux(u, SVector(1.0, 0.0, 0.0), equations) ==
        Trixi.flux(u, 1, equations)

    @test Trixi.flux(u, SVector(0.0, 1.0, 0.0), equations) ==
        Trixi.flux(u, 2, equations)

    @test Trixi.flux(u, SVector(0.0, 0.0, 1.0), equations) ==
        Trixi.flux(u, 3, equations)

    normal = SVector(2.0, -1.0, 0.5)
    @test Trixi.flux(u, normal, equations) ==
        SVector(34.0, 40.0, -56.0, -4.0, -5.5, 5.0)
end
