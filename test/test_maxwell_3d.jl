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

@testset "Characteristic wave speeds" begin
    equations = MaxwellEquations3D(2.0)
    u_ll = SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    u_rr = -u_ll

    for orientation in 1:3
        @test Trixi.max_abs_speed_naive(u_ll, u_rr, orientation, equations) == 2.0
    end

    # norm(normal) = 3, so the directional speed is c * 3 = 6
    normal = SVector(2.0, -1.0, 2.0)

    @test Trixi.max_abs_speed_naive(u_ll, u_rr, normal, equations) == 6.0

    # Verify that Trixi's generic fallback reaches our implementation
    @test Trixi.max_abs_speed(u_ll, u_rr, normal, equations) == 6.0

    @test Trixi.have_constant_speed(equations) === Trixi.True()
    @test Trixi.max_abs_speeds(equations) == (2.0, 2.0, 2.0)
end

@testset "Lax-Friedrichs surface flux" begin
    equations = MaxwellEquations3D(2.0)
    surface_flux = Trixi.flux_lax_friedrichs

    u_ll = SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    u_rr = SVector(-2.0, 0.5, 4.0, -1.0, 3.0, -0.5)
    normal = SVector(2.0, -1.0, 2.0)

    # no jump if both are equal
    @test surface_flux(u_ll, u_ll, 1, equations) ==
        Trixi.flux(u_ll, 1, equations)

    @test surface_flux(u_ll, u_ll, normal, equations) ==
        Trixi.flux(u_ll, normal, equations)

    # Lax-Friedrich
    central_flux = 0.5 * (Trixi.flux(u_ll, normal, equations) + Trixi.flux(u_rr, normal, equations))

    lambda_max = equations.speed_of_light * norm(normal)
    dissipation = -0.5 * lambda_max * (u_rr - u_ll)
    expected_flux = central_flux + dissipation

    @test surface_flux(u_ll, u_rr, normal, equations) ≈ expected_flux

    @test surface_flux(u_ll, u_rr, normal, equations) ≈
        -surface_flux(u_rr, u_ll, -normal, equations)
end

@testset "Plane-wave initial condition" begin
    c = 2.0
    wavelength = 1.0
    equations = MaxwellEquations3D(c)
    initial_condition = Trixi.initial_condition_convergence_test

    # At 1/4 wavelength: sin(2πx) = 1
    x_peak = SVector(wavelength / 4, 0.0, 0.0)
    u_peak = initial_condition(x_peak, 0.0, equations)

    @test u_peak ≈ SVector(0.0, -c, 0.0, 0.0, 0.0, 1.0)
    @test u_peak[2] ≈ -c * u_peak[6]

    # wave travels one eigth of wavelength in this time
    x = SVector(wavelength / 8, 1 / 3, 2 / 5)
    t = wavelength / (8 * c)

    travel_distance = c * t
    x_shifted = SVector(x[1] + travel_distance, x[2], x[3])

    @test initial_condition(x, t, equations) ≈
        initial_condition(x_shifted, 0.0, equations)

    equations32 = MaxwellEquations3D(1.0f0)
    x32 = SVector(0.25f0, 0.0f0, 0.0f0)

    @test eltype(initial_condition(x32, 0.0f0, equations32)) == Float32

end
