# ==========================================================================================
# 2D Poiseuille Flow Simulation (Weakly Compressible SPH)
# Comparison: Newtonian vs Carreau-Yasuda (shear-thinning) viscosity
#
# Based on:
#   Zhan, X., et al. "Dynamical pressure boundary condition for weakly compressible smoothed particle hydrodynamics"
#   Physics of Fluids, Volume 37
#   https://doi.org/10.1063/5.0254575
#
# This example sets up a 2D Poiseuille flow simulation in a rectangular channel
# including open boundary conditions.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
const wall_distance = 0.001 # distance between top and bottom wall
const flow_length = 0.01 # distance between inflow and outflow (increased for flow development)

particle_spacing = wall_distance / 30  # 50 particles across channel height

# NOTE: For proper validation of Poiseuille flow:
#   - Resolution: >= 50 particles across height (wall_distance / particle_spacing)
#   - Length: >= 5× height to minimize entrance effects (flow_length / wall_distance >= 5)
#   - Time: Run until steady state (monitor velocity convergence)
#   - Expected: Parabolic velocity profile with u_max = H²·ΔP / (8·μ·L)

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 4

# Make sure that the kernel support of fluid particles at an open boundary is always
# fully sampled.
open_boundary_layers = 10

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 15.0)  # Extended time to reach steady state
wcsph = true

domain_size = (flow_length, wall_distance)

open_boundary_size = (open_boundary_layers * particle_spacing, domain_size[2])

fluid_density = 1000.0
reynolds_number = 50
const pressure_drop = 0.1
pressure_out = 0.1
pressure_in = pressure_out + pressure_drop
const dynamic_viscosity = sqrt(fluid_density * wall_distance^3 * pressure_drop /
                               (8 * flow_length * reynolds_number))

v_max = wall_distance^2 * pressure_drop / (8 * dynamic_viscosity * flow_length)

sound_speed_factor = 100
sound_speed = sound_speed_factor * v_max

flow_direction = (1.0, 0.0)

# Carreau-Yasuda parameters (kinematic viscosities)
# For shear-thinning: nu_inf < nu0 and n < 1
kinematic_viscosity = dynamic_viscosity / fluid_density
ν0 = kinematic_viscosity        # Zero-shear-rate viscosity
ν∞ = 0.1 * kinematic_viscosity  # Infinite-shear-rate viscosity (90% reduction at high shear)
λ = 0.5                         # Time constant (controls transition region)
a_cy = 2.0                      # Yasuda exponent
n_cy = 0.4                      # Power-law index (< 1 for shear-thinning, lower = more thinning)
ϵ_cy = 0.1 * particle_spacing   # Regularization parameter

# ==========================================================================================
# ==== Build Simulation Function
function build_simulation(viscosity_model, output_prefix::String)
    # Geometry
    pipe = RectangularTank(particle_spacing, domain_size, domain_size, fluid_density,
                           pressure=(pos) -> pressure_out +
                                             pressure_drop * (1 - (pos[1] / flow_length)),
                           n_layers=boundary_layers, faces=(false, false, true, true),
                           coordinates_eltype=Float64)

    inlet = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                            fluid_density, n_layers=boundary_layers, pressure=pressure_in,
                            min_coordinates=(0.0, 0.0), faces=(false, false, true, true),
                            coordinates_eltype=Float64)

    outlet = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                             fluid_density, n_layers=boundary_layers,
                             min_coordinates=(pipe.fluid_size[1] - open_boundary_size[1], 0.0),
                             faces=(false, false, true, true),
                             coordinates_eltype=Float64)

    fluid = setdiff(pipe.fluid, inlet.fluid, outlet.fluid)
    n_buffer_particles = 10 * pipe.n_particles_per_dimension[2]^2

    # Fluid system
    smoothing_length = 2 * particle_spacing
    smoothing_kernel = WendlandC2Kernel{2}()
    fluid_density_calculator = ContinuityDensity()
    background_pressure = 7 * sound_speed_factor / 10 * fluid_density * v_max^2
    shifting_technique = TransportVelocityAdami(; background_pressure)

    if wcsph
        state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                           exponent=1)
        fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                                   state_equation, smoothing_kernel,
                                                   buffer_size=n_buffer_particles,
                                                   shifting_technique=shifting_technique,
                                                   density_diffusion=DensityDiffusionMolteniColagrossi(delta=0.1),
                                                   smoothing_length, viscosity=viscosity_model)
    else
        state_equation = nothing
        fluid_system = EntropicallyDampedSPHSystem(fluid, smoothing_kernel,
                                                   smoothing_length,
                                                   sound_speed, viscosity=viscosity_model,
                                                   density_calculator=fluid_density_calculator,
                                                   shifting_technique=shifting_technique,
                                                   buffer_size=n_buffer_particles)
    end

    # Open boundaries
    open_boundary_model = BoundaryModelDynamicalPressureZhang()
    boundary_type_in = BidirectionalFlow()
    face_in = ([open_boundary_size[1], 0.0], [open_boundary_size[1], pipe.fluid_size[2]])
    inflow = BoundaryZone(; boundary_face=face_in, face_normal=flow_direction,
                          open_boundary_layers, density=fluid_density, particle_spacing,
                          reference_velocity=nothing, reference_pressure=0.2,
                          initial_condition=inlet.fluid, boundary_type=boundary_type_in)

    boundary_type_out = BidirectionalFlow()
    face_out = ([pipe.fluid_size[1] - open_boundary_size[1], 0.0],
                [pipe.fluid_size[1] - open_boundary_size[1], pipe.fluid_size[2]])
    outflow = BoundaryZone(; boundary_face=face_out, face_normal=(.-(flow_direction)),
                           open_boundary_layers, density=fluid_density, particle_spacing,
                           reference_velocity=nothing, reference_pressure=0.1,
                           initial_condition=outlet.fluid, boundary_type=boundary_type_out)

    open_boundary = OpenBoundarySystem(inflow, outflow; fluid_system,
                                       boundary_model=open_boundary_model,
                                       calculate_flow_rate=true,
                                       buffer_size=n_buffer_particles)

    # Wall boundaries
    wall = union(pipe.boundary)
    boundary_model = BoundaryModelDummyParticles(wall.density, wall.mass,
                                                 AdamiPressureExtrapolation(),
                                                 state_equation=state_equation,
                                                 viscosity=viscosity_model,
                                                 smoothing_kernel, smoothing_length)
    boundary_system = WallBoundarySystem(wall, boundary_model)

    # Neighborhood search
    min_corner = minimum(wall.coordinates .- 2 * particle_spacing, dims=2)
    max_corner = maximum(wall.coordinates .+ 2 * particle_spacing, dims=2)
    nhs = GridNeighborhoodSearch{2}(; cell_list=FullGridCellList(; min_corner, max_corner),
                                    update_strategy=ParallelUpdate())

    # Semidiscretization
    semi = Semidiscretization(fluid_system, open_boundary,
                              boundary_system, neighborhood_search=nhs,
                              parallelization_backend=PolyesterBackend())

    ode = semidiscretize(semi, tspan)

    # Callbacks
    info_callback = InfoCallback(interval=50)
    saving_callback = SolutionSavingCallback(dt=0.05, prefix=output_prefix, output_directory="out")
    callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback())

    return ode, callbacks
end

# ==========================================================================================


# ==========================================================================================
# ==== Case 2: Carreau-Yasuda (shear-thinning)
println("\n" * "="^80)
println("Running CARREAU-YASUDA simulation...")
println("="^80)

visc_cy = ViscosityCarreauYasuda(nu0=ν0, nu_inf=ν∞, lambda=λ, a=a_cy, n=n_cy, epsilon=ϵ_cy)
ode_cy, cb_cy = build_simulation(visc_cy, "poiseuille_carreau_yasuda")

sol_cy = solve(ode_cy, RDPK3SpFSAL35(),
               abstol=1e-6, reltol=1e-4, dtmax=1e-2,
               save_everystep=false, callback=cb_cy)

println("Carreau-Yasuda simulation complete!")

# ==========================================================================================
println("\n" * "="^80)
println("COMPARISON GUIDE - What to observe:")
println("="^80)
println("1. VELOCITY PROFILE (most important!):")
println("   - Newtonian: Classic parabolic profile (max velocity at center)")
println("   - Carreau-Yasuda: FLATTER in center, STEEPER near walls")
println("   - Higher shear rates near walls → lower viscosity → sharper gradients")
println("")
println("2. FLOW RATE:")
println("   - C-Y should have HIGHER flow rate (less resistance at high shear)")
println("   - Check open boundary flow rate output")
println("")
println("3. DEVELOPMENT LENGTH:")
println("   - Distance for flow to fully develop may differ")
println("   - C-Y may develop differently due to shear-dependent viscosity")
println("")
println("4. VISUALIZATION:")
println("   - Load both datasets in ParaView")
println("   - Create velocity magnitude contours")
println("   - Compare velocity profiles at x = L/2 (middle of channel)")
println("   - Plot velocity vs y-position to see profile shape")
println("")
println("Output files:")
println("  - out/poiseuille_newtonian_*.vtu")
println("  - out/poiseuille_carreau_yasuda_*.vtu")
println("="^80)
