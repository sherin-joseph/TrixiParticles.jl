# ==========================================================================================
# 2D Periodic Couette Flow (moving top wall, fixed bottom wall)
#
# Domain:   x ∈ [0, Lx] (periodic), y ∈ [0, H] (solid walls)
# Walls:    bottom wall u = 0, top wall u = Uwall
# Goal:     benchmark Couette flow (laminar -> transitional) with controlled Re
#
# Notes:
# - "PeriodicBox" in neighbor search is periodic in all directions; we therefore choose a
#   *large* y-period so that no particle ever interacts with its y-image (same trick as your code).
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq
using Random
using StaticArrays: SVector

# ==========================================================================================
# ==== User parameters (benchmark knobs)
Lx    = 1.0            # domain length (periodic direction)
H     = 0.5            # channel height
dp    = 0.02           # particle spacing

Uwall = 1.0            # top wall speed (bottom is 0)
Re    = 500.0          # Reynolds number based on (Uwall, H): Re = Uwall*H/nu
nu    = Uwall * H / Re # kinematic viscosity (m^2/s)

# runtime: for laminar transient to settle ~ O(H^2/nu)
t_end = 5.0 * H^2 / nu
tspan = (0.0, t_end)

# reproducibility + perturbations to trigger transition (increase for "turbulence-like" 2D dynamics)
Random.seed!(42)
perturbation_strength = 0.02 # as fraction of Uwall

# boundaries
boundary_layers = 3
spacing_ratio   = 1

# WCSPH equation of state: choose sound speed from Uwall (keep Mach low)
fluid_density = 1000.0
sound_speed   = 20.0 * Uwall
state_equation = StateEquationCole(; sound_speed,
                                   reference_density=fluid_density,
                                   exponent=7)

# ==========================================================================================
# ==== Geometry: rectangular "tank" but no left/right walls (periodic x), only top/bottom walls
tank_size         = (Lx, H)
initial_fluid_size = tank_size

tank = RectangularTank(dp, initial_fluid_size, tank_size, fluid_density;
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       faces=(false, false, true, true),     # only bottom/top walls
                       velocity=(0.0, 0.0),                  # start everything at rest; we set profiles below
                       coordinates_eltype=Float64)

# ==========================================================================================
# ==== Small helpers to be robust against possible storage layouts (Vector-of-SVector vs matrix)
@inline ycoord(coords::AbstractMatrix, i) = coords[2, i]
@inline ycoord(coords::AbstractVector, i) = coords[i][2]

@inline function set_velocity!(vel::AbstractVector, i, v::SVector{2,Float64})
    vel[i] = v
end
@inline function set_velocity!(vel::AbstractMatrix, i, v::SVector{2,Float64})
    vel[1, i] = v[1]
    vel[2, i] = v[2]
end

# ==========================================================================================
# ==== Initialize fluid velocity: Couette profile + small perturbations
function initialize_couette!(fluid, Uwall, H; eps=0.0)
    coords = fluid.coordinates
    vel    = fluid.velocity
    N      = vel isa AbstractMatrix ? size(vel, 2) : length(vel)

    for i in 1:N
        y = ycoord(coords, i)
        u_mean = Uwall * (y / H)                 # ideal Couette profile
        du = eps * Uwall * (2rand() - 1)
        dv = eps * Uwall * (2rand() - 1)
        set_velocity!(vel, i, SVector(u_mean + du, dv))
    end
    return nothing
end

initialize_couette!(tank.fluid, Uwall, H; eps=perturbation_strength)

# ==========================================================================================
# ==== Identify top vs bottom wall particles (simple split by mid-height)
boundary_coords = tank.boundary.coordinates
Nwall = tank.boundary.velocity isa AbstractMatrix ? size(tank.boundary.velocity, 2) : length(tank.boundary.velocity)

top_wall_ids = Int[]
bottom_wall_ids = Int[]

for i in 1:Nwall
    if ycoord(boundary_coords, i) > 0.5H
        push!(top_wall_ids, i)
    else
        push!(bottom_wall_ids, i)
    end
end

# Set initial wall velocities
for i in bottom_wall_ids
    set_velocity!(tank.boundary.velocity, i, SVector(0.0, 0.0))
end
for i in top_wall_ids
    set_velocity!(tank.boundary.velocity, i, SVector(Uwall, 0.0))
end

# Optional: smooth ramp-up of wall speed (reduces initial acoustic/transient shock)
t_ramp = 0.25 * (H / Uwall)  # ramp time scale
wall_speed(t) = Uwall * (1 - exp(-t / t_ramp))

function enforce_wall_velocities!(t)
    Uw = wall_speed(t)
    for i in bottom_wall_ids
        set_velocity!(tank.boundary.velocity, i, SVector(0.0, 0.0))
    end
    for i in top_wall_ids
        set_velocity!(tank.boundary.velocity, i, SVector(Uw, 0.0))
    end
    return nothing
end

# ==========================================================================================
# ==== Fluid system (WCSPH)
smoothing_length = 1.2 * dp
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

fluid_density_calculator = ContinuityDensity()

# Use physical viscosity model (good for Re studies) 
viscosity = ViscosityAdami(nu=nu)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid,
                                           fluid_density_calculator,
                                           state_equation,
                                           smoothing_kernel,
                                           smoothing_length;
                                           viscosity=viscosity,
                                           shifting_technique=nothing,
                                           pressure_acceleration=nothing)

# ==========================================================================================
# ==== Boundary system (dummy particles) + no-slip via viscosity coupling
boundary_density_calculator = AdamiPressureExtrapolation()
viscosity_wall = ViscosityAdami(nu=nu)  # consistent with fluid viscosity

boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                             tank.boundary.mass,
                                             boundary_density_calculator,
                                             smoothing_kernel,
                                             smoothing_length;
                                             viscosity=viscosity_wall,
                                             state_equation=state_equation)

boundary_system = WallBoundarySystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation + periodic neighbor search
# Make y-period large so y-images are far outside kernel support (avoid unintended y-periodicity)
periodic_box = PeriodicBox(min_corner=[0.0, -2H], max_corner=[Lx, 3H])
neighborhood_search = GridNeighborhoodSearch{2}(; periodic_box)

semi = Semidiscretization(fluid_system, boundary_system;
                          neighborhood_search,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

# ==========================================================================================
# ==== Callbacks
info_callback   = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="couette_")

# Enforce prescribed wall velocities every step (safe even if boundary velocities get touched elsewhere)
wall_velocity_callback = DiscreteCallback(
    (u, t, integrator) -> true,
    integrator -> enforce_wall_velocities!(integrator.t)
)

callbacks = CallbackSet(info_callback, saving_callback, wall_velocity_callback)

# ==========================================================================================
# ==== Time integration
# A conservative dtmax for WCSPH acoustics; you can tighten this for higher Re / smaller dp.
dtmax = 0.25 * smoothing_length / sound_speed

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-8,
            reltol=1e-4,
            dtmax=dtmax,
            save_everystep=false,
            callback=callbacks)

# After the run:
# - Compare mean u(y) against analytic Couette: u(y)=Uwall*y/H (laminar check)
# - Increase Re, resolution, and perturbation_strength to study transition behavior
