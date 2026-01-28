# ==========================================================================================
# 2D Lid-Driven Cavity (WCSPH) — TrixiParticles.jl
#
# Square cavity, no-slip on all walls, moving lid on top.
# Classic benchmark: compare centerline velocity profiles (e.g. Ghia et al. 1982).
#
# Key knobs:
#   Re = U*L/nu  (increase Re for transition/unsteadiness)
#   particle_spacing (resolution)
#   sound_speed_factor (compressibility; typical 10)
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Parameters (nondimensional-ish)
L = 1.0                    # cavity length
U_lid = 1.0                # lid speed
Re = 1000.0                # try: 100, 400, 1000, 5000 (2D becomes unsteady as Re increases)
nu = U_lid * L / Re        # kinematic viscosity

rho0 = 1000.0              # reference density
sound_speed_factor = 10.0  # c ≈ 10*U keeps Mach ~ 0.1 in WCSPH
c0 = sound_speed_factor * U_lid

# Time span: for steady benchmark at moderate Re, run longer (e.g. 20–50)
tspan = (0.0, 20.0)

# ==========================================================================================
# ==== Resolution / Discretization
fluid_particle_spacing = 0.01          # try 0.02 for quick, 0.01 or 0.005 for benchmark-quality
boundary_layers = 3
spacing_ratio = 1

smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# ==========================================================================================
# ==== Equation of State (WCSPH)
state_equation = StateEquationCole(; sound_speed=c0,
                                   reference_density=rho0,
                                   exponent=7)

# ==========================================================================================
# ==== Build cavity (fluid + boundary)
# All four walls present: faces=(true,true,true,true)
tank_size = (L, L)
initial_fluid_size = tank_size

cavity = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, rho0;
                         n_layers=boundary_layers,
                         spacing_ratio=spacing_ratio,
                         faces=(true, true, true, true),
                         velocity=(0.0, 0.0),
                         coordinates_eltype=Float64)

# --- Prescribe lid velocity on the TOP boundary layers (largest y), keep others fixed
# InitialCondition stores coordinates/velocity as arrays with particles in columns. :contentReference[oaicite:2]{index=2}
yb = cavity.boundary.coordinates[2, :]
y_top = maximum(yb)
tol = 0.6 * fluid_particle_spacing

for i in axes(cavity.boundary.coordinates, 2)
    if cavity.boundary.coordinates[2, i] > y_top - tol
        cavity.boundary.velocity[:, i] .= (U_lid, 0.0)
    else
        cavity.boundary.velocity[:, i] .= (0.0, 0.0)
    end
end

# ==========================================================================================
# ==== Fluid system (WCSPH)
fluid_density_calculator = ContinuityDensity()

# Physical viscosity in the fluid
viscosity_fluid = ViscosityAdami(nu=nu)

# Density diffusion to reduce pressure noise (recommended with ContinuityDensity). :contentReference[oaicite:3]{index=3}
density_diffusion = DensityDiffusionMolteniColagrossi(; delta=0.1)

# Particle shifting helps prevent clumping/voids near walls at higher Re
shifting = nothing

fluid_system = WeaklyCompressibleSPHSystem(cavity.fluid,
                                           fluid_density_calculator,
                                           state_equation,
                                           smoothing_kernel,
                                           smoothing_length;
                                           viscosity=viscosity_fluid,
                                           density_diffusion=density_diffusion,
                                           shifting_technique=shifting,
                                           pressure_acceleration=nothing)

# ==========================================================================================
# ==== Boundary system (dummy particles, no-slip)
boundary_density_calculator = AdamiPressureExtrapolation()

# No-slip is imposed by ViscosityAdami + prescribed boundary particle velocity. :contentReference[oaicite:4]{index=4}
viscosity_wall = ViscosityAdami(nu=nu)

boundary_model = BoundaryModelDummyParticles(cavity.boundary.density,
                                             cavity.boundary.mass,
                                             boundary_density_calculator,
                                             smoothing_kernel,
                                             smoothing_length;
                                             state_equation=state_equation,
                                             viscosity=viscosity_wall)

boundary_system = WallBoundarySystem(cavity.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation setup
# No periodicity here
neighborhood_search = GridNeighborhoodSearch{2}()

semi = Semidiscretization(fluid_system, boundary_system;
                          neighborhood_search,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

info_callback   = InfoCallback(interval=200)
saving_callback = SolutionSavingCallback(dt=0.05,
                                         prefix="lid_cavity")

callbacks = CallbackSet(info_callback, saving_callback)

# A conservative dtmax helps with stability near corners/walls in WCSPH
dtmax = 0.25 * fluid_particle_spacing / (c0 + U_lid)

sol = solve(ode,
            RDPK3SpFSAL35(),
            abstol=1e-8,
            reltol=1e-4,
            dtmax=dtmax,
            save_everystep=false,
            callback=callbacks)

# ==========================================================================================
# End
# Postprocess idea: sample u(x=0, y) and v(x, y=0) centerlines and compare to Ghia et al.
