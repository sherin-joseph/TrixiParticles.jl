# ==========================================================================================
# 2D Double Shear Layer / Kelvin–Helmholtz Roll-up (Fully Periodic)


# Domain: [0, Lx) x [0, Ly), periodic in x and y
# IC: Two shear layers (around y=0.25Ly and y=0.75Ly) + small sinusoidal perturbation
# Wang et al., An efficient truncation scheme for Eulerian and total Lagrangian SPH methods, Physics of Fluids (2024)
# Brown & Minion, Performance of Under-resolved Two-Dimensional Incompressible Flow Simulations (Journal of Computational Physics, 1995)
# Obeidat & Bordas, Three-dimensional remeshed smoothed particle hydrodynamics for the simulation of isotropic turbulence, International Journal for Numerical Methods in Fluids (2017/2018).
# Output: SolutionSavingCallback writes snapshots you can postprocess/visualize.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq
using StaticArrays

# ==========================================================================================
# ==== Resolution / Domain
particle_spacing = 0.01                 # decrease for better turbulence range (cost ↑)
Lx_target, Ly_target = 1.0, 1.0

# Make domain exactly compatible with the spacing to avoid "almost periodic" grids
nx = round(Int, Lx_target / particle_spacing)
ny = round(Int, Ly_target / particle_spacing)
Lx = nx * particle_spacing
Ly = ny * particle_spacing

# ==========================================================================================
# ==== Double shear layer initial condition parameters
U0 = 1.0                                # velocity scale
delta = 0.05 * Ly                       # shear layer thickness
eps_v = 0.01 * U0                       # perturbation amplitude

# Piecewise shear profile (classic "double shear layer")
# y in [0, Ly)
@inline function u_shear(y)
    ymid = 0.5 * Ly
    if y < ymid
        return U0 * tanh((y - 0.25 * Ly) / delta)
    else
        return U0 * tanh((0.75 * Ly - y) / delta)
    end
end

# Small perturbation to trigger KH roll-up
@inline function v_perturb(x, y)
    # optionally localize around the shear layers by multiplying with Gaussians;
    # here we keep it simple (common benchmark choice):
    return eps_v * sin(2π * x / Lx)
end

# Velocity function expected by InitialCondition (coords -> velocity vector)
velocity_ic = (x -> begin
    xx, yy = x[1], x[2]
    SVector(u_shear(yy), v_perturb(xx, yy))
end)

fluid_density = 1000.0

# ==========================================================================================
# ==== Initial condition: rectangular lattice of particles
# NOTE: RectangularShape creates a filled rectangle of particles.
fluid_ic = RectangularShape(particle_spacing, (nx, ny), (0.0, 0.0);
                            density=fluid_density,
                            velocity=velocity_ic,
                            coordinates_perturbation=nothing)

# ==========================================================================================
# ==== Fluid model (WCSPH)
smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# Speed of sound (controls compressibility; higher -> less compressible, smaller dt)
sound_speed = 20 * U0
state_equation = StateEquationCole(; sound_speed,
                                   reference_density=fluid_density,
                                   exponent=7,
                                   background_pressure=0.0)

density_calculator = ContinuityDensity()

# Reynolds number control: nu = U0 * Ly / Re
Re = 5_000.0
nu = U0 * Ly / Re

# --- Viscosity options ---
# (A) Plain physical viscosity:
# viscosity = ViscosityAdami(; nu)

# (B) LES-style SGS viscosity (Smagorinsky-type) on top of base nu:
viscosity = ViscosityAdamiSGS(; nu, C_S=0.12)

# --- Density diffusion (recommended for WCSPH) ---
density_diffusion = DensityDiffusionMolteniColagrossi(; delta=0.1)

# --- Optional stabilization (useful for long runs / negative pressures) ---
shifting_technique = ConsistentShiftingSun2019()  # set to nothing to disable
pressure_acceleration = tensile_instability_control # set to nothing to disable

fluid_system = WeaklyCompressibleSPHSystem(fluid_ic,
                                          density_calculator,
                                          state_equation,
                                          smoothing_kernel,
                                          smoothing_length;
                                          viscosity=viscosity,
                                          density_diffusion=density_diffusion,
                                          shifting_technique=shifting_technique,
                                          pressure_acceleration=pressure_acceleration)

# ==========================================================================================
# ==== Periodic domain neighborhood search
periodic_box = PeriodicBox(min_corner=[0.0, 0.0], max_corner=[Lx, Ly])
neighborhood_search = GridNeighborhoodSearch{2}(; periodic_box, update_strategy=SerialUpdate())

# ==========================================================================================
# ==== Time integration / callbacks
tspan = (0.0, 5.0)                    # increase if you want more roll-up / cascade

semi = Semidiscretization(fluid_system;
                          neighborhood_search,
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="kh_doubleshear_")
callbacks = CallbackSet(info_callback, saving_callback)

# Adaptive RK with a dt cap (helps prevent rare catastrophic steps)
sol = solve(ode, RDPK3SpFSAL35();
            abstol=1e-8,
            reltol=1e-4,
            dtmax=2e-3,
            save_everystep=false,
            callback=callbacks)


