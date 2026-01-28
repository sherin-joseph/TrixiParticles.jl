# ==========================================================================================
# Brown & Minion (1995) doubly-periodic double shear layer benchmark (unit square)
#
# Target problem (paper):
#   Domain: [0,1) × [0,1), periodic in x and y
#   IC:
#     u(y) = tanh(ρ (y - 0.25)) for y ≤ 0.5
#          = tanh(ρ (0.75 - y)) for y > 0.5
#     v(x) = δ sin(2π x), δ = 0.05
#
#   Thick case: ρ=30,  ν=1/10000
#   Thin  case: ρ=100, ν=1/20000
#
# What this script does:
#   - Runs WCSPH in TrixiParticles on the same IC/ν and periodic domain.
#   - Uses a "paper-matching" knob set: higher sound speed (lower Mach),
#     optional shifting (for particle regularity), and *no* density diffusion by default.
#   - Writes VTU/PVD snapshots for contour/spectrum postprocessing.
#
# Recommended:
#   - For meaningful comparison: N=128 (quick), N=256 (good), N=512 (best, expensive)
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq
using StaticArrays

# -------------------------
# Benchmark choice / resolution
# -------------------------
const CASE = :thick            # :thick or :thin
const N    = 256             # 128/256/512 for comparisons (64 mainly for debugging)

# Stabilization toggles:
const USE_SHIFTING        = false      # true keeps particle distribution clean; set false for "closest physics"
const USE_DENS_DIFFUSION  = false    # paper does not have this; only enable if you get density noise

# -------------------------
# Domain / discretization
# -------------------------
const Lx = 1.0
const Ly = 1.0
const dx = Lx / N
const dy = Ly / N
@assert isapprox(dx, dy; atol=0, rtol=0) "Use square grid: Lx/N must equal Ly/N"

# Particle positions at cell-centers like FD: (i-1/2)h
const x0 = 0.5 * dx
const y0 = 0.5 * dy

# -------------------------
# Brown–Minion parameters
# -------------------------
const δ = 0.05
const ρ, ν = CASE == :thick ? (30.0, 1.0 / 10_000.0) :
              CASE == :thin  ? (100.0, 1.0 / 20_000.0) :
              error("Unknown CASE = $CASE")

const ρ0 = 1.0   # nondimensional reference density

# -------------------------
# Initial condition
# -------------------------
@inline function u_ic(y::Float64)
    if y <= 0.5
        return tanh(ρ * (y - 0.25))
    else
        return tanh(ρ * (0.75 - y))
    end
end

@inline v_ic(x::Float64) = δ * sin(2π * x)

velocity_ic = x -> begin
    xx, yy = x[1], x[2]
    SVector(u_ic(yy), v_ic(xx))
end

fluid_ic = RectangularShape(dx, (N, N), (x0, y0);
                            density=ρ0,
                            velocity=velocity_ic,
                            coordinates_perturbation=nothing)

# -------------------------
# WCSPH model choices
# -------------------------
# Kernel + smoothing length
const h = 1.3 * dx

# Prefer Wendland if available (often less noisy), else fall back to cubic spline.
const kernel = if isdefined(TrixiParticles, :WendlandC2Kernel)
    WendlandC2Kernel{2}()
else
    @warn "WendlandC2Kernel not found; using SchoenbergCubicSplineKernel instead."
    SchoenbergCubicSplineKernel{2}()
end

# Sound speed for low Mach number (more incompressible)
const Umax_est = 1.0
const c0       = 500.0 * Umax_est     # try 150 for cleaner spectra (smaller dt)

state_equation = StateEquationCole(; sound_speed=c0,
                                   reference_density=ρ0,
                                   exponent=7,
                                   background_pressure=0.0)

# Density evolution and viscosity
density_calculator = ContinuityDensity()
viscosity = ViscosityAdami(; nu=ν)

# Stabilizers (paper does not include these; keep minimal)
density_diffusion  = USE_DENS_DIFFUSION ? DensityDiffusionMolteniColagrossi(; delta=0.01) : nothing
shifting_technique = USE_SHIFTING       ? ConsistentShiftingSun2019() : nothing

fluid_system = WeaklyCompressibleSPHSystem(fluid_ic,
                                          density_calculator,
                                          state_equation,
                                          kernel,
                                          h;
                                          viscosity=viscosity,
                                          density_diffusion=density_diffusion,
                                          shifting_technique=shifting_technique,
                                          pressure_acceleration=nothing)

# -------------------------
# Periodic neighborhood search
# -------------------------
periodic_box = PeriodicBox(min_corner=[0.0, 0.0], max_corner=[Lx, Ly])
neighborhood_search = GridNeighborhoodSearch{2}(; periodic_box,
                                               update_strategy=SerialUpdate())

# -------------------------
# Time integration
# -------------------------
const t_end = 1.6
tspan = (0.0, t_end)

semi = Semidiscretization(fluid_system;
                          neighborhood_search,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

# Callbacks
info_callback = InfoCallback(interval=100)

# Writes VTU/PVD snapshots (for ParaView and postprocessing)
saving_callback = SolutionSavingCallback(dt=0.02,
                                        prefix="_doubleshear_$(CASE)_N$(N)_")

callbacks = CallbackSet(info_callback, saving_callback)

# Stability timestep caps: acoustic + viscous
dt_acoustic = 0.25 * dx / c0
dt_visc     = 0.125 * dx^2 / ν
dtmax       = min(dt_acoustic, dt_visc)

# Also store solution exactly at benchmark times (for quick checks)
save_times = [0.0, 0.1, 0.8, 1.2, 1.6]
save_times = filter(t -> t <= t_end + 1e-12, save_times)

sol = solve(ode, RDPK3SpFSAL35();
            abstol=1e-8,
            reltol=1e-4,
            dtmax=dtmax,
            save_everystep=false,
            saveat=save_times,
            callback=callbacks)

println("\nDone.")
println("CASE=$CASE, N=$N, ν=$ν, ρ=$ρ")
println("c0=$c0, h=$(h), shifting=$(USE_SHIFTING), density_diffusion=$(USE_DENS_DIFFUSION)")
println("Saved snapshots with prefix: bm_doubleshear_$(CASE)_N$(N)_")
println("Stored solution at times: ", sol.t)
