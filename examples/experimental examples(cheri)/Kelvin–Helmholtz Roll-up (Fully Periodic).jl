# ==========================================================================================
# 2D Double Shear Layer / Kelvin–Helmholtz Roll-up (Fully Periodic, WCSPH, TrixiParticles)
#
# Variants:
#   (A) Baseline               : stabilization ON,     no SGS
#   (B) Baseline + SGS         : stabilization ON,     SGS ON
#   (C) Reduced stabilization  : stabilization OFF,    no SGS
#   (D) Reduced stabilization + SGS : stabilization OFF, SGS ON
#
# Output:
#   - VTK snapshots via SolutionSavingCallback
#   - Time series via PostprocessCallback (kinetic_energy, avg_pressure)
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq
using StaticArrays

# -----------------------------------------------------------------------------
# USER CONTROLS (aligned with TGV / Poiseuille / DHIT)
# -----------------------------------------------------------------------------
const VARIANT  = :A            # :A, :B, :C, :D
const SGS_KIND = :AdamiSGS     # :AdamiSGS or :MorrisSGS

# Artificial viscosity mode:
#   :OFF           -> no artificial viscosity
#   :BASELINE_ONLY -> only when stabilization is ON (A/B)
#   :ALWAYS        -> all variants, replaces main viscosity
#   :COMBINED      -> added on top of main viscosity
const AV_MODE  = :OFF          # :OFF, :BASELINE_ONLY, :ALWAYS, :COMBINED
const AV_ALPHA = 0.02
const AV_BETA  = 0.0

# include("sgs_viscosity.jl")  # if your SGS types live in another file

# -----------------------------------------------------------------------------
# Variant logic (same semantics as other cases)
# -----------------------------------------------------------------------------
function variant_settings(variant::Symbol)
    if variant == :A
        return (use_stabilization=true,  use_sgs=false)
    elseif variant == :B
        return (use_stabilization=true,  use_sgs=true)
    elseif variant == :C
        return (use_stabilization=false, use_sgs=false)
    elseif variant == :D
        return (use_stabilization=false, use_sgs=true)
    else
        error("Unknown VARIANT=$variant. Use :A,:B,:C,:D")
    end
end
vs = variant_settings(VARIANT)

# Tag used only for filenames, based on whether SGS is actually active
sgs_tag = vs.use_sgs ? String(SGS_KIND) : "noSGS"

# -----------------------------------------------------------------------------
# Combined viscosity wrapper (for AV_MODE == :COMBINED)
# -----------------------------------------------------------------------------
struct CombinedViscosity{V1,V2}
    v1::V1
    v2::V2
end

@inline function (cv::CombinedViscosity)(particle_system, neighbor_system,
                                        v_particle_system, v_neighbor_system,
                                        particle, neighbor, pos_diff, distance,
                                        sound_speed, m_a, m_b, rho_a, rho_b, grad_kernel)
    return cv.v1(particle_system, neighbor_system,
                 v_particle_system, v_neighbor_system,
                 particle, neighbor, pos_diff, distance,
                 sound_speed, m_a, m_b, rho_a, rho_b, grad_kernel) .+
           cv.v2(particle_system, neighbor_system,
                 v_particle_system, v_neighbor_system,
                 particle, neighbor, pos_diff, distance,
                 sound_speed, m_a, m_b, rho_a, rho_b, grad_kernel)
end

# ==========================================================================================
# ==== Resolution / Domain
particle_spacing = 0.01                 # decrease for better turbulence range (cost ↑)
Lx_target, Ly_target = 1.0, 1.0

nx = round(Int, Lx_target / particle_spacing)
ny = round(Int, Ly_target / particle_spacing)
Lx = nx * particle_spacing
Ly = ny * particle_spacing

# ==========================================================================================
# ==== Double shear layer initial condition parameters
U0 = 1.0                                # velocity scale
delta = 0.05 * Ly                       # shear layer thickness
eps_v = 0.01 * U0                       # perturbation amplitude

@inline function u_shear(y)
    ymid = 0.5 * Ly
    if y < ymid
        return U0 * tanh((y - 0.25 * Ly) / delta)
    else
        return U0 * tanh((0.75 * Ly - y) / delta)
    end
end

@inline function v_perturb(x, y)
    return eps_v * sin(2π * x / Lx)
end

velocity_ic = (x -> begin
    xx, yy = x[1], x[2]
    SVector(u_shear(yy), v_perturb(xx, yy))
end)

fluid_density = 1000.0

# ==========================================================================================
# ==== Initial condition: rectangular lattice of particles
fluid_ic = RectangularShape(particle_spacing, (nx, ny), (0.0, 0.0);
                            density=fluid_density,
                            velocity=velocity_ic,
                            coordinates_perturbation=nothing)

# ==========================================================================================
# ==== Fluid model (WCSPH)
smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

sound_speed = 20 * U0
state_equation = StateEquationCole(; sound_speed,
                                   reference_density=fluid_density,
                                   exponent=7,
                                   background_pressure=0.0)

density_calculator = ContinuityDensity()

# Reynolds number control: nu = U0 * Ly / Re
Re = 5_000.0
nu = U0 * Ly / Re

# -----------------------------------------------------------------------------
# Stabilization knobs
# baseline stabilization := density diffusion + shifting + tensile-instability control
# -----------------------------------------------------------------------------
density_diffusion = vs.use_stabilization ?
    DensityDiffusionMolteniColagrossi(; delta=0.1) : nothing

shifting_technique = vs.use_stabilization ?
    ConsistentShiftingSun2019() : nothing

pressure_acceleration = vs.use_stabilization ?
    tensile_instability_control : nothing

# -----------------------------------------------------------------------------
# Viscosity selection (Adami vs SGS) + optional artificial viscosity
# -----------------------------------------------------------------------------
function base_viscosity(use_sgs::Bool)
    if !use_sgs
        return ViscosityAdami(; nu)
    end
    if SGS_KIND == :AdamiSGS
        return ViscosityAdamiSGS(; nu, C_S=0.12)  # your original default
    elseif SGS_KIND == :MorrisSGS
        return ViscosityMorrisSGS(; nu, C_S=0.12)
    else
        error("Unknown SGS_KIND=$SGS_KIND")
    end
end

artificial_viscosity() = ArtificialViscosityMonaghan(; alpha=AV_ALPHA, beta=AV_BETA, epsilon=0.01)

function pick_viscosity()
    main = base_viscosity(vs.use_sgs)
    use_av = (AV_MODE == :ALWAYS) || (AV_MODE == :BASELINE_ONLY && vs.use_stabilization)

    if AV_MODE == :COMBINED
        return use_av ? CombinedViscosity(main, artificial_viscosity()) : main
    else
        return use_av ? artificial_viscosity() : main
    end
end

viscosity = pick_viscosity()

# -----------------------------------------------------------------------------
# Build WCSPH system
# -----------------------------------------------------------------------------
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

# Fixed dt (same philosophy as TGV/Poiseuille)
cfl_c = 0.20
cfl_v = 0.10
dt_acoustic = cfl_c * smoothing_length / (sound_speed + U0)
dt_viscous  = cfl_v * smoothing_length^2 / nu
dt = min(dt_acoustic, dt_viscous)

@info "KH double shear dt" dt dt_acoustic dt_viscous

info_callback = InfoCallback(interval=100)

prefix = "kh_WCSPH_$(String(VARIANT))_$(sgs_tag)"
saving_callback = SolutionSavingCallback(dt=0.02, prefix=prefix)

# Postprocessing: same style as TGV/Poiseuille: KE + avg pressure
pp_filename = prefix * "_pp"
pp_callback = PostprocessCallback(; dt=0.01,
                                  filename=pp_filename,
                                  kinetic_energy=kinetic_energy,
                                  avg_pressure=avg_pressure,
                                  write_file_interval=0)

callbacks = CallbackSet(info_callback, saving_callback, pp_callback)

sol = solve(ode, RDPK3SpFSAL35();
            dt=dt,
            adaptive=false,
            abstol=1e-8,
            reltol=1e-4,
            save_everystep=false,
            callback=callbacks)

@info "Done KH double shear" VARIANT sgs_tag AV_MODE dt
