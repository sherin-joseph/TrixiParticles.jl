# ==========================================================================================
# 2D Taylor–Green Vortex — unified kernel/h with Poiseuille
# A–D variants + SGS + optional artificial viscosity + Postprocessing
# Methods: :WCSPH, :EDAC, optional :IISPH
# Fixed dt (same across variants)
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq
using StaticArrays

# -----------------------------------------------------------------------------
# USER CONTROLS
# -----------------------------------------------------------------------------
const METHOD   = :WCSPH        # :WCSPH, :EDAC, :IISPH
const VARIANT  = :A            # :A, :B, :C, :D
const SGS_KIND = :AdamiSGS     # :AdamiSGS or :MorrisSGS

# Artificial viscosity mode:
#   :OFF           -> no artificial viscosity anywhere
#   :BASELINE_ONLY -> only in A/B (i.e. only when stabilization is enabled)
#   :ALWAYS        -> in all variants (replaces main viscosity unless COMBINED)
#   :COMBINED      -> add artificial viscosity on top of main viscosity
const AV_MODE  = :OFF          # :OFF, :BASELINE_ONLY, :ALWAYS, :COMBINED
const AV_ALPHA = 0.02
const AV_BETA  = 0.0

# include("sgs_viscosity.jl")  # if your SGS types live in another file

# -----------------------------------------------------------------------------
# A–D variant logic (same meaning as Poiseuille)
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

# -----------------------------------------------------------------------------
# Case setup (same physics as your original TGV)
# -----------------------------------------------------------------------------
particle_spacing = 0.02
tspan = (0.0, 5.0)

Re = 100.0
L  = 1.0

U  = 1.0
ρ0 = 1.0
c0 = 10 * U
ν  = U * L / Re

nxy = round(Int, L / particle_spacing)

# ---- Unified kernel/h (match Poiseuille)
h = 2 * particle_spacing
kernel = WendlandC2Kernel{2}()

background_pressure = c0^2 * ρ0
b = -8pi^2 / Re

pressure_function(pos, t) = begin
    x, y = pos[1], pos[2]
    -U^2 * exp(2b*t) * (cos(4pi*x) + cos(4pi*y)) / 4
end

velocity_function(pos, t) = begin
    x, y = pos[1], pos[2]
    v = U * exp(b*t) * [-cos(2pi*x)*sin(2pi*y),
                         sin(2pi*x)*cos(2pi*y)]
    SVector{2}(v)
end

# -----------------------------------------------------------------------------
# Fixed dt (same recipe as Poiseuille; then frozen)
# -----------------------------------------------------------------------------
cfl_c = 0.20
cfl_v = 0.10
dt_acoustic = cfl_c * h / (c0 + U)
dt_viscous  = cfl_v * h^2 / ν
dt = min(dt_acoustic, dt_viscous)

# -----------------------------------------------------------------------------
# Stabilization knobs (baseline vs reduced)
# - baseline := shifting + diffusion + perturbation
# -----------------------------------------------------------------------------
shifting_technique = vs.use_stabilization ?
    TransportVelocityAdami(; background_pressure) : nothing

perturb_coordinates = vs.use_stabilization

density_diffusion = vs.use_stabilization ?
    DensityDiffusionMolteniColagrossi(delta=0.1) : nothing

# -----------------------------------------------------------------------------
# Viscosity selection (physical vs SGS) + optional artificial viscosity
# -----------------------------------------------------------------------------
function base_viscosity(use_sgs::Bool)
    if !use_sgs
        return ViscosityAdami(; nu=ν)
    end
    if SGS_KIND == :AdamiSGS
        return ViscosityAdamiSGS(; nu=ν, C_S=0.1, epsilon=0.001)
    elseif SGS_KIND == :MorrisSGS
        return ViscosityMorrisSGS(; nu=ν, C_S=0.1, epsilon=0.001)
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
        # replacement mode
        return use_av ? artificial_viscosity() : main
    end
end

viscosity = pick_viscosity()

# -----------------------------------------------------------------------------
# Particles
# -----------------------------------------------------------------------------
fluid = RectangularShape(
    particle_spacing, (nxy, nxy), (0.0, 0.0);
    coordinates_perturbation = perturb_coordinates ? 0.2 : nothing,
    density  = ρ0,
    pressure = pos -> pressure_function(pos, 0.0),
    velocity = pos -> velocity_function(pos, 0.0)
)

# -----------------------------------------------------------------------------
# Build system
# -----------------------------------------------------------------------------
function build_system(method::Symbol)
    if method == :WCSPH
        density_calc = ContinuityDensity()
        eos = StateEquationCole(; sound_speed=c0, reference_density=ρ0, exponent=1)

        return WeaklyCompressibleSPHSystem(
            fluid, density_calc, eos, kernel, h;
            pressure_acceleration = TrixiParticles.inter_particle_averaged_pressure,
            viscosity = viscosity,
            shifting_technique = shifting_technique,
            density_diffusion = density_diffusion
        )

    elseif method == :EDAC
        density_calc = SummationDensity()
        return EntropicallyDampedSPHSystem(
            fluid, kernel, h, c0;
            density_calculator = density_calc,
            viscosity = viscosity,
            shifting_technique = shifting_technique
        )

    elseif method == :IISPH
        return ImplicitIncompressibleSPHSystem(
            fluid, kernel, h, ρ0;
            viscosity = viscosity,
            time_step = dt
        )
    else
        error("Unknown METHOD=$method")
    end
end

system = build_system(METHOD)

periodic_box = PeriodicBox(min_corner=[0.0, 0.0], max_corner=[L, L])
semi = Semidiscretization(system; neighborhood_search=GridNeighborhoodSearch{2}(; periodic_box))
ode  = semidiscretize(semi, tspan)

# -----------------------------------------------------------------------------
# Callbacks / solve (fixed dt + postprocessing)
# -----------------------------------------------------------------------------
info_callback = InfoCallback(interval=100)

prefix = "tgv_$(String(METHOD))_$(String(VARIANT))_$(sgs_tag)"
saving_callback = SolutionSavingCallback(dt=0.02, prefix=prefix)

# Postprocessing: kinetic energy + average pressure vs time (dt_pp = 0.01)
pp_filename = prefix * "_pp"
pp_callback = PostprocessCallback(; dt=0.01,
                                  filename=pp_filename,
                                  kinetic_energy,
                                  avg_pressure,
                                  write_file_interval=0)

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback(), pp_callback)

if METHOD == :IISPH
    sol = solve(ode, SymplecticEuler(); dt=dt, adaptive=false, save_everystep=false, callback=callbacks)
else
    sol = solve(ode, RDPK3SpFSAL49(); dt=dt, adaptive=false, save_everystep=false, callback=callbacks)
end

@info "Done" METHOD VARIANT sgs_tag AV_MODE dt h kernel pp_filename
