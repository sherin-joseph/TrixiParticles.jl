# ==========================================================================================
# 3D Decaying Homogeneous Isotropic Turbulence (DHIT) in a periodic box (WCSPH, TrixiParticles)
#
# - Periodic cube [0, L]^3
# - Divergence-free random Fourier initial velocity field
# - No forcing -> turbulence decays
# - Variants:
#     (A) Baseline (stabilization only)
#     (B) Baseline + SGS
#     (C) Reduced stabilization
#     (D) Reduced stabilization + SGS
# - Outputs:
#     * VTK snapshots via SolutionSavingCallback
#     * time series CSV/JSON via PostprocessCallback
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq
using Random
using StaticArrays
using LinearAlgebra

# -----------------------------------------------------------------------------
# USER CONTROLS (aligned with TGV + Poiseuille)
# -----------------------------------------------------------------------------
const VARIANT  = :A            # :A, :B, :C, :D
const SGS_KIND = :AdamiSGS     # :AdamiSGS or :MorrisSGS

# Artificial viscosity mode (same semantics as in 2D cases):
#   :OFF           -> no artificial viscosity anywhere
#   :BASELINE_ONLY -> only in A/B (stabilization variants)
#   :ALWAYS        -> in all variants, replaces main viscosity
#   :COMBINED      -> add artificial viscosity on top of main viscosity
const AV_MODE  = :OFF          # :OFF, :BASELINE_ONLY, :ALWAYS, :COMBINED
const AV_ALPHA = 0.02
const AV_BETA  = 0.0

# include("sgs_viscosity.jl")  # if your SGS types live externally

# -----------------------------------------------------------------------------
# Variant logic (same as TGV/Poiseuille)
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

# Tag for filenames: only say SGS when actually used
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
# Resolution / domain
# -----------------------------------------------------------------------------
const L  = 1.0
const nx = 32                 # 32^3 particles; increase to 48 or 64 for better inertial range
const particle_spacing = L / nx

const n_particles = (nx, nx, nx)
const min_corner = (0.0, 0.0, 0.0)

# -----------------------------------------------------------------------------
# Physical/numerical parameters
# -----------------------------------------------------------------------------
const rho0 = 1000.0

# Target RMS velocity (controls Mach and Re)
const u_rms_target = 1.0

# Re_L = u_rms * L / ν => ν = u_rms * L / Re_L
const Re_L = 1000.0
const nu   = u_rms_target * L / Re_L

# Artificial speed of sound for WCSPH (low Mach)
const c0 = 20.0 * u_rms_target
state_equation = StateEquationCole(; sound_speed=c0, reference_density=rho0, exponent=7)

# Unified kernel/h (similar philosophy as TGV/Poiseuille, but 3D)
smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{3}()  # keep as in original DHIT

# Density evolution base calculator
density_calculator = ContinuityDensity()

# -----------------------------------------------------------------------------
# Stabilization knobs (baseline vs reduced)
# baseline stabilization := density diffusion + TVF shifting
# -----------------------------------------------------------------------------
background_pressure = rho0 * c0^2

density_diffusion = vs.use_stabilization ?
    DensityDiffusionMolteniColagrossi(; delta=0.1) : nothing

shifting_technique = vs.use_stabilization ?
    TransportVelocityAdami(; background_pressure) : nothing

# -----------------------------------------------------------------------------
# Viscosity selection (physical ν vs SGS) + optional artificial viscosity
# -----------------------------------------------------------------------------
function base_viscosity(use_sgs::Bool)
    if !use_sgs
        return ViscosityAdami(nu=nu)
    end
    if SGS_KIND == :AdamiSGS
        return ViscosityAdamiSGS(; nu=nu, C_S=0.1, epsilon=0.001)
    elseif SGS_KIND == :MorrisSGS
        return ViscosityMorrisSGS(; nu=nu, C_S=0.1, epsilon=0.001)
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
# Divergence-free random Fourier initial velocity field
# -----------------------------------------------------------------------------
struct FourierMode
    k::SVector{3, Int}
    a::SVector{3, Float64}  # cosine amplitude (div-free)
    b::SVector{3, Float64}  # sine amplitude (div-free)
end

@inline function project_perp(v::SVector{3,Float64}, k::SVector{3,Float64})
    kk = dot(k, k)
    kk == 0.0 && return v
    return v - (dot(v, k) / kk) * k
end

function generate_divfree_modes(; kmin=1, kmax=6, k0=4.0, seed=1)
    rng = MersenneTwister(seed)
    modes = FourierMode[]

    # half-space enumeration to avoid redundant ±k
    for kx in 0:kmax, ky in -kmax:kmax, kz in -kmax:kmax
        if kx == 0 && ky == 0 && kz == 0
            continue
        end
        # keep one representative for each ±k
        if !(kx > 0 || (kx == 0 && ky > 0) || (kx == 0 && ky == 0 && kz > 0))
            continue
        end

        kmag = sqrt(kx^2 + ky^2 + kz^2)
        if kmag < kmin || kmag > kmax
            continue
        end

        k_int = SVector{3,Int}(kx, ky, kz)
        k_vec = SVector{3,Float64}(Float64(kx), Float64(ky), Float64(kz))

        # simple smooth spectrum bump centered at k0
        amp = (kmag^2) * exp(-(kmag / k0)^2)

        r1 = SVector{3,Float64}(randn(rng), randn(rng), randn(rng))
        r2 = SVector{3,Float64}(randn(rng), randn(rng), randn(rng))

        a = project_perp(r1, k_vec)
        b = project_perp(r2, k_vec)

        na = norm(a); nb = norm(b)
        if na < 1e-12 || nb < 1e-12
            continue
        end

        a = (amp / na) * a
        b = (amp / nb) * b

        push!(modes, FourierMode(k_int, a, b))
    end

    return modes
end

function make_velocity_field(modes::Vector{FourierMode}; L=1.0, u_rms_target=1.0, seed=1)
    rng = MersenneTwister(seed + 12345)
    nsamples = 2000
    sum_u2 = 0.0

    @inline function u_raw(x)
        x1, x2, x3 = x[1], x[2], x[3]
        u = SVector{3,Float64}(0.0, 0.0, 0.0)
        for m in modes
            kx, ky, kz = m.k
            θ = 2π * (kx * x1 / L + ky * x2 / L + kz * x3 / L)
            u += m.a * cos(θ) + m.b * sin(θ)
        end
        return u
    end

    for _ in 1:nsamples
        x = SVector{3,Float64}(rand(rng)*L, rand(rng)*L, rand(rng)*L)
        u = u_raw(x)
        sum_u2 += dot(u, u)
    end

    u_rms = sqrt((sum_u2 / nsamples) / 3.0)   # RMS per component
    scale = u_rms_target / u_rms

    modes_scaled = FourierMode[]
    for m in modes
        push!(modes_scaled, FourierMode(m.k, scale*m.a, scale*m.b))
    end

    velocity_field = function (x)
        x1, x2, x3 = x[1], x[2], x[3]
        u = SVector{3,Float64}(0.0, 0.0, 0.0)
        for m in modes_scaled
            kx, ky, kz = m.k
            θ = 2π * (kx * x1 / L + ky * x2 / L + kz * x3 / L)
            u += m.a * cos(θ) + m.b * sin(θ)
        end
        return u
    end

    return velocity_field, u_rms, scale
end

modes = generate_divfree_modes(; kmin=1, kmax=6, k0=4.0, seed=42)
velocity_field, u_rms_est, scale = make_velocity_field(modes; L=L, u_rms_target=u_rms_target, seed=42)

@info "DHIT initial field" u_rms_est scale

# -----------------------------------------------------------------------------
# Initial condition (3D lattice in periodic box)
# -----------------------------------------------------------------------------
fluid_ic = RectangularShape(particle_spacing, n_particles, min_corner;
                            density=rho0,
                            velocity=velocity_field,
                            coordinates_eltype=Float64)

# -----------------------------------------------------------------------------
# Fluid system (WCSPH) — same pattern as other WCSPH setups
# -----------------------------------------------------------------------------
fluid_system = WeaklyCompressibleSPHSystem(fluid_ic, density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length;
                                           viscosity=viscosity,
                                           density_diffusion=density_diffusion,
                                           shifting_technique=shifting_technique,
                                           pressure_acceleration=nothing)

# -----------------------------------------------------------------------------
# Periodic neighborhood search + semidiscretization
# -----------------------------------------------------------------------------
periodic_box = PeriodicBox(min_corner=[0.0, 0.0, 0.0], max_corner=[L, L, L])

neighborhood_search = GridNeighborhoodSearch{3}(;
    periodic_box,
    update_strategy=SerialUpdate()
)

semi = Semidiscretization(fluid_system;
                          neighborhood_search=neighborhood_search,
                          parallelization_backend=PolyesterBackend())

# -----------------------------------------------------------------------------
# Time integration setup (fixed dt, as in TGV/Poiseuille)
# -----------------------------------------------------------------------------
tspan = (0.0, 2.0)   # a couple of turnover times
ode = semidiscretize(semi, tspan)

# CFL/viscous constraints for WCSPH
cfl_c = 0.20
cfl_v = 0.10
dt_acoustic = cfl_c * smoothing_length / (c0 + u_rms_target)
dt_viscous  = cfl_v * smoothing_length^2 / nu
dt = min(dt_acoustic, dt_viscous)

@info "DHIT dt" dt dt_acoustic dt_viscous

info_callback = InfoCallback()

# Save VTK snapshots (Paraview)
prefix = "dhit_WCSPH_$(String(VARIANT))_$(sgs_tag)"
saving_callback = SolutionSavingCallback(dt=0.05, prefix=prefix)

# Postprocessing: DHIT diagnostics (similar style to PPT + hydrostatic example)
# - kinetic_energy: classic decay curve
# - avg_density: compressibility error indicator
# - max/min pressure: pressure range evolution
pp_filename = prefix * "_pp"
postprocess_callback = PostprocessCallback(; dt=0.01,
                                           filename=pp_filename,
                                           kinetic_energy=kinetic_energy,
                                           avg_density=avg_density,
                                           max_pressure=max_pressure,
                                           min_pressure=min_pressure,
                                           write_file_interval=0)

callbacks = CallbackSet(info_callback, saving_callback, postprocess_callback)

sol = solve(ode, RDPK3SpFSAL35();
            dt=dt,
            adaptive=false,
            abstol=1e-8,
            reltol=1e-4,
            save_everystep=false,
            callback=callbacks)

@info "Done DHIT" VARIANT sgs_tag AV_MODE dt
