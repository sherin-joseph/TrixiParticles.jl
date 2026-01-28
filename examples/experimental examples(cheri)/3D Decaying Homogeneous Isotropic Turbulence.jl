# ==========================================================================================
# 3D Decaying Homogeneous Isotropic Turbulence (DHIT) in a periodic box (WCSPH, TrixiParticles)
# Di Mascio et al. (2017, Physics of Fluids) "Smoothed particle hydrodynamics method from a large eddy simulation perspective"
#  "Direct numerical simulation of three-dimensional isotropic turbulence with smoothed particle hydrodynamics"
# Ricci, Vacondio & Tafuni (Physics of Fluids, 2023)
# - Periodic cube [0, L]^3
# - Divergence-free random Fourier initial velocity field (reproducible via RNG seed)
# - No forcing -> turbulence decays
# - Outputs:
#     * VTK snapshots via SolutionSavingCallback
#     * time series CSV/JSON via PostprocessCallback (kinetic_energy, avg_density, etc.)
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq
using Random
using StaticArrays
using LinearAlgebra

# -----------------------------
# Resolution / domain
# -----------------------------
const L = 1.0                 # box size
const nx = 32                 # particles per dimension (32^3 = 32768). Increase to 48 or 64 for better inertial range.
const particle_spacing = L / nx

const n_particles = (nx, nx, nx)
const min_corner = (0.0, 0.0, 0.0)

# -----------------------------
# Physical/numerical parameters
# -----------------------------
const rho0 = 1000.0

# Target initial RMS velocity (controls Mach and Reynolds number)
const u_rms_target = 1.0

# Choose a (nominal) Reynolds number based on box length:
#   Re_L = u_rms * L / ν  =>  ν = u_rms * L / Re_L
const Re_L = 1000.0
const nu = u_rms_target * L / Re_L

# Artificial speed of sound for WCSPH: choose c0 ~ 20 * u_rms for low-Mach (Ma ~ 0.05)
const c0 = 20.0 * u_rms_target
state_equation = StateEquationCole(; sound_speed=c0, reference_density=rho0, exponent=7)

# SPH kernel
smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{3}()

# Density evolution + stabilization
density_calculator = ContinuityDensity()
density_diffusion = DensityDiffusionMolteniColagrossi(; delta=0.1)  # recommended for WCSPH turbulence
# If you want *less* damping, try delta=0.05; if you want more stability, try delta=0.15.

# Viscosity model (physical viscosity)
viscosity = ViscosityAdami(nu=nu)

# -----------------------------
# Divergence-free random Fourier initial velocity field
# -----------------------------
struct FourierMode
    k::SVector{3, Int}
    a::SVector{3, Float64}  # cosine amplitude (div-free)
    b::SVector{3, Float64}  # sine amplitude (div-free)
end

# Project v to the plane orthogonal to k (for incompressibility: k·v = 0)
@inline function project_perp(v::SVector{3,Float64}, k::SVector{3,Float64})
    kk = dot(k, k)
    kk == 0.0 && return v
    return v - (dot(v, k) / kk) * k
end

function generate_divfree_modes(; kmin=1, kmax=6, k0=4.0, seed=1)
    rng = MersenneTwister(seed)
    modes = FourierMode[]

    # "Half-space" enumeration to avoid redundant ±k pairs (sine/cos formulation stays real anyway)
    for kx in 0:kmax, ky in -kmax:kmax, kz in -kmax:kmax
        if kx == 0 && ky == 0 && kz == 0
            continue
        end
        # keep only one representative for each ±k
        if !(kx > 0 || (kx == 0 && ky > 0) || (kx == 0 && ky == 0 && kz > 0))
            continue
        end

        kmag = sqrt(kx^2 + ky^2 + kz^2)
        if kmag < kmin || kmag > kmax
            continue
        end

        k_int = SVector{3,Int}(kx, ky, kz)
        k_vec = SVector{3,Float64}(Float64(kx), Float64(ky), Float64(kz))

        # A simple smooth spectrum "bump" centered around k0 (not unique; just a reasonable default)
        # Larger k0 pushes energy to smaller scales.
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
    # Estimate u_rms by sampling random points, then scale coefficients to match u_rms_target
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

    # Scale all mode amplitudes
    modes_scaled = FourierMode[]
    for m in modes
        push!(modes_scaled, FourierMode(m.k, scale*m.a, scale*m.b))
    end

    # Return closure used by RectangularShape velocity=...
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

# Build modes + velocity function
modes = generate_divfree_modes(; kmin=1, kmax=6, k0=4.0, seed=42)
velocity_field, u_rms_est, scale = make_velocity_field(modes; L=L, u_rms_target=u_rms_target, seed=42)

@info "Initial field: estimated u_rms (before scaling) = $u_rms_est, scaling = $scale"

# -----------------------------
# Initial condition (3D lattice in periodic box)
# -----------------------------
fluid_ic = RectangularShape(particle_spacing, n_particles, min_corner;
                            density=rho0,
                            velocity=velocity_field,
                            coordinates_eltype=Float64)

# -----------------------------
# Fluid system (WCSPH)
# -----------------------------
fluid_system = WeaklyCompressibleSPHSystem(fluid_ic, density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length;
                                           viscosity=viscosity,
                                           density_diffusion=density_diffusion,
                                           shifting_technique=nothing,
                                           pressure_acceleration=nothing)

# -----------------------------
# Periodic neighborhood search + semidiscretization
# -----------------------------
periodic_box = PeriodicBox(min_corner=[0.0, 0.0, 0.0], max_corner=[L, L, L])

neighborhood_search = GridNeighborhoodSearch{3}(;
    periodic_box,
    update_strategy=SerialUpdate()
)

semi = Semidiscretization(fluid_system;
                          neighborhood_search=neighborhood_search,
                          parallelization_backend=PolyesterBackend())

                          

# -----------------------------
# Time integration
# -----------------------------
tspan = (0.0, 2.0)   # a couple of turnover times in nondimensional units
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback()

# Save VTK snapshots (for Paraview): velocity/pressure/density etc.
saving_callback = SolutionSavingCallback(dt=0.05, prefix="dhit_")

# Write time series to CSV/JSON (kinetic energy decay is the classic DHIT diagnostic)
postprocess_callback = PostprocessCallback(dt=0.01,
                                          filename="dhit_values",
                                          kinetic_energy=kinetic_energy,
                                          avg_density=avg_density,
                                          max_pressure=max_pressure,
                                          min_pressure=min_pressure)

callbacks = CallbackSet(info_callback, saving_callback, postprocess_callback)

sol = solve(ode, RDPK3SpFSAL35();
            abstol=1e-8,
            reltol=1e-4,
            dtmax=1e-2,
            save_everystep=false,
            callback=callbacks)
