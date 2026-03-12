# ==========================================================================================
# 3D Pulsatile Flow in Aorta-Like STL Geometry — Elastic TLSPH Wall + Windkessel Outlet
#
# Extends `pipe_flow_3d_aorta_tlsph.jl` with two physiological additions:
#
#   1. **Pulsatile inlet** : a cardiac-cycle waveform (sine-squared systolic peak +
#      diastolic plateau) replaces the steady ramped inflow.
#
#   2. **Windkessel outlet** : an RCR (proximal resistance R1, capacitor C, distal
#      resistance R2) lumped model provides a realistic, time-varying outlet pressure
#      instead of the fixed p=0 condition.  The model is auto-tuned from the domain
#      geometry and flow parameters using `auto_tune_windkessel!`.
#
# The vessel wall is elastic (TLSPH), identical to the parent file.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq
using LinearAlgebra: dot, norm, cross
using StaticArrays: SVector

# ==========================================================================================
# ==== Windkessel model (include local module)
include(joinpath(@__DIR__, "WindkesselModel3d.jl"))
using .WindkesselModel

# ==========================================================================================
# ==== Resolution
particle_spacing = 0.003

# Make sure that the kernel support of fluid particles at an open boundary is always
# fully sampled.
open_boundary_layers = 6

# Wall thickness for the elastic structure
structure_layers = 3

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 1.6)   # two full cardiac cycles (T_cardiac = 0.8 s)

fluid_density = 1000.0
reynolds_number = 10

# --------------------------------------------------------------------------
# Pulsatile (cardiac) inlet parameters
# --------------------------------------------------------------------------
const T_cardiac   = 0.8          # cardiac period [s]  (~75 bpm)
const T_systole   = T_cardiac / 3 # systolic duration  (~0.267 s)
const v_mean      = 0.05         # cycle-averaged inlet speed [m/s]
const v_peak      = v_mean * 3.0  # systolic peak speed [m/s]
const v_diastolic = v_mean * 0.3  # diastolic baseline speed [m/s]
const inflow_ramp_time = T_cardiac # ramp over one cardiac cycle to avoid impulse at t=0

"""
    pulsatile_inlet_speed(t)

Returns the prescribed inlet speed at time `t` following a simplified aortic
flow waveform: a sine-squared systolic pulse in the first third of each cardiac
cycle, and a low-flow diastolic plateau in the remainder.
A linear ramp over one cardiac cycle avoids the impulsive start.
"""
function pulsatile_inlet_speed(t)
    ramp   = min(t / inflow_ramp_time, 1.0)
    t_mod  = mod(t, T_cardiac)
    speed  = t_mod < T_systole ?
             v_peak * sin(π * t_mod / T_systole)^2 :
             v_diastolic
    return ramp * speed
end

# ==========================================================================================
# ==== STL Geometry Loading

geometry_dir    = @__DIR__
aorta_geometry  = load_geometry(joinpath(geometry_dir, "arch_tube_closed_fixed_winding.stl"))
inlet_geometry  = load_geometry(joinpath(geometry_dir, "inlet_disk.stl"))
outlet_geometry = load_geometry(joinpath(geometry_dir, "outlet_disk.stl"))

function orient_normal_towards_domain(face, face_normal, domain_center)
    face_center = (face[1] + face[2] + face[3]) / 3
    return dot(face_normal, domain_center - face_center) > 0 ? face_normal : -face_normal
end

inlet_face_data  = planar_geometry_to_face(inlet_geometry)
outlet_face_data = planar_geometry_to_face(outlet_geometry)

domain_center = (aorta_geometry.min_corner + aorta_geometry.max_corner) / 2
inlet_normal  = orient_normal_towards_domain(inlet_face_data.face,  inlet_face_data.face_normal,  domain_center)
outlet_normal = orient_normal_towards_domain(outlet_face_data.face, outlet_face_data.face_normal, domain_center)

# Geometry-consistent inlet velocity direction (follows STL face normal)
function pulsatile_velocity_function3d(pos, t)
    return pulsatile_inlet_speed(t) * inlet_normal
end

# ==========================================================================================
# ==== Windkessel outlet pressure (RCR model)

# Estimate the inlet cross-section area from the two edges of the face triangle.
# The actual area of a triangle is half the magnitude of the cross product of its two edges;
# we use the full parallelogram area as a conservative estimate of the open-boundary face.
inlet_edge1 = inlet_face_data.face[2] - inlet_face_data.face[1]
inlet_edge2 = inlet_face_data.face[3] - inlet_face_data.face[1]
A_inlet     = norm(cross(Vector(inlet_edge1), Vector(inlet_edge2)))

# Instantiate Windkessel with default diastolic pressure ~80 mmHg = 10 660 Pa.
# Parameters are then auto-tuned from domain geometry and flow conditions.
# Start from p0 = 0 so there is no initial pressure shock against the zero-velocity fluid.
# The Windkessel pressure will build up naturally from the first cardiac cycle.
wk = Windkessel(; R1 = 100.0, R2 = 900.0, C = 1.0e-5, p0 = 0.0)

characteristic_length     = maximum(aorta_geometry.max_corner - aorta_geometry.min_corner)
kinematic_viscosity       = v_mean * characteristic_length / reynolds_number
domain_size               = Tuple(aorta_geometry.max_corner - aorta_geometry.min_corner)

auto_tune_windkessel!(wk, domain_size, kinematic_viscosity, Float64(reynolds_number),
                      T_cardiac; ρ = fluid_density, ΔP_target = 1333.0)  # ~10 mmHg drop

# Guard so the Windkessel is advanced only once per unique time step, even though
# `reference_pressure_out` is evaluated once per outlet boundary particle.
const wk_last_t = Ref(-Inf)

function get_windkessel_pressure(t)
    if t > wk_last_t[]
        Q_out = A_inlet * pulsatile_inlet_speed(t)   # mass-conservation approximation
        update_windkessel!(wk, Q_out, t)
        wk_last_t[] = t
    end
    return wk.p
end

# ==========================================================================================
# ==== Fluid Particles (from STL interior)

point_in_geometry_algorithm = WindingNumberJacobson(; geometry = aorta_geometry,
                                                    hierarchical_winding = true)

fluid = ComplexShape(aorta_geometry; particle_spacing, density = fluid_density,
                     velocity = (0.0, 0.0, 0.0), point_in_geometry_algorithm)

NDIMS = ndims(fluid)

face_extent_1    = norm(inlet_face_data.face[2] - inlet_face_data.face[1])
face_extent_2    = norm(inlet_face_data.face[3] - inlet_face_data.face[1])
n_face_particles = ceil(Int, face_extent_1 / particle_spacing) *
                   ceil(Int, face_extent_2 / particle_spacing)
n_buffer_particles = max(300, 10 * n_face_particles)

# ==========================================================================================
# ==== WCSPH Fluid System

wcsph = true

smoothing_length = 1.5 * particle_spacing
smoothing_kernel = WendlandC2Kernel{NDIMS}()

fluid_density_calculator = ContinuityDensity()

viscosity   = ViscosityAdami(nu = kinematic_viscosity)
sound_speed = 20 * v_peak   # Mach << 1 even at systolic peak

if wcsph
    state_equation   = StateEquationCole(; sound_speed, reference_density = fluid_density,
                                         exponent = 1)
    # delta = 0.3 is aggressive and can seed pressure oscillations; 0.1 is safer.
    density_diffusion = DensityDiffusionMolteniColagrossi(delta = 0.1)

    fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                               state_equation, smoothing_kernel,
                                               density_diffusion  = density_diffusion,
                                               smoothing_length,
                                               viscosity          = viscosity,
                                               shifting_technique = ParticleShiftingTechnique(v_max_factor = 0.3),
                                               buffer_size        = n_buffer_particles)
else
    state_equation = nothing

    fluid_system = EntropicallyDampedSPHSystem(fluid, smoothing_kernel,
                                               smoothing_length, sound_speed,
                                               viscosity            = viscosity,
                                               density_calculator   = fluid_density_calculator,
                                               shifting_technique   = ParticleShiftingTechnique(),
                                               buffer_size          = n_buffer_particles)
end

# ==========================================================================================
# ==== Open Boundary (pulsatile inlet + Windkessel outlet)

open_boundary_model = BoundaryModelMirroringTafuni(; mirror_method = ZerothOrderMirroring())

# ---- Inlet (pulsatile velocity, no pressure/density prescription) ----
inflow = BoundaryZone(; boundary_face      = inlet_face_data.face,
                      face_normal         = inlet_normal,
                      open_boundary_layers,
                      density             = fluid_density,
                      particle_spacing,
                      reference_density   = nothing,
                      reference_pressure  = nothing,
                      reference_velocity  = pulsatile_velocity_function3d,
                      boundary_type       = InFlow())

# ---- Outlet (Windkessel pressure, no velocity/density prescription) ----
outflow = BoundaryZone(; boundary_face      = outlet_face_data.face,
                       face_normal         = outlet_normal,
                       open_boundary_layers,
                       density             = fluid_density,
                       particle_spacing,
                       reference_density   = nothing,
                       reference_pressure  = (pos, t) -> get_windkessel_pressure(t),
                       reference_velocity  = nothing,
                       boundary_type       = OutFlow())

open_boundary = OpenBoundarySystem(inflow, outflow; fluid_system,
                                   boundary_model = open_boundary_model,
                                   buffer_size    = n_buffer_particles)

# ==========================================================================================
# ==== Elastic Structure (TLSPH wall)

structure_density  = 1200.0
E                  = 1.0e6
# nu = 0.45 is nearly incompressible and causes TLSPH volumetric locking/instability.
# 0.40 is still highly elastic but avoids this issue.
nu                 = 0.40

structure_thickness = structure_layers * particle_spacing
signed_distance_field = SignedDistanceField(aorta_geometry, particle_spacing;
                                            use_for_boundary_packing = true,
                                            max_signed_distance = structure_thickness)

structure = sample_boundary(signed_distance_field;
                            boundary_density  = structure_density,
                            boundary_thickness = structure_thickness,
                            place_on_shell    = true)

# Clamp particles close to the inlet and outlet rings (Dirichlet BC)
structure_coords = reinterpret(reshape,
                               SVector{NDIMS, eltype(structure.coordinates)},
                               structure.coordinates)

inlet_origin  = inlet_face_data.face[1]
outlet_origin = outlet_face_data.face[1]
clamp_width   = 2 * particle_spacing

inlet_clamped  = findall(x -> abs(dot(x - inlet_origin,  inlet_normal))  <= clamp_width,
                         structure_coords)
outlet_clamped = findall(x -> abs(dot(x - outlet_origin, outlet_normal)) <= clamp_width,
                         structure_coords)
clamped_particles = unique(vcat(inlet_clamped, outlet_clamped))

structure_smoothing_length = sqrt(2) * particle_spacing
structure_smoothing_kernel = WendlandC2Kernel{NDIMS}()

hydrodynamic_densities = fluid_density * ones(size(structure.density))
hydrodynamic_masses    = hydrodynamic_densities * particle_spacing^NDIMS

boundary_model_structure = BoundaryModelDummyParticles(hydrodynamic_densities,
                                                       hydrodynamic_masses,
                                                       AdamiPressureExtrapolation(),
                                                       state_equation = state_equation,
                                                       viscosity      = viscosity,
                                                       smoothing_kernel,
                                                       smoothing_length)

structure_system = TotalLagrangianSPHSystem(structure,
                                            structure_smoothing_kernel,
                                            structure_smoothing_length,
                                            E, nu,
                                            boundary_model    = boundary_model_structure,
                                            clamped_particles = clamped_particles,
                                            # Larger alpha damps hourglass oscillations in the elastic wall.
                                            penalty_force     = PenaltyForceGanzenmueller(alpha = 0.1))

# ==========================================================================================
# ==== Neighbourhood Search + Semidiscretization

all_coordinates = hcat(structure.coordinates, fluid.coordinates,
                       inflow.initial_condition.coordinates,
                       outflow.initial_condition.coordinates)

# Use a generous padding so that open-boundary buffer particles spawned during
# the pulsatile peak (higher velocity) and Windkessel back-pressure events
# never escape the fixed cell-list bounding box.
nhs_padding = (3 * open_boundary_layers + structure_layers + 8) * particle_spacing
min_corner  = minimum(all_coordinates .- nhs_padding, dims = 2)
max_corner  = maximum(all_coordinates .+ nhs_padding, dims = 2)

nhs = GridNeighborhoodSearch{NDIMS}(; cell_list       = FullGridCellList(; min_corner, max_corner),
                                    update_strategy = ParallelUpdate())

semi = Semidiscretization(fluid_system, open_boundary, structure_system,
                          neighborhood_search     = nhs,
                          parallelization_backend = PolyesterBackend())

ode = semidiscretize(semi, tspan)

# ==========================================================================================
# ==== Callbacks + Solve

info_callback   = InfoCallback(interval = 100)
saving_callback = SolutionSavingCallback(dt = 0.02, prefix = "aorta_pulsatile_wk")

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback())

# dtmax is set tight because the pulsatile peak speed is v_peak = 0.15 m/s.
# For longer production runs consider increasing particle_spacing and relaxing dtmax.
sol = solve(ode, RDPK3SpFSAL35(),
            abstol        = 1.0e-6,
            reltol        = 1.0e-4,
            dtmax         = 1.0e-5,
            save_everystep = false,
            callback      = callbacks)
