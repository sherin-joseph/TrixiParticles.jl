# ==========================================================================================
# 3D Pipe Flow in Aorta-Like STL Geometry with Elastic TLSPH Wall
#
# This example keeps the same flow/open-boundary setup as `pipe_flow_3d_aorta.jl`
# and replaces the rigid wall boundary by an elastic structure wall modeled with TLSPH.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq
using LinearAlgebra: dot, norm
using StaticArrays: SVector

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
tspan = (0.0, 0.5)

fluid_density = 1000.0
reynolds_number = 10
const prescribed_inlet_speed = 0.05
const inflow_ramp_time = 0.05

geometry_dir = @__DIR__
aorta_geometry = load_geometry(joinpath(geometry_dir, "arch_tube_closed_fixed_winding.stl"))
inlet_geometry = load_geometry(joinpath(geometry_dir, "inlet_disk.stl"))
outlet_geometry = load_geometry(joinpath(geometry_dir, "outlet_disk.stl"))

function orient_normal_towards_domain(face, face_normal, domain_center)
    face_center = (face[1] + face[2] + face[3]) / 3
    return dot(face_normal, domain_center - face_center) > 0 ? face_normal : -face_normal
end

inlet_face_data = planar_geometry_to_face(inlet_geometry)
outlet_face_data = planar_geometry_to_face(outlet_geometry)

domain_center = (aorta_geometry.min_corner + aorta_geometry.max_corner) / 2
inlet_normal = orient_normal_towards_domain(inlet_face_data.face, inlet_face_data.face_normal,
                                            domain_center)
outlet_normal = orient_normal_towards_domain(outlet_face_data.face,
                                             outlet_face_data.face_normal,
                                             domain_center)
inlet_velocity = prescribed_inlet_speed * inlet_normal

point_in_geometry_algorithm = WindingNumberJacobson(; geometry=aorta_geometry,
                                                    hierarchical_winding=true)

fluid = ComplexShape(aorta_geometry; particle_spacing, density=fluid_density,
                     velocity=(0.0, 0.0, 0.0), point_in_geometry_algorithm)

NDIMS = ndims(fluid)

face_extent_1 = norm(inlet_face_data.face[2] - inlet_face_data.face[1])
face_extent_2 = norm(inlet_face_data.face[3] - inlet_face_data.face[1])
n_face_particles = ceil(Int, face_extent_1 / particle_spacing) *
                   ceil(Int, face_extent_2 / particle_spacing)
n_buffer_particles = max(300, 10 * n_face_particles)

# ==========================================================================================
# ==== Fluid
wcsph = true

smoothing_length = 1.5 * particle_spacing
smoothing_kernel = WendlandC2Kernel{NDIMS}()

fluid_density_calculator = ContinuityDensity()

characteristic_length = maximum(aorta_geometry.max_corner - aorta_geometry.min_corner)
kinematic_viscosity = prescribed_inlet_speed * characteristic_length / reynolds_number

viscosity = ViscosityAdami(nu=kinematic_viscosity)
sound_speed = 20 * prescribed_inlet_speed

if wcsph
    state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                       exponent=1)
    density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.3)

    fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                               state_equation, smoothing_kernel,
                                               density_diffusion=density_diffusion,
                                               smoothing_length, viscosity=viscosity,
                                               shifting_technique=ParticleShiftingTechnique(v_max_factor=0.3),
                                               buffer_size=n_buffer_particles)
else
    state_equation = nothing

    fluid_system = EntropicallyDampedSPHSystem(fluid, smoothing_kernel,
                                               smoothing_length, sound_speed,
                                               viscosity=viscosity,
                                               density_calculator=fluid_density_calculator,
                                               shifting_technique=ParticleShiftingTechnique(),
                                               buffer_size=n_buffer_particles)
end

# ==========================================================================================
# ==== Open Boundary

function velocity_function3d(pos, t)
    ramp = min(t / inflow_ramp_time, 1.0)
    return ramp * inlet_velocity
end

open_boundary_model = BoundaryModelMirroringTafuni(; mirror_method=ZerothOrderMirroring())

reference_velocity_in = velocity_function3d
reference_pressure_in = nothing
reference_density_in = nothing
boundary_type_in = InFlow()
inflow = BoundaryZone(; boundary_face=inlet_face_data.face, face_normal=inlet_normal,
                      open_boundary_layers, density=fluid_density, particle_spacing,
                      reference_density=reference_density_in,
                      reference_pressure=reference_pressure_in,
                      reference_velocity=reference_velocity_in,
                      boundary_type=boundary_type_in)

reference_velocity_out = nothing
reference_pressure_out = 0.0
reference_density_out = nothing
boundary_type_out = OutFlow()
outflow = BoundaryZone(; boundary_face=outlet_face_data.face, face_normal=outlet_normal,
                       open_boundary_layers, density=fluid_density, particle_spacing,
                       reference_density=reference_density_out,
                       reference_pressure=reference_pressure_out,
                       reference_velocity=reference_velocity_out,
                       boundary_type=boundary_type_out)

open_boundary = OpenBoundarySystem(inflow, outflow; fluid_system,
                                   boundary_model=open_boundary_model,
                                   buffer_size=n_buffer_particles)

# ==========================================================================================
# ==== Elastic Structure (TLSPH wall)
structure_density = 1200.0
E = 1.0e6
nu = 0.45

structure_thickness = structure_layers * particle_spacing
signed_distance_field = SignedDistanceField(aorta_geometry, particle_spacing;
                                            use_for_boundary_packing=true,
                                            max_signed_distance=structure_thickness)

structure = sample_boundary(signed_distance_field; boundary_density=structure_density,
                            boundary_thickness=structure_thickness, place_on_shell=true)

# Clamp particles close to inlet and outlet sections
structure_coords = reinterpret(reshape,
                               SVector{NDIMS, eltype(structure.coordinates)},
                               structure.coordinates)

inlet_origin = inlet_face_data.face[1]
outlet_origin = outlet_face_data.face[1]
clamp_width = 2 * particle_spacing

inlet_clamped = findall(x -> abs(dot(x - inlet_origin, inlet_normal)) <= clamp_width,
                        structure_coords)
outlet_clamped = findall(x -> abs(dot(x - outlet_origin, outlet_normal)) <= clamp_width,
                         structure_coords)
clamped_particles = unique(vcat(inlet_clamped, outlet_clamped))

structure_smoothing_length = sqrt(2) * particle_spacing
structure_smoothing_kernel = WendlandC2Kernel{NDIMS}()

hydrodynamic_densities = fluid_density * ones(size(structure.density))
hydrodynamic_masses = hydrodynamic_densities * particle_spacing^NDIMS

boundary_model_structure = BoundaryModelDummyParticles(hydrodynamic_densities,
                                                       hydrodynamic_masses,
                                                       AdamiPressureExtrapolation(),
                                                       state_equation=state_equation,
                                                       viscosity=viscosity,
                                                       smoothing_kernel,
                                                       smoothing_length)

structure_system = TotalLagrangianSPHSystem(structure,
                                            structure_smoothing_kernel,
                                            structure_smoothing_length,
                                            E, nu,
                                            boundary_model=boundary_model_structure,
                                            clamped_particles=clamped_particles,
                                            penalty_force=PenaltyForceGanzenmueller(alpha=0.01))

# ==========================================================================================
# ==== Simulation
all_coordinates = hcat(structure.coordinates, fluid.coordinates,
                       inflow.initial_condition.coordinates,
                       outflow.initial_condition.coordinates)

nhs_padding = (open_boundary_layers + structure_layers + 4) * particle_spacing
min_corner = minimum(all_coordinates .- nhs_padding, dims=2)
max_corner = maximum(all_coordinates .+ nhs_padding, dims=2)

nhs = GridNeighborhoodSearch{NDIMS}(; cell_list=FullGridCellList(; min_corner, max_corner),
                                    update_strategy=ParallelUpdate())

semi = Semidiscretization(fluid_system, open_boundary, structure_system,
                          neighborhood_search=nhs,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.01, prefix="elastic_wall")

extra_callback = nothing

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback(), extra_callback)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-5,
            reltol=1e-3,
            dtmax=5e-5,
            save_everystep=false, callback=callbacks)
