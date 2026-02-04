# ==========================================================================================
# 2D Poiseuille Flow (Open boundaries) — unified kernel/h with Taylor–Green
# A–D variants + SGS + optional artificial viscosity + Postprocessing
# Methods: :WCSPH (default), optional :EDAC
# Fixed dt (same across variants)
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# -----------------------------------------------------------------------------
# USER CONTROLS
# -----------------------------------------------------------------------------
const VARIANT  = :A            # :A, :B, :C, :D
const METHOD   = :WCSPH        # :WCSPH or :EDAC
const SGS_KIND = :AdamiSGS     # :AdamiSGS or :MorrisSGS

const AV_MODE  = :OFF          # :OFF, :BASELINE_ONLY, :ALWAYS, :COMBINED
const AV_ALPHA = 0.02
const AV_BETA  = 0.0

# include("sgs_viscosity.jl")

# -----------------------------------------------------------------------------
# A–D variant logic (same as TGV)
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
# Combined viscosity wrapper
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
# ==== Geometry / resolution (same as your original Poiseuille)
const wall_distance = 0.001
const flow_length   = 0.004

particle_spacing = wall_distance / 50
boundary_layers = 4
open_boundary_layers = 10

domain_size = (flow_length, wall_distance)
open_boundary_size = (open_boundary_layers * particle_spacing, domain_size[2])

# ==========================================================================================
# ==== Experiment setup
tspan = (0.0, 2.0)

fluid_density = 1000.0
reynolds_number = 50
const pressure_drop = 0.1
pressure_out = 0.1
pressure_in  = pressure_out + pressure_drop

const dynamic_viscosity = sqrt(fluid_density * wall_distance^3 * pressure_drop /
                               (8 * flow_length * reynolds_number))

v_max = wall_distance^2 * pressure_drop / (8 * dynamic_viscosity * flow_length)

sound_speed_factor = 100
sound_speed = sound_speed_factor * v_max

flow_direction = (1.0, 0.0)

# ==========================================================================================
# ==== Particles / geometry
pipe = RectangularTank(particle_spacing, domain_size, domain_size, fluid_density,
                       pressure=(pos) -> pressure_out +
                                         pressure_drop * (1 - (pos[1] / flow_length)),
                       n_layers=boundary_layers, faces=(false, false, true, true),
                       coordinates_eltype=Float64)

inlet = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                        fluid_density, n_layers=boundary_layers, pressure=pressure_in,
                        min_coordinates=(0.0, 0.0), faces=(false, false, true, true),
                        coordinates_eltype=Float64)

outlet = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                         fluid_density, n_layers=boundary_layers,
                         min_coordinates=(pipe.fluid_size[1] - open_boundary_size[1], 0.0),
                         faces=(false, false, true, true),
                         coordinates_eltype=Float64)

fluid = setdiff(pipe.fluid, inlet.fluid, outlet.fluid)
n_buffer_particles = 10 * pipe.n_particles_per_dimension[2]^2

# ==========================================================================================
# ==== Unified kernel/h (match Taylor–Green)
h = 2 * particle_spacing
kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()
kinematic_viscosity = dynamic_viscosity / fluid_density

# -----------------------------------------------------------------------------
# Viscosity selection (physical vs SGS) + optional artificial viscosity
# -----------------------------------------------------------------------------
function base_viscosity(use_sgs::Bool)
    if !use_sgs
        return ViscosityAdami(nu=kinematic_viscosity)
    end
    if SGS_KIND == :AdamiSGS
        return ViscosityAdamiSGS(; nu=kinematic_viscosity, C_S=0.1, epsilon=0.001)
    elseif SGS_KIND == :MorrisSGS
        return ViscosityMorrisSGS(; nu=kinematic_viscosity, C_S=0.1, epsilon=0.001)
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
# Stabilization knobs (baseline vs reduced) — SAME definition as Taylor–Green
# baseline stabilization := shifting + density diffusion
# -----------------------------------------------------------------------------
background_pressure = 7 * sound_speed_factor / 10 * fluid_density * v_max^2

shifting_technique = vs.use_stabilization ?
    TransportVelocityAdami(; background_pressure) : nothing

density_diffusion = vs.use_stabilization ?
    DensityDiffusionMolteniColagrossi(delta=0.1) : nothing

# ==========================================================================================
# ==== Build fluid system
if METHOD == :WCSPH
    state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density, exponent=1)

    fluid_system = WeaklyCompressibleSPHSystem(
        fluid, fluid_density_calculator, state_equation, kernel, h;
        buffer_size = n_buffer_particles,
        shifting_technique = shifting_technique,
        density_diffusion  = density_diffusion,
        viscosity          = viscosity
    )

elseif METHOD == :EDAC
    state_equation = nothing

    fluid_system = EntropicallyDampedSPHSystem(
        fluid, kernel, h, sound_speed;
        viscosity = viscosity,
        density_calculator = fluid_density_calculator,
        shifting_technique = shifting_technique,
        buffer_size = n_buffer_particles
    )
else
    error("Unknown METHOD=$METHOD. Use :WCSPH or :EDAC")
end

# ==========================================================================================
# ==== Open Boundary (unchanged)
open_boundary_model = BoundaryModelDynamicalPressureZhang()

boundary_type_in = BidirectionalFlow()
face_in = ([open_boundary_size[1], 0.0], [open_boundary_size[1], pipe.fluid_size[2]])
reference_velocity_in = nothing
reference_pressure_in = 0.2

inflow = BoundaryZone(; boundary_face=face_in, face_normal=flow_direction,
                      open_boundary_layers, density=fluid_density, particle_spacing,
                      reference_velocity=reference_velocity_in,
                      reference_pressure=reference_pressure_in,
                      initial_condition=inlet.fluid, boundary_type=boundary_type_in)

boundary_type_out = BidirectionalFlow()
face_out = ([pipe.fluid_size[1] - open_boundary_size[1], 0.0],
            [pipe.fluid_size[1] - open_boundary_size[1], pipe.fluid_size[2]])
reference_velocity_out = nothing
reference_pressure_out = 0.1

outflow = BoundaryZone(; boundary_face=face_out, face_normal=(.-(flow_direction)),
                       open_boundary_layers, density=fluid_density, particle_spacing,
                       reference_velocity=reference_velocity_out,
                       reference_pressure=reference_pressure_out,
                       initial_condition=outlet.fluid, boundary_type=boundary_type_out)

open_boundary = OpenBoundarySystem(inflow, outflow; fluid_system,
                                   boundary_model=open_boundary_model,
                                   calculate_flow_rate=true,
                                   buffer_size=n_buffer_particles)

# ==========================================================================================
# ==== Wall boundary (unchanged)
wall = union(pipe.boundary)

boundary_model = BoundaryModelDummyParticles(wall.density, wall.mass,
                                             AdamiPressureExtrapolation(),
                                             kernel, h;
                                             state_equation=state_equation,
                                             viscosity=viscosity)

boundary_system = WallBoundarySystem(wall, boundary_model)

# ==========================================================================================
# ==== Simulation (unchanged neighborhood search)
min_corner = minimum(wall.coordinates .- 2 * particle_spacing, dims=2)
max_corner = maximum(wall.coordinates .+ 2 * particle_spacing, dims=2)

nhs = GridNeighborhoodSearch{2}(;
    cell_list=FullGridCellList(; min_corner, max_corner),
    update_strategy=ParallelUpdate()
)

semi = Semidiscretization(fluid_system, open_boundary, boundary_system;
                          neighborhood_search=nhs,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

# ==========================================================================================
# ==== Fixed dt (same recipe as Taylor–Green; then frozen)
cfl_c = 0.20
cfl_v = 0.10
dt_acoustic = cfl_c * h / (sound_speed + v_max)
dt_viscous  = cfl_v * h^2 / kinematic_viscosity
dt = min(dt_acoustic, dt_viscous)

# ==========================================================================================
# ==== Callbacks / solve (with postprocessing)
info_callback = InfoCallback(interval=100)

prefix = "poiseuille_$(String(METHOD))_$(String(VARIANT))_$(sgs_tag)"
saving_callback = SolutionSavingCallback(dt=0.02, prefix=prefix, output_directory="out")

# Postprocessing: kinetic energy + average pressure vs time (dt_pp = 0.01)
pp_filename = joinpath("out", prefix * "_pp")
pp_callback = PostprocessCallback(; dt=0.01,
                                  filename=pp_filename,
                                  kinetic_energy,
                                  avg_pressure,
                                  write_file_interval=0)

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback(), pp_callback)

sol = solve(ode, RDPK3SpFSAL35();
            dt=dt, adaptive=false,
            abstol=1e-6, reltol=1e-4,
            save_everystep=false, callback=callbacks)

@info "Done" METHOD VARIANT sgs_tag AV_MODE dt h kernel pp_filename
