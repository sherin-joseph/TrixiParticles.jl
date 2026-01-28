using TrixiParticles
using OrdinaryDiffEq
using StaticArrays
using LinearAlgebra

# ==========================================================================================
# ==== FILES
# ==========================================================================================
const vessel_stl = joinpath(@__DIR__, "aorta", "arch_tube_closed_fixed_winding.stl")
const inlet_stl  = joinpath(@__DIR__, "aorta", "inlet_plane_fixed.stl")
const outlet_stl = joinpath(@__DIR__, "aorta", "outlet_plane_fixed.stl")

# ==========================================================================================
# ==== SETTINGS
# ==========================================================================================
const NDIMS = 3
particle_spacing      = 0.005
boundary_layers       = 3
open_boundary_layers  = 4

tspan         = (0.0, 7.0)
save_interval = 0.05

REMOVE_ENDCAPS = true

# Outlet Windkessel
USE_WINDKESSEL = true
R1 = 2.0e6
R2 = 8.0e6
C  = 2.0e-9

# Disc shrink (buffer points back from wall)
DISC_SHRINK = 2.5

# Open boundary model (robust default for complex/FSI cases)
USE_DYNAMICAL_PRESSURE_BOUNDARY = true

# ==========================================================================================
# ==== FLUID PARAMETERS (WCSPH)
# ==========================================================================================
ρ_fluid  = 1050.0
ν_fluid  = 1.0e-5
c0_fluid = 25.0
p_bg     = 0.0
U_mean   = 0.10

# ==========================================================================================
# ==== SOLID PARAMETERS (TLSPH)
# ==========================================================================================
ρ_wall = 1100.0
E_wall = 5.0e5
ν_wall = 0.45

# ==========================================================================================
# ==== HELPERS
# ==========================================================================================
toS3(x) = SVector{3}(x[1], x[2], x[3])

function face_center(face)
    A = toS3(face[1]); B = toS3(face[2]); C = toS3(face[3])
    A + 0.5*(B-A) + 0.5*(C-A)
end

function face_basis(face)
    A = toS3(face[1]); B = toS3(face[2]); C = toS3(face[3])
    t1 = B - A
    t2 = C - A
    L1 = norm(t1); L2 = norm(t2)
    t1 = t1 / L1
    t2 = t2 - dot(t2, t1)*t1
    t2 = t2 / norm(t2)
    return t1, t2, L1, L2
end

function estimate_radius_from_fluid(fluid_ic, cin, axis; s_min, s_max)
    X = fluid_ic.coordinates
    rs = Float64[]
    for i in 1:size(X, 2)
        x = SVector{3}(X[:,i]...)
        s = dot(x - cin, axis)
        if s_min <= s <= s_max
            r = norm((x - cin) - s*axis)
            push!(rs, r)
        end
    end
    if length(rs) < 50
        return NaN
    end
    sort!(rs)
    return rs[Int(ceil(0.995*length(rs)))]  # robust max radius
end

function disc_on_face(face; dx, R, shrink=2.5)
    t1, t2, _, _ = face_basis(face)
    c = face_center(face)
    r_eff = max(R - shrink*dx, 2dx)
    us = collect(-r_eff:dx:r_eff)
    pts = SVector{3,Float64}[]
    for u in us, v in us
        if u*u + v*v <= r_eff*r_eff
            push!(pts, c + u*t1 + v*t2)
        end
    end
    X = reduce(hcat, pts)
    return InitialCondition(; coordinates=X, density=ρ_fluid, particle_spacing=dx), r_eff
end

function parabolic_inflow(face, n̂, R; Umean)
    n = normalize(toS3(n̂))
    t1, t2, _, _ = face_basis(face)
    c = face_center(face)
    Umax = 2.0*Umean
    return function (x, t)
        ξ = toS3(x) - c
        r = sqrt((dot(ξ, t1))^2 + (dot(ξ, t2))^2)
        s = clamp(r/R, 0.0, 1.0)
        (Umax*(1 - s*s)) * n
    end
end

function remove_endcaps(solid_ic, cin, cout, axis, r_keep; dx, tol=1.1)
    X = solid_ic.coordinates
    keep = trues(size(X,2))
    L = norm(cout - cin)
    for i in 1:size(X,2)
        x = SVector{3}(X[:,i]...)
        s = dot(x - cin, axis)
        r = norm((x - cin) - s*axis)
        if (abs(s) <= tol*dx && r <= r_keep) || (abs(s-L) <= tol*dx && r <= r_keep)
            keep[i] = false
        end
    end
    return InitialCondition(; coordinates=X[:,keep],
        density=solid_ic.density[keep],
        mass=solid_ic.mass[keep],
        velocity=solid_ic.velocity[:,keep],
        particle_spacing=solid_ic.particle_spacing
    )
end

# ------------------------------------------------------------------------------------------
# FIXED: remove SOLID particles ONLY in the OUTSIDE (extruded) open-boundary slab (one-sided)
# ------------------------------------------------------------------------------------------
function remove_solid_in_open_boundary_regions(
    solid_ic, inlet_face, inlet_n, outlet_face, outlet_n;
    dx, open_boundary_layers, R_in_eff, R_out_eff,
    slab_factor=1.25, radial_margin=0.75
)
    X = solid_ic.coordinates
    keep = trues(size(X, 2))

    function carve!(face, n_inside, R_eff)
        t1, t2, _, _ = face_basis(face)
        c = face_center(face)
        n = normalize(toS3(n_inside))  # MUST point INSIDE the fluid
        slab = slab_factor * open_boundary_layers * dx
        r_cut = R_eff + radial_margin * dx

        for i in 1:size(X,2)
            keep[i] || continue
            x = SVector{3}(X[:,i]...)
            ξ = x - c

            d = dot(ξ, n)  # signed distance along inside normal
            # Open boundary is extruded opposite n (OUTSIDE): that is d in [-slab, 0]
            if (-slab <= d <= 0.0)
                r = sqrt((dot(ξ, t1))^2 + (dot(ξ, t2))^2)
                if r <= r_cut
                    keep[i] = false
                end
            end
        end
    end

    carve!(inlet_face,  inlet_n,  R_in_eff)
    carve!(outlet_face, outlet_n, R_out_eff)

    return InitialCondition(; coordinates=X[:,keep],
        density=solid_ic.density[keep],
        mass=solid_ic.mass[keep],
        velocity=solid_ic.velocity[:,keep],
        particle_spacing=solid_ic.particle_spacing
    )
end

# ==========================================================================================
# ==== LOAD GEOMETRY
# ==========================================================================================
vessel_geo = load_geometry(vessel_stl)
inlet_geo  = load_geometry(inlet_stl)
outlet_geo = load_geometry(outlet_stl)

inlet_face, inlet_n0   = TrixiParticles.planar_geometry_to_face(inlet_geo)
outlet_face, outlet_n0 = TrixiParticles.planar_geometry_to_face(outlet_geo)

cin  = face_center(inlet_face)
cout = face_center(outlet_face)
axis = normalize(cout - cin)
Laxial = norm(cout - cin)

# Make face normals point INSIDE the fluid:
inlet_n  = normalize(toS3(inlet_n0))
outlet_n = normalize(toS3(outlet_n0))

if dot(inlet_n, axis) < 0
    inlet_n = -inlet_n
end
if dot(outlet_n, axis) > 0
    outlet_n = -outlet_n
end

# ==========================================================================================
# ==== SDF + INITIAL PARTICLES
# ==========================================================================================
boundary_thickness = boundary_layers * particle_spacing
sdf = SignedDistanceField(vessel_geo, particle_spacing;
                          use_for_boundary_packing=true,
                          max_signed_distance=boundary_thickness)

fluid_ic0 = ComplexShape(vessel_geo;
    particle_spacing,
    density=ρ_fluid,
    point_in_geometry_algorithm=WindingNumberJacobson(; geometry=vessel_geo),
    pad_initial_particle_grid=2*particle_spacing)

solid_ic0 = sample_boundary(sdf;
    boundary_density=ρ_wall,
    boundary_thickness=boundary_thickness,
    place_on_shell=true)

# ==========================================================================================
# ==== PACKING
# ==========================================================================================
packing_kernel = SchoenbergQuinticSplineKernel{NDIMS}()
packing_h = 0.8*particle_spacing
packing_bg_p = 1.0

packing_fluid = ParticlePackingSystem(fluid_ic0;
    smoothing_kernel=packing_kernel,
    smoothing_length=packing_h,
    signed_distance_field=sdf,
    background_pressure=packing_bg_p)

packing_solid = ParticlePackingSystem(solid_ic0;
    is_boundary=true,
    smoothing_kernel=packing_kernel,
    smoothing_length=packing_h,
    signed_distance_field=sdf,
    background_pressure=packing_bg_p,
    boundary_compress_factor=0.9)

semi_pack = Semidiscretization(packing_fluid, packing_solid;
    neighborhood_search=TrivialNeighborhoodSearch{NDIMS}())

ode_pack = semidiscretize(semi_pack, (0.0, 1.5))
sol_pack = solve(ode_pack, RDPK3SpFSAL35(); save_everystep=false,
    callback=CallbackSet(InfoCallback(interval=100), UpdateCallback()))

fluid_ic = InitialCondition(sol_pack, packing_fluid, semi_pack)
solid_ic = InitialCondition(sol_pack, packing_solid, semi_pack)

# ==========================================================================================
# ==== TRUE OPENING RADIUS
# ==========================================================================================
window = 4.0*particle_spacing
R_in  = estimate_radius_from_fluid(fluid_ic, cin, axis; s_min=0.0, s_max=window)
R_out = estimate_radius_from_fluid(fluid_ic, cin, axis; s_min=Laxial-window, s_max=Laxial)

@show R_in R_out

r_keep = max(min(R_in, R_out) - 2particle_spacing, 2particle_spacing)

if REMOVE_ENDCAPS
    solid_ic = remove_endcaps(solid_ic, cin, cout, axis, r_keep; dx=particle_spacing)
end

# ==========================================================================================
# ==== SYSTEMS
# ==========================================================================================
smoothing_kernel = WendlandC2Kernel{NDIMS}()
smoothing_length = 1.5*particle_spacing

density_calculator = ContinuityDensity()
state_equation = StateEquationCole(; sound_speed=c0_fluid,
    reference_density=ρ_fluid, exponent=1, background_pressure=p_bg)

viscosity = ViscosityAdami(nu=ν_fluid)
density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)

# Fluid buffer "discs" (also used as sample points for Windkessel Q integration)
inlet_disc,  R_in_eff  = disc_on_face(inlet_face;  dx=particle_spacing, R=R_in,  shrink=DISC_SHRINK)
outlet_disc, R_out_eff = disc_on_face(outlet_face; dx=particle_spacing, R=R_out, shrink=DISC_SHRINK)

# IMPORTANT: ensure NO solid particles exist in the OUTSIDE open-boundary slabs
solid_ic = remove_solid_in_open_boundary_regions(
    solid_ic, inlet_face, inlet_n, outlet_face, outlet_n;
    dx=particle_spacing,
    open_boundary_layers=open_boundary_layers,
    R_in_eff=R_in_eff,
    R_out_eff=R_out_eff
)

inlet_buf  = ceil(Int, 5.0*size(inlet_disc.coordinates,2)*open_boundary_layers) + 5000
outlet_buf = ceil(Int, 5.0*size(outlet_disc.coordinates,2)*open_boundary_layers) + 5000

fluid_system = WeaklyCompressibleSPHSystem(
    fluid_ic, density_calculator, state_equation,
    smoothing_kernel, smoothing_length;
    viscosity, density_diffusion,
    buffer_size=inlet_buf + outlet_buf + 10000
)

hydro_mass    = solid_ic.mass .* (ρ_fluid / ρ_wall)
hydro_density = fill(ρ_fluid, length(hydro_mass))

fsi_boundary_model = BoundaryModelDummyParticles(
    hydro_density, hydro_mass,
    BernoulliPressureExtrapolation(),
    smoothing_kernel, smoothing_length;
    state_equation, viscosity
)

# Clamp end rings
clamped = Int[]
clamp_len = 3particle_spacing
coordsS = solid_ic.coordinates
for i in 1:size(coordsS,2)
    x = SVector{3}(coordsS[:,i]...)
    s = dot(x - cin, axis)
    r = norm((x - cin) - s*axis)
    if (s <= clamp_len || s >= (Laxial - clamp_len)) && (r >= r_keep)
        push!(clamped, i)
    end
end

solid_system = TotalLagrangianSPHSystem(
    solid_ic, smoothing_kernel, smoothing_length,
    E_wall, ν_wall;
    clamped_particles=clamped,
    penalty_force=PenaltyForceGanzenmueller(alpha=0.1),
    boundary_model=fsi_boundary_model
)

# ==========================================================================================
# ==== OPEN BOUNDARIES
# ==========================================================================================
inflow_func = parabolic_inflow(inlet_face, inlet_n, R_in_eff; Umean=U_mean)

outlet_pressure = USE_WINDKESSEL ?
    RCRWindkesselModel(; characteristic_resistance=R1,
                        peripheral_resistance=R2,
                        compliance=C) : p_bg

# CRITICAL FIX:
# - use sample_points matching the *disc* (circular opening)
# - extrude_geometry stays as disc too
inlet_zone = BoundaryZone(;
    boundary_face=inlet_face,
    face_normal=inlet_n,               # points INSIDE
    density=ρ_fluid,
    particle_spacing,
    open_boundary_layers,
    boundary_type=InFlow(),
    reference_velocity=inflow_func,
    reference_pressure=p_bg,
    reference_density=ρ_fluid,
    extrude_geometry=inlet_disc,
    sample_points=inlet_disc.coordinates
)

outlet_zone = BoundaryZone(;
    boundary_face=outlet_face,
    face_normal=outlet_n,              # points INSIDE
    density=ρ_fluid,
    particle_spacing,
    open_boundary_layers,
    boundary_type=OutFlow(),
    reference_pressure=outlet_pressure,
    reference_density=ρ_fluid,
    extrude_geometry=outlet_disc,
    sample_points=outlet_disc.coordinates
)

open_boundary_model = USE_DYNAMICAL_PRESSURE_BOUNDARY ?
    BoundaryModelDynamicalPressureZhang() :
    BoundaryModelMirroringTafuni()

open_inlet = OpenBoundarySystem(inlet_zone;
    fluid_system=fluid_system,
    boundary_model=open_boundary_model,
    buffer_size=inlet_buf
)

open_outlet = OpenBoundarySystem(outlet_zone;
    fluid_system=fluid_system,
    boundary_model=open_boundary_model,
    buffer_size=outlet_buf
)

systems = [fluid_system, solid_system, open_inlet, open_outlet]

# ==========================================================================================
# ==== NEIGHBORHOOD SEARCH
# ==========================================================================================
all_coords = hcat(fluid_ic.coordinates, solid_ic.coordinates)
domain_min = minimum(all_coords, dims=2)
domain_max = maximum(all_coords, dims=2)

margin = 30*smoothing_length
min_corner = domain_min .- margin
max_corner = domain_max .+ margin

nhs = GridNeighborhoodSearch{NDIMS}(;
    cell_list=FullGridCellList(; min_corner, max_corner),
    update_strategy=ParallelUpdate()
)

# ==========================================================================================
# ==== SEMI + SOLVE
# ==========================================================================================
semi = Semidiscretization(systems...;
    neighborhood_search=nhs,
    parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

callbacks = CallbackSet(
    InfoCallback(interval=50),
    SolutionSavingCallback(dt=save_interval, prefix="tube_open_fixed_"),
    UpdateCallback()
)

sol = solve(ode, RDPK3SpFSAL35();
    abstol=1e-4, reltol=1e-2,
    dt=2e-5, dtmax=1e-4, adaptive=true,
    save_everystep=false, callback=callbacks)
