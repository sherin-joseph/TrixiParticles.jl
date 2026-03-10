# ==========================================================================================
# 2D Taylor-Green Vortex (ABC Batch)
#
# Batch runner for A/B/C variant comparisons over multiple time horizons and
# VMS parameters.
#
# Variants per horizon:
#   A) Baseline Adami
#   B) Adami + classical SGS (ViscosityAdamiSGS)
#   C) Adami + VMSLES (sweep over C_s and min_shepard)
#
# Outputs under `output_root`:
# - `summary_runs.csv`      (one row per run)
# - `summary_comparison.csv` (C rows augmented with diffs vs A and B)
# - per-run VTK and KE time series
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# To be set via `trixi_include`
particle_spacing = 0.02
t_final_list = [0.1, 0.5, 1.0]
reynolds_number = 100.0
save_dt = 0.02
info_interval = 200
perturb_coordinates = false
output_root = "out/tgv_abc_batch"

# VMS sweep parameters
vms_C_s_list = [0.01, 0.03, 0.05, 0.08]
vms_min_shepard_list = [0.7, 0.9]
vms_epsilon = 0.01
vms_strain_mode = :fine

# Classical SGS parameter
sgs_C_S = 0.1
sgs_epsilon = 0.001

box_length = 1.0
U = 1.0
fluid_density = 1.0
sound_speed = 10U

nu = U * box_length / reynolds_number
b = -8pi^2 / reynolds_number

background_pressure = sound_speed^2 * fluid_density
smoothing_length = 1.0 * particle_spacing
smoothing_kernel = SchoenbergQuinticSplineKernel{2}()

function pressure_function(pos, t)
    x = pos[1]
    y = pos[2]
    return -U^2 * exp(2 * b * t) * (cos(4pi * x) + cos(4pi * y)) / 4
end

initial_pressure_function(pos) = pressure_function(pos, 0.0)

function velocity_function(pos, t)
    x = pos[1]
    y = pos[2]

    vel = U * exp(b * t) * [-cos(2pi * x) * sin(2pi * y),
                            sin(2pi * x) * cos(2pi * y)]

    return SVector{2}(vel)
end

initial_velocity_function(pos) = velocity_function(pos, 0.0)

function make_fluid_shape()
    n_particles_xy = round(Int, box_length / particle_spacing)

    return RectangularShape(particle_spacing,
                            (n_particles_xy, n_particles_xy),
                            (0.0, 0.0),
                            coordinates_perturbation=perturb_coordinates ? 0.2 : nothing,
                            density=fluid_density,
                            pressure=initial_pressure_function,
                            velocity=initial_velocity_function)
end

function make_wcsph_system(; viscosity, vms_les=nothing)
    fluid = make_fluid_shape()
    density_calculator = ContinuityDensity()
    state_equation = StateEquationCole(; sound_speed,
                                        reference_density=fluid_density,
                                        exponent=1)

    return WeaklyCompressibleSPHSystem(fluid,
                                       density_calculator,
                                       state_equation,
                                       smoothing_kernel,
                                       smoothing_length,
                                       pressure_acceleration=TrixiParticles.inter_particle_averaged_pressure,
                                       viscosity=viscosity,
                                       vms_les=vms_les,
                                       shifting_technique=TransportVelocityAdami(; background_pressure))
end

function make_semi(fluid_system)
    periodic_box = PeriodicBox(min_corner=[0.0, 0.0],
                               max_corner=[box_length, box_length])

    return Semidiscretization(fluid_system,
                              neighborhood_search=GridNeighborhoodSearch{2}(; periodic_box))
end

function kinetic_energy_from_v(v_state, mass)
    vel = view(v_state, 1:2, :)
    speed2 = vec(sum(abs2, vel; dims=1))
    return 0.5 * sum(mass .* speed2)
end

function write_ke_timeseries(path, sol, mass)
    open(path, "w") do io
        println(io, "t,kinetic_energy")
        for i in eachindex(sol.t)
            v_state = sol.u[i].x[2]
            ke = kinetic_energy_from_v(v_state, mass)
            println(io, "$(sol.t[i]),$(ke)")
        end
    end
end

function safe_tag(s)
    return replace(string(s), "." => "p", ":" => "")
end

function run_variant(name; t_final, viscosity, vms_les=nothing)
    output_directory = joinpath(output_root, "t$(safe_tag(t_final))", name)
    mkpath(output_directory)

    fluid_system = make_wcsph_system(; viscosity, vms_les)
    semi = make_semi(fluid_system)
    ode = semidiscretize(semi, (0.0, t_final))

    info_callback = InfoCallback(interval=info_interval)
    saving_callback = SolutionSavingCallback(dt=save_dt,
                                             output_directory=output_directory,
                                             save_initial_solution=true,
                                             save_final_solution=true)
    callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback())

    dt_max = min(smoothing_length / 4 * (sound_speed + U),
                 smoothing_length^2 / (8 * nu))

    runtime = @elapsed sol = solve(ode, RDPK3SpFSAL49();
                                   abstol=1e-8,
                                   reltol=1e-4,
                                   dtmax=dt_max,
                                   save_everystep=false,
                                   saveat=save_dt,
                                   callback=callbacks)

    u_end = sol.u[end].x[1]
    v_end = sol.u[end].x[2]
    ke_end = kinetic_energy_from_v(v_end, fluid_system.mass)

    write_ke_timeseries(joinpath(output_directory,
                                 "kinetic_energy_timeseries_$(name).csv"),
                        sol, fluid_system.mass)

    return (; name,
            t_final,
            retcode=sol.retcode,
            nsteps=length(sol.t),
            ke_final=ke_end,
            runtime,
            u_end,
            v_end,
            output_directory)
end

function max_abs_diff(a, b)
    return maximum(abs.(a .- b))
end

mkpath(output_root)

all_rows = NamedTuple[]
comparison_rows = NamedTuple[]

for t_final in t_final_list
    println("Running horizon t_final=$(t_final)")

    result_a = run_variant("A_baseline_adami";
                           t_final,
                           viscosity=ViscosityAdami(; nu),
                           vms_les=nothing)

    push!(all_rows, (; t_final,
                      variant=result_a.name,
                      mode="baseline",
                      C_s=missing,
                      min_shepard=missing,
                      retcode=result_a.retcode,
                      nsteps=result_a.nsteps,
                      ke_final=result_a.ke_final,
                      runtime_seconds=result_a.runtime,
                      output_directory=result_a.output_directory))

    result_b = run_variant("B_adami_sgs";
                           t_final,
                           viscosity=ViscosityAdamiSGS(; nu, C_S=sgs_C_S, epsilon=sgs_epsilon),
                           vms_les=nothing)

    push!(all_rows, (; t_final,
                      variant=result_b.name,
                      mode="sgs",
                      C_s=missing,
                      min_shepard=missing,
                      retcode=result_b.retcode,
                      nsteps=result_b.nsteps,
                      ke_final=result_b.ke_final,
                      runtime_seconds=result_b.runtime,
                      output_directory=result_b.output_directory))

    for C_s in vms_C_s_list, min_shepard in vms_min_shepard_list
        variant_name = "C_adami_vmsles_Cs$(safe_tag(C_s))_ms$(safe_tag(min_shepard))"

        result_c = run_variant(variant_name;
                               t_final,
                               viscosity=ViscosityAdami(; nu),
                               vms_les=VMSLES(C_s=C_s,
                                              epsilon=vms_epsilon,
                                              strain_mode=vms_strain_mode,
                                              min_shepard=min_shepard))

        push!(all_rows, (; t_final,
                          variant=result_c.name,
                          mode="vms",
                          C_s,
                          min_shepard,
                          retcode=result_c.retcode,
                          nsteps=result_c.nsteps,
                          ke_final=result_c.ke_final,
                          runtime_seconds=result_c.runtime,
                          output_directory=result_c.output_directory))

        push!(comparison_rows, (; t_final,
                                 variant=result_c.name,
                                 C_s,
                                 min_shepard,
                                 retcode=result_c.retcode,
                                 ke_final_A=result_a.ke_final,
                                 ke_final_B=result_b.ke_final,
                                 ke_final_C=result_c.ke_final,
                                 max_abs_diff_u_A_vs_C=max_abs_diff(result_a.u_end, result_c.u_end),
                                 max_abs_diff_v_A_vs_C=max_abs_diff(result_a.v_end, result_c.v_end),
                                 max_abs_diff_u_B_vs_C=max_abs_diff(result_b.u_end, result_c.u_end),
                                 max_abs_diff_v_B_vs_C=max_abs_diff(result_b.v_end, result_c.v_end),
                                 output_directory=result_c.output_directory))
    end
end

summary_runs_path = joinpath(output_root, "summary_runs.csv")
open(summary_runs_path, "w") do io
    println(io, "t_final,variant,mode,C_s,min_shepard,retcode,nsteps,ke_final,runtime_seconds,output_directory")
    for row in all_rows
        println(io, "$(row.t_final),$(row.variant),$(row.mode),$(row.C_s),$(row.min_shepard),$(row.retcode),$(row.nsteps),$(row.ke_final),$(row.runtime_seconds),$(row.output_directory)")
    end
end

summary_comparison_path = joinpath(output_root, "summary_comparison.csv")
open(summary_comparison_path, "w") do io
    println(io, "t_final,variant,C_s,min_shepard,retcode,ke_final_A,ke_final_B,ke_final_C,max_abs_diff_u_A_vs_C,max_abs_diff_v_A_vs_C,max_abs_diff_u_B_vs_C,max_abs_diff_v_B_vs_C,output_directory")
    for row in comparison_rows
        println(io, "$(row.t_final),$(row.variant),$(row.C_s),$(row.min_shepard),$(row.retcode),$(row.ke_final_A),$(row.ke_final_B),$(row.ke_final_C),$(row.max_abs_diff_u_A_vs_C),$(row.max_abs_diff_v_A_vs_C),$(row.max_abs_diff_u_B_vs_C),$(row.max_abs_diff_v_B_vs_C),$(row.output_directory)")
    end
end

println("ABC batch run completed.")
println("Run summary: $(summary_runs_path)")
println("Comparison summary: $(summary_comparison_path)")