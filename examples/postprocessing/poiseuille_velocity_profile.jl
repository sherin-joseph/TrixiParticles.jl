# ==========================================================================================
# POISEUILLE FLOW VELOCITY PROFILE COMPARISON
# ==========================================================================================
# Compares velocity profiles between Newtonian and Carreau-Yasuda (Non-Newtonian) fluids
# 
# Usage:
#   julia poiseuille_velocity_profile.jl
#
# Requirements: ReadVTK, DataFrames, Plots, Statistics
# ==========================================================================================

using ReadVTK
using Plots
using Statistics

# Configuration
const OUT_DIR = joinpath(@__DIR__, "..", "..", "out")
const PLOT_DIR = joinpath(OUT_DIR, "plots")
mkpath(PLOT_DIR)

# Simulation parameters (must match poiseuille_flow_2d.jl)
const wall_distance = 0.001
const flow_length = 0.05  # Updated to match corrected simulation
const particle_spacing = wall_distance / 50  # Updated to match corrected simulation
const fluid_density = 1000.0
const reynolds_number = 50
const pressure_drop = 0.1
const dynamic_viscosity = sqrt(fluid_density * wall_distance^3 * pressure_drop /
                               (8 * flow_length * reynolds_number))
const v_max = wall_distance^2 * pressure_drop / (8 * dynamic_viscosity * flow_length)

# Theoretical Poiseuille velocity profile (parabolic)
function theoretical_velocity(y, wall_dist, v_max_theory)
    # Normalized y position: y_norm = 0 at center, ±0.5 at walls
    y_center = wall_dist / 2
    y_norm = (y - y_center) / wall_dist
    
    # Parabolic profile: u(y) = u_max * (1 - (2*y_norm)^2)
    return v_max_theory * (1.0 - 4.0 * y_norm^2)
end

# Extract velocity profile from VTU file
function extract_velocity_profile(vtu_file, n_bins=20)
    println("  Reading: $vtu_file")
    
    if !isfile(vtu_file)
        @warn "File not found: $vtu_file"
        return nothing, nothing
    end
    
    # Read VTU file
    vtk = VTKFile(vtu_file)
    
    # Extract data
    points = get_points(vtk)
    
    # Try to get velocity data
    point_data = get_point_data(vtk)
    
    # Check available fields
    available_fields = keys(point_data)
    println("  Available fields: ", available_fields)
    
    # Check for velocity field (might be named "velocity", "v", or components "velocity_1", "velocity_2")
    velocity = nothing
    for name in ["velocity", "v", "Velocity"]
        if name in available_fields
            velocity = get_data(point_data[name])
            break
        end
    end
    
    if velocity === nothing
        @warn "No velocity data found in VTU file"
        # Try component-wise
        if "velocity_1" in available_fields && "velocity_2" in available_fields
            v1 = get_data(point_data["velocity_1"])
            v2 = get_data(point_data["velocity_2"])
            # Stack as 2xN matrix
            velocity = hcat(v1, v2)'
        else
            return nothing, nothing
        end
    end
    
    # Extract y-positions and x-velocities
    y_positions = Float64[]
    u_velocities = Float64[]
    
    n_points = size(points, 2)
    
    for i in 1:n_points
        y = points[2, i]
        
        # Get x-component of velocity
        u = 0.0
        if velocity isa AbstractMatrix
            # velocity is 2xN or Nx2
            if size(velocity, 1) == 2
                u = velocity[1, i]  # x-component
            elseif size(velocity, 2) == 2
                u = velocity[i, 1]  # x-component
            end
        elseif velocity isa AbstractVector
            # Try different access patterns
            try
                vel = velocity[i]
                u = typeof(vel) <: Number ? vel : vel[1]
            catch
                continue
            end
        else
            continue
        end
        
        # Filter particles in the main flow region (exclude boundaries)
        if y > particle_spacing && y < (wall_distance - particle_spacing)
            push!(y_positions, y)
            push!(u_velocities, u)
        end
    end
    
    if isempty(y_positions)
        @warn "No valid data points found"
        return nothing, nothing
    end
    
    # Bin the data
    y_min, y_max = extrema(y_positions)
    bin_edges = range(y_min, y_max, length=n_bins+1)
    bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in 1:n_bins]
    bin_velocities = zeros(n_bins)
    bin_counts = zeros(Int, n_bins)
    
    for (y, u) in zip(y_positions, u_velocities)
        bin_idx = searchsortedfirst(bin_edges, y) - 1
        bin_idx = clamp(bin_idx, 1, n_bins)
        bin_velocities[bin_idx] += u
        bin_counts[bin_idx] += 1
    end
    
    # Average velocities in each bin
    for i in 1:n_bins
        if bin_counts[i] > 0
            bin_velocities[i] /= bin_counts[i]
        end
    end
    
    # Filter out empty bins
    valid_bins = bin_counts .> 0
    
    return bin_centers[valid_bins], bin_velocities[valid_bins]
end

# Find latest timestep
function find_latest_timestep(prefix)
    pattern = Regex("$(prefix)_fluid_1_(\\d+)\\.vtu")
    max_step = -1
    
    for file in readdir(OUT_DIR)
        m = match(pattern, file)
        if m !== nothing
            step = parse(Int, m.captures[1])
            max_step = max(max_step, step)
        end
    end
    
    return max_step
end

# Main plotting function
function compare_velocity_profiles()
    println("\n" * "="^80)
    println("POISEUILLE FLOW VELOCITY PROFILE COMPARISON")
    println("="^80)
    
    # Find latest timesteps
    newtonian_step = find_latest_timestep("poiseuille_newtonian")
    carreau_step = find_latest_timestep("poiseuille_carreau_yasuda")
    
    if newtonian_step < 0 && carreau_step < 0
        @warn "No simulation data found"
        println("Expected files: poiseuille_newtonian_fluid_1_*.vtu or poiseuille_carreau_yasuda_fluid_1_*.vtu")
        return
    end
    
    println("\nFound timesteps:")
    println("  Newtonian: $newtonian_step")
    println("  Carreau-Yasuda: $carreau_step")
    
    # Extract profiles
    y_newt, u_newt = nothing, nothing
    y_carr, u_carr = nothing, nothing
    
    if newtonian_step >= 0
        vtu_file = joinpath(OUT_DIR, "poiseuille_newtonian_fluid_1_$(newtonian_step).vtu")
        y_newt, u_newt = extract_velocity_profile(vtu_file)
    end
    
    if carreau_step >= 0
        vtu_file = joinpath(OUT_DIR, "poiseuille_carreau_yasuda_fluid_1_$(carreau_step).vtu")
        y_carr, u_carr = extract_velocity_profile(vtu_file)
    end
    
    # Create theoretical profile
    y_theory = range(0, wall_distance, length=100)
    u_theory = [theoretical_velocity(y, wall_distance, v_max) for y in y_theory]
    
    # Plot velocity profiles
    p = plot(xlabel="y [m]", ylabel="u [m/s]",
             title="Poiseuille Flow: Velocity Profile Comparison",
             legend=:top,
             grid=true,
             size=(900, 700),
             dpi=150)
    
    # Theoretical profile
    plot!(p, y_theory, u_theory, 
          label="Theoretical (Newtonian)", 
          color=:black, 
          linestyle=:dash, 
          linewidth=2)
    
    # Newtonian simulation
    if y_newt !== nothing && u_newt !== nothing
        scatter!(p, y_newt, u_newt,
                label="Newtonian (SPH)",
                color=:blue,
                markersize=5,
                markerstrokewidth=0,
                alpha=0.7)
    end
    
    # Carreau-Yasuda simulation
    if y_carr !== nothing && u_carr !== nothing
        scatter!(p, y_carr, u_carr,
                label="Carreau-Yasuda (Non-Newtonian)",
                color=:red,
                markersize=5,
                markerstrokewidth=0,
                alpha=0.7)
    end
    
    savefig(p, joinpath(PLOT_DIR, "poiseuille_velocity_profile.png"))
    @info "Saved: poiseuille_velocity_profile.png"
    
    # Plot normalized velocity profiles
    p2 = plot(xlabel="y/H (normalized)", ylabel="u/u_max (normalized)",
              title="Poiseuille Flow: Normalized Velocity Profiles",
              legend=:top,
              grid=true,
              size=(900, 700),
              dpi=150)
    
    y_norm_theory = y_theory ./ wall_distance
    u_norm_theory = u_theory ./ v_max
    plot!(p2, y_norm_theory, u_norm_theory,
          label="Theoretical",
          color=:black,
          linestyle=:dash,
          linewidth=2)
    
    if y_newt !== nothing && u_newt !== nothing && !isempty(u_newt)
        u_max_newt = maximum(abs.(u_newt))
        scatter!(p2, y_newt ./ wall_distance, u_newt ./ u_max_newt,
                label="Newtonian",
                color=:blue,
                markersize=5,
                markerstrokewidth=0,
                alpha=0.7)
    end
    
    if y_carr !== nothing && u_carr !== nothing && !isempty(u_carr)
        u_max_carr = maximum(abs.(u_carr))
        scatter!(p2, y_carr ./ wall_distance, u_carr ./ u_max_carr,
                label="Carreau-Yasuda",
                color=:red,
                markersize=5,
                markerstrokewidth=0,
                alpha=0.7)
    end
    
    savefig(p2, joinpath(PLOT_DIR, "poiseuille_velocity_profile_normalized.png"))
    @info "Saved: poiseuille_velocity_profile_normalized.png"
    
    # Print statistics
    println("\n" * "-"^80)
    println("VELOCITY PROFILE STATISTICS")
    println("-"^80)
    
    if u_newt !== nothing && !isempty(u_newt)
        println("\nNewtonian:")
        println("  Max velocity: $(round(maximum(abs.(u_newt)), sigdigits=4)) m/s")
        println("  Theoretical max: $(round(v_max, sigdigits=4)) m/s")
        println("  Relative error: $(round(100*(maximum(abs.(u_newt))/v_max - 1), digits=2))%")
    end
    
    if u_carr !== nothing && !isempty(u_carr)
        println("\nCarreau-Yasuda (Non-Newtonian):")
        println("  Max velocity: $(round(maximum(abs.(u_carr)), sigdigits=4)) m/s")
        if u_newt !== nothing
            println("  Ratio to Newtonian: $(round(maximum(abs.(u_carr))/maximum(abs.(u_newt)), digits=3))")
            println("  → Shear-thinning effect: lower viscosity → $(maximum(abs.(u_carr)) > maximum(abs.(u_newt)) ? "higher" : "lower") flow rate")
        end
    end
    
    println("\n" * "="^80)
    println("✓ All plots saved to: $PLOT_DIR")
    println("="^80 * "\n")
end

# Run automatically
compare_velocity_profiles()
