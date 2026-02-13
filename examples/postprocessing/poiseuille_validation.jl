# ==========================================================================================
# POISEUILLE FLOW COMPREHENSIVE VALIDATION
# ==========================================================================================
# Detailed validation analysis for Newtonian and Carreau-Yasuda implementations
# 
# Usage:
#   julia poiseuille_validation.jl
#
# Requirements: ReadVTK, DataFrames, Plots, Statistics
# ==========================================================================================

using ReadVTK
using Plots
using Statistics
using Printf

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
const v_max_theory = wall_distance^2 * pressure_drop / (8 * dynamic_viscosity * flow_length)

# Theoretical Poiseuille velocity profile (parabolic)
function theoretical_velocity(y, wall_dist, v_max)
    y_center = wall_dist / 2
    y_norm = (y - y_center) / wall_dist
    return v_max * (1.0 - 4.0 * y_norm^2)
end

# Extract comprehensive data from VTU file
function extract_flow_data(vtu_file, n_bins=20)
    println("  Reading: $(basename(vtu_file))")
    
    if !isfile(vtu_file)
        @warn "File not found: $vtu_file"
        return nothing
    end
    
    vtk = VTKFile(vtu_file)
    points = get_points(vtk)
    point_data = get_point_data(vtk)
    
    available_fields = keys(point_data)
    
    # Extract velocity
    velocity = nothing
    if "velocity" in available_fields
        velocity = get_data(point_data["velocity"])
    end
    
    # Extract pressure
    pressure = nothing
    if "pressure" in available_fields
        pressure = get_data(point_data["pressure"])
    end
    
    # Extract density
    density_field = nothing
    if "density" in available_fields
        density_field = get_data(point_data["density"])
    end
    
    if velocity === nothing
        @warn "No velocity data found"
        return nothing
    end
    
    # Collect particle data
    y_positions = Float64[]
    x_positions = Float64[]
    u_velocities = Float64[]
    v_velocities = Float64[]
    pressures = Float64[]
    densities = Float64[]
    
    n_points = size(points, 2)
    
    for i in 1:n_points
        x = points[1, i]
        y = points[2, i]
        
        # Get velocity components
        u, v = 0.0, 0.0
        if velocity isa AbstractMatrix
            if size(velocity, 1) == 2
                u, v = velocity[1, i], velocity[2, i]
            elseif size(velocity, 2) == 2
                u, v = velocity[i, 1], velocity[i, 2]
            end
        end
        
        # Get pressure
        p = pressure !== nothing ? pressure[i] : 0.0
        
        # Get density
        rho = density_field !== nothing ? density_field[i] : fluid_density
        
        # Filter particles in main flow region
        if y > particle_spacing && y < (wall_distance - particle_spacing) &&
           x > 0.003 && x < (flow_length - 0.003)  # Central region
            push!(x_positions, x)
            push!(y_positions, y)
            push!(u_velocities, u)
            push!(v_velocities, v)
            push!(pressures, p)
            push!(densities, rho)
        end
    end
    
    if isempty(y_positions)
        @warn "No valid data points found"
        return nothing
    end
    
    # Bin the data by y-position
    y_min, y_max = extrema(y_positions)
    bin_edges = range(y_min, y_max, length=n_bins+1)
    bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in 1:n_bins]
    
    bin_u = zeros(n_bins)
    bin_v = zeros(n_bins)
    bin_p = zeros(n_bins)
    bin_rho = zeros(n_bins)
    bin_counts = zeros(Int, n_bins)
    
    for i in 1:length(y_positions)
        bin_idx = searchsortedfirst(bin_edges, y_positions[i]) - 1
        bin_idx = clamp(bin_idx, 1, n_bins)
        bin_u[bin_idx] += u_velocities[i]
        bin_v[bin_idx] += v_velocities[i]
        bin_p[bin_idx] += pressures[i]
        bin_rho[bin_idx] += densities[i]
        bin_counts[bin_idx] += 1
    end
    
    # Average
    for i in 1:n_bins
        if bin_counts[i] > 0
            bin_u[i] /= bin_counts[i]
            bin_v[i] /= bin_counts[i]
            bin_p[i] /= bin_counts[i]
            bin_rho[i] /= bin_counts[i]
        end
    end
    
    valid_bins = bin_counts .> 0
    
    return Dict(
        "y" => bin_centers[valid_bins],
        "u" => bin_u[valid_bins],
        "v" => bin_v[valid_bins],
        "p" => bin_p[valid_bins],
        "rho" => bin_rho[valid_bins],
        "u_raw" => u_velocities,
        "v_raw" => v_velocities,
        "p_raw" => pressures
    )
end

# Calculate flow rate
function calculate_flow_rate(y, u, width=1.0)
    # Numerical integration using trapezoidal rule
    if length(y) < 2
        return 0.0
    end
    
    Q = 0.0
    for i in 1:(length(y)-1)
        dy = y[i+1] - y[i]
        Q += 0.5 * (u[i] + u[i+1]) * dy
    end
    return Q * width  # Multiply by unit width
end

# Calculate metrics
function calculate_metrics(data, name)
    if data === nothing
        return nothing
    end
    
    y = data["y"]
    u = data["u"]
    v = data["v"]
    p = data["p"]
    
    # Max velocity
    u_max = maximum(abs.(u))
    u_max_theory = v_max_theory
    
    # Flow rate
    Q_sim = calculate_flow_rate(y, u)
    Q_theory = 2.0 / 3.0 * u_max_theory * wall_distance  # Theoretical flow rate per unit width
    
    # Mean velocity
    u_mean = mean(u)
    
    # Cross-flow velocity (should be ~0)
    v_mean = mean(abs.(v))
    v_max = maximum(abs.(v))
    
    # Pressure statistics
    p_mean = mean(p)
    p_std = std(p)
    
    # R² for profile shape
    y_theory = y
    u_theory = [theoretical_velocity(yi, wall_distance, v_max_theory) for yi in y_theory]
    
    # Calculate R² (coefficient of determination)
    ss_res = sum((u .- u_theory).^2)
    ss_tot = sum((u .- mean(u)).^2)
    r_squared = 1.0 - ss_res/ss_tot
    
    # RMS error
    rmse = sqrt(mean((u .- u_theory).^2))
    
    return Dict(
        "name" => name,
        "u_max" => u_max,
        "u_max_error%" => 100 * (u_max / u_max_theory - 1),
        "Q" => Q_sim,
        "Q_theory" => Q_theory,
        "Q_error%" => 100 * (Q_sim / Q_theory - 1),
        "u_mean" => u_mean,
        "v_mean" => v_mean,
        "v_max" => v_max,
        "p_mean" => p_mean,
        "p_std" => p_std,
        "R²" => r_squared,
        "RMSE" => rmse,
        "RMSE%" => 100 * rmse / u_max_theory
    )
end

# Find all timesteps
function find_all_timesteps(prefix)
    pattern = Regex("$(prefix)_fluid_1_(\\d+)\\.vtu")
    timesteps = Int[]
    
    for file in readdir(OUT_DIR)
        m = match(pattern, file)
        if m !== nothing
            step = parse(Int, m.captures[1])
            push!(timesteps, step)
        end
    end
    
    return sort(timesteps)
end

# Main validation function
function comprehensive_validation()
    println("\n" * "="^80)
    println("POISEUILLE FLOW COMPREHENSIVE VALIDATION")
    println("="^80)
    
    # Find timesteps
    newtonian_steps = find_all_timesteps("poiseuille_newtonian")
    carreau_steps = find_all_timesteps("poiseuille_carreau_yasuda")
    
    if isempty(newtonian_steps) && isempty(carreau_steps)
        @warn "No simulation data found"
        return
    end
    
    println("\n" * "="^80)
    println("1. STEADY STATE ANALYSIS")
    println("="^80)
    
    # Check convergence by looking at last few timesteps
    if !isempty(newtonian_steps) && length(newtonian_steps) >= 3
        println("\nNewtonian - Last 3 timesteps:")
        last_steps = newtonian_steps[max(1, end-2):end]
        u_max_vals = Float64[]
        
        for step in last_steps
            vtu_file = joinpath(OUT_DIR, "poiseuille_newtonian_fluid_1_$(step).vtu")
            data = extract_flow_data(vtu_file, 20)
            if data !== nothing
                u_max = maximum(abs.(data["u"]))
                push!(u_max_vals, u_max)
                @printf("  Step %2d: u_max = %.6f m/s\n", step, u_max)
            end
        end
        
        if length(u_max_vals) >= 2
            variation = std(u_max_vals) / mean(u_max_vals) * 100
            @printf("  Variation (CV): %.2f%%\n", variation)
            if variation < 5.0
                println("  ✓ Flow appears to be at steady state (CV < 5%)")
            else
                println("  ⚠ Flow may not be fully developed (CV > 5%)")
            end
        end
    end
    
    if !isempty(carreau_steps) && length(carreau_steps) >= 3
        println("\nCarreau-Yasuda - Last 3 timesteps:")
        last_steps = carreau_steps[max(1, end-2):end]
        u_max_vals = Float64[]
        
        for step in last_steps
            vtu_file = joinpath(OUT_DIR, "poiseuille_carreau_yasuda_fluid_1_$(step).vtu")
            data = extract_flow_data(vtu_file, 20)
            if data !== nothing
                u_max = maximum(abs.(data["u"]))
                push!(u_max_vals, u_max)
                @printf("  Step %2d: u_max = %.6f m/s\n", step, u_max)
            end
        end
        
        if length(u_max_vals) >= 2
            variation = std(u_max_vals) / mean(u_max_vals) * 100
            @printf("  Variation (CV): %.2f%%\n", variation)
            if variation < 5.0
                println("  ✓ Flow appears to be at steady state")
            else
                println("  ⚠ Flow may not be fully developed")
            end
        end
    end
    
    # Use latest timestep for detailed analysis
    newtonian_step = isempty(newtonian_steps) ? -1 : newtonian_steps[end]
    carreau_step = isempty(carreau_steps) ? -1 : carreau_steps[end]
    
    println("\n" * "="^80)
    println("2. DETAILED METRICS (Latest Timestep)")
    println("="^80)
    
    # Extract data
    newt_data = nothing
    carr_data = nothing
    
    if newtonian_step >= 0
        vtu_file = joinpath(OUT_DIR, "poiseuille_newtonian_fluid_1_$(newtonian_step).vtu")
        newt_data = extract_flow_data(vtu_file)
    end
    
    if carreau_step >= 0
        vtu_file = joinpath(OUT_DIR, "poiseuille_carreau_yasuda_fluid_1_$(carreau_step).vtu")
        carr_data = extract_flow_data(vtu_file)
    end
    
    # Calculate metrics
    newt_metrics = calculate_metrics(newt_data, "Newtonian")
    carr_metrics = calculate_metrics(carr_data, "Carreau-Yasuda")
    
    # Print metrics table
    @printf("\n%-30s %15s %15s %15s\n", "Metric", "Newtonian", "Carreau-Yasuda", "Theory")
    println("-"^80)
    @printf("%-30s %15.6f %15.6f %15.6f\n", "Max velocity (m/s)", 
            newt_metrics !== nothing ? newt_metrics["u_max"] : NaN,
            carr_metrics !== nothing ? carr_metrics["u_max"] : NaN,
            v_max_theory)
    @printf("%-30s %14.2f%% %14.2f%% %15s\n", "  Error vs theory",
            newt_metrics !== nothing ? newt_metrics["u_max_error%"] : NaN,
            carr_metrics !== nothing ? carr_metrics["u_max_error%"] : NaN,
            "-")
    println()
    @printf("%-30s %15.8f %15.8f %15.8f\n", "Flow rate (m²/s)",
            newt_metrics !== nothing ? newt_metrics["Q"] : NaN,
            carr_metrics !== nothing ? carr_metrics["Q"] : NaN,
            newt_metrics !== nothing ? newt_metrics["Q_theory"] : NaN)
    @printf("%-30s %14.2f%% %14.2f%% %15s\n", "  Error vs theory",
            newt_metrics !== nothing ? newt_metrics["Q_error%"] : NaN,
            carr_metrics !== nothing ? carr_metrics["Q_error%"] : NaN,
            "-")
    println()
    @printf("%-30s %15.6f %15.6f %15s\n", "Mean velocity (m/s)",
            newt_metrics !== nothing ? newt_metrics["u_mean"] : NaN,
            carr_metrics !== nothing ? carr_metrics["u_mean"] : NaN,
            "-")
    @printf("%-30s %15.6f %15.6f %15s\n", "Cross-flow |v| mean (m/s)",
            newt_metrics !== nothing ? newt_metrics["v_mean"] : NaN,
            carr_metrics !== nothing ? carr_metrics["v_mean"] : NaN,
            "~0")
    @printf("%-30s %15.6f %15.6f %15s\n", "Cross-flow |v| max (m/s)",
            newt_metrics !== nothing ? newt_metrics["v_max"] : NaN,
            carr_metrics !== nothing ? carr_metrics["v_max"] : NaN,
            "~0")
    println()
    @printf("%-30s %15.4f %15.4f %15s\n", "R² (profile shape)",
            newt_metrics !== nothing ? newt_metrics["R²"] : NaN,
            carr_metrics !== nothing ? carr_metrics["R²"] : NaN,
            "1.0")
    @printf("%-30s %15.6f %15.6f %15s\n", "RMSE (m/s)",
            newt_metrics !== nothing ? newt_metrics["RMSE"] : NaN,
            carr_metrics !== nothing ? carr_metrics["RMSE"] : NaN,
            "0")
    
    println("\n" * "="^80)
    println("3. VALIDATION ASSESSMENT")
    println("="^80)
    
    println("\n📊 NEWTONIAN VALIDATION:")
    if newt_metrics !== nothing
        u_err = abs(newt_metrics["u_max_error%"])
        q_err = abs(newt_metrics["Q_error%"])
        r2 = newt_metrics["R²"]
        
        if u_err < 10.0 && q_err < 15.0 && r2 > 0.95
            println("  ✅ VALIDATED - Good agreement with theory")
        elseif u_err < 20.0 && q_err < 25.0 && r2 > 0.9
            println("  ⚠️  PARTIALLY VALIDATED - Acceptable agreement")
            println("     Consider: longer simulation time, finer resolution, or check boundary conditions")
        else
            println("  ❌ NOT VALIDATED - Significant deviation from theory")
            println("     Issues:")
            if u_err >= 20.0
                println("       • Max velocity error too large ($(round(u_err, digits=1))% > 20%)")
            end
            if q_err >= 25.0
                println("       • Flow rate error too large ($(round(q_err, digits=1))% > 25%)")
            end
            if r2 <= 0.9
                println("       • Profile shape mismatch (R² = $(round(r2, digits=3)) < 0.9)")
            end
        end
    else
        println("  ❌ No data available")
    end
    
    println("\n📊 CARREAU-YASUDA VALIDATION:")
    if carr_metrics !== nothing && newt_metrics !== nothing
        ratio = carr_metrics["u_max"] / newt_metrics["u_max"]
        println("  Velocity ratio (CY/Newtonian): $(round(ratio, digits=3))")
        
        if ratio > 1.1
            println("  ✓ Shows expected shear-thinning behavior (higher velocity)")
        elseif ratio > 1.0
            println("  ⚠ Weak shear-thinning effect detected")
        else
            println("  ❌ Unexpected behavior (velocity should be higher for shear-thinning)")
        end
        
        println("\n  Full validation requires:")
        println("    1. ✅ Newtonian baseline must be validated first")
        println("    2. □ Compare with analytical Carreau-Yasuda Poiseuille solution")
        println("    3. □ Check apparent viscosity distribution")
        println("    4. □ Verify shear rate dependence")
        println("    5. □ Compare with experimental or DNS data from literature")
    else
        println("  ❌ Insufficient data for validation")
    end
    
    println("\n" * "="^80)
    println("✓ Analysis complete. See plots in: $PLOT_DIR")
    println("="^80 * "\n")
end

# Run validation
comprehensive_validation()
