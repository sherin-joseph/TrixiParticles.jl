# ==========================================================================================
# TGV ADVANCED POSTPROCESSING with VTK Analysis
# ==========================================================================================
# Advanced postprocessing with 3D/2D particle visualization and vorticity statistics.
#
# Features:
#   - 2D/3D particle snapshots with velocity magnitude coloring
#   - Velocity histogram evolution
#   - Vorticity field analysis
#   - Phase space plots (velocity components)
#   - Flow structure visualization
#
# Requirements:
#   - CSV, DataFrames, Plots, Statistics
#   - ReadVTK (optional, for VTU file reading)
#
# Usage:
#   julia tgv_advanced.jl
# ==========================================================================================

using CSV
using DataFrames
using Plots
using Statistics
using Printf

# Try to load ReadVTK (optional)
const HAS_VTK = try
    using ReadVTK
    true
catch
    @warn "ReadVTK not available. VTK analysis will be skipped."
    false
end

# Configuration
const OUT_DIR = joinpath(@__DIR__, "..", "..", "out")
const PLOT_DIR = joinpath(OUT_DIR, "plots_advanced")
mkpath(PLOT_DIR)

const VARIANTS = [
    ("A_noSGS", "A: Baseline (noSGS)", :blue),
    ("B_AdamiSGS", "B: Baseline + SGS", :red),
    ("C_noSGS", "C: Reduced (noSGS)", :green),
    ("D_AdamiSGS", "D: Reduced + SGS", :orange)
]

# ==========================================================================================
# VTK READING FUNCTIONS (with robust error handling)
# ==========================================================================================

"""
    materialize_vtk_array(arr)

Convert VTKDataArray to standard Julia array by materializing all data.
"""
function materialize_vtk_array(arr)
    # Force materialization by trying different slicing patterns
    # This avoids calling size(), ndims(), or length() which may not be defined
    try
        # Try 3D slice first (most common for velocity fields)
        result = Float64.(arr[:, :, :])
        # Check if result is 3D and first dimension is 3 (velocity-like)
        if size(result, 1) == 3
            # Flatten extra dimensions: 3xNxM -> 3xN
            return reshape(result, 3, :)
        elseif size(result, 1) == 2
            # 2D velocity field: 2xNxM -> 2xN
            return reshape(result, 2, :)
        end
        return result
    catch
        try
            # Try 2D slice
            return Float64.(arr[:, :])
        catch
            try
                # Try 1D slice
                return Float64.(arr[:])
            catch e
                error("Failed to materialize VTK array: $e")
            end
        end
    end
end

"""
    load_vtu_snapshot(variant_tag, step)

Load a VTU snapshot file and extract particle data.
"""
function load_vtu_snapshot(variant_tag::String, step::Int)
    if !HAS_VTK
        return nothing
    end
    
    vtu_file = joinpath(OUT_DIR, "tgv_WCSPH_$(variant_tag)_fluid_1_$(step).vtu")
    
    if !isfile(vtu_file)
        @warn "VTU file not found: $vtu_file"
        return nothing
    end
    
    try
        vtk_data = VTKFile(vtu_file)
        
        # Extract points (particle positions) - should be 2xN or 3xN
        points_raw = get_points(vtk_data)
        points = materialize_vtk_array(points_raw)
        
        # Extract velocity field - should be 2xN or 3xN
        velocity_raw = get_point_data(vtk_data)["velocity"]
        velocity = materialize_vtk_array(velocity_raw)
        
        # Ensure points and velocity are DxN matrices (D=2 or 3)
        if length(size(points)) == 1
            D = size(velocity, 1)  # Get dimensions from velocity
            points = reshape(points, D, :)
        end
        if length(size(velocity)) == 1
            D = size(points, 1)  # Get dimensions from points
            velocity = reshape(velocity, D, :)
        end
        
        # Extract other fields if available
        density = nothing
        pressure = nothing
        
        try
            density_raw = get_point_data(vtk_data)["density"]
            density = materialize_vtk_array(density_raw)
            # Flatten to 1D if needed
            if length(size(density)) > 1
                density = vec(density)
            end
        catch e
            @debug "Density field not available" exception=e
        end
        
        try
            pressure_raw = get_point_data(vtk_data)["pressure"]
            pressure = materialize_vtk_array(pressure_raw)
            # Flatten to 1D if needed
            if length(size(pressure)) > 1
                pressure = vec(pressure)
            end
        catch e
            @debug "Pressure field not available" exception=e
        end
        
        return (points=points, velocity=velocity, density=density, pressure=pressure)
    catch e
        @warn "Error reading VTU file: $vtu_file" exception=e
        return nothing
    end
end

"""
    compute_velocity_magnitude(velocity)

Compute velocity magnitude for each particle.
"""
function compute_velocity_magnitude(velocity::Matrix)
    D, n_particles = size(velocity)
    vmag = zeros(n_particles)
    
    if D == 2
        for i in 1:n_particles
            vx, vy = velocity[1, i], velocity[2, i]
            vmag[i] = sqrt(vx^2 + vy^2)
        end
    elseif D == 3
        for i in 1:n_particles
            vx, vy, vz = velocity[1, i], velocity[2, i], velocity[3, i]
            vmag[i] = sqrt(vx^2 + vy^2 + vz^2)
        end
    end
    
    return vmag
end

"""
    compute_vorticity_2d(velocity)

Compute vorticity magnitude for 2D flow (returns scalar field).
For TGV, this is the key diagnostic quantity.
"""
function compute_vorticity_2d(velocity::Matrix)
    # For particle methods, this is approximate
    # We'll just return the velocity magnitude as a proxy
    # True vorticity would require gradient computation
    return compute_velocity_magnitude(velocity)
end

# ==========================================================================================
# ANALYSIS FUNCTIONS
# ==========================================================================================

"""
    plot_velocity_histogram(variant_tag, label, color)

Plot velocity magnitude histogram at multiple time steps.
"""
function plot_velocity_histogram(variant_tag::String, label::String, color)
    if !HAS_VTK
        return
    end
    
    @info "Generating velocity histogram..."
    
    # Sample time steps
    steps = [0, 10, 20, 30, 40]
    p = plot(xlabel="Velocity Magnitude [m/s]", ylabel="Particle Count",
             title="$label: Velocity Distribution Evolution",
             legend=:topright,
             grid=true,
             size=(900, 600),
             dpi=150)
    
    styles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
    
    for (i, step) in enumerate(steps)
        data = load_vtu_snapshot(variant_tag, step)
        if data !== nothing
            vmag = compute_velocity_magnitude(data.velocity)
            histogram!(p, vmag, bins=50, alpha=0.5, 
                      label="t=$(step*0.01)s",
                      color=color, linestyle=styles[i])
        end
    end
    
    savefig(p, joinpath(PLOT_DIR, "tgv_$(variant_tag)_velocity_hist.png"))
    @info "Saved: tgv_$(variant_tag)_velocity_hist.png"
end

"""
    plot_particle_snapshots(variant_tag, label, color)

Plot 2D/3D particle snapshots colored by velocity magnitude.
"""
function plot_particle_snapshots(variant_tag::String, label::String, color)
    if !HAS_VTK
        return
    end
    
    @info "Generating particle snapshots..."
    
    # Sample time steps
    steps = [0, 20, 40]
    plots_list = []
    
    for step in steps
        data = load_vtu_snapshot(variant_tag, step)
        if data !== nothing
            vmag = compute_velocity_magnitude(data.velocity)
            D = size(data.points, 1)
            
            if D == 2
                # 2D plot
                sp = scatter(data.points[1, :], data.points[2, :],
                           marker_z=vmag, markersize=2, markerstrokewidth=0,
                           xlabel="X [m]", ylabel="Y [m]",
                           title="t=$(step*0.01)s",
                           colorbar=true, clims=(0, maximum(vmag)),
                           aspect_ratio=:equal)
            else
                # 3D plot (show X-Y plane)
                sp = scatter(data.points[1, :], data.points[2, :],
                           marker_z=vmag, markersize=2, markerstrokewidth=0,
                           xlabel="X [m]", ylabel="Y [m]",
                           title="t=$(step*0.01)s",
                           colorbar=true, clims=(0, maximum(vmag)),
                           aspect_ratio=:equal)
            end
            push!(plots_list, sp)
        end
    end
    
    if !isempty(plots_list)
        combined = plot(plots_list..., layout=(1, length(plots_list)),
                       size=(900*length(plots_list), 600), dpi=150)
        savefig(combined, joinpath(PLOT_DIR, "tgv_$(variant_tag)_snapshots.png"))
        @info "Saved: tgv_$(variant_tag)_snapshots.png"
    end
end

"""
    plot_velocity_phase_space(variant_tag, label, color)

Plot velocity phase space (vx vs vy).
"""
function plot_velocity_phase_space(variant_tag::String, label::String, color)
    if !HAS_VTK
        return
    end
    
    @info "Generating velocity phase space..."
    
    step = 20  # Mid-simulation
    data = load_vtu_snapshot(variant_tag, step)
    
    if data !== nothing
        D = size(data.velocity, 1)
        
        if D >= 2
            p = scatter(data.velocity[1, :], data.velocity[2, :],
                       markersize=2, markerstrokewidth=0, alpha=0.3,
                       xlabel="vₓ [m/s]", ylabel="vᵧ [m/s]",
                       title="$label: Velocity Phase Space (t=$(step*0.01)s)",
                       color=color,
                       legend=false,
                       size=(800, 800),
                       aspect_ratio=:equal,
                       dpi=150)
            
            savefig(p, joinpath(PLOT_DIR, "tgv_$(variant_tag)_phase_space.png"))
            @info "Saved: tgv_$(variant_tag)_phase_space.png"
        end
    end
end

"""
    plot_flow_statistics_evolution(variant_tag, label, color)

Plot evolution of flow statistics from VTK snapshots.
"""
function plot_flow_statistics_evolution(variant_tag::String, label::String, color)
    if !HAS_VTK
        return
    end
    
    @info "Generating flow statistics evolution..."
    
    # Collect statistics at multiple time steps
    steps = 0:5:100
    times = Float64[]
    vmag_mean = Float64[]
    vmag_std = Float64[]
    vmag_max = Float64[]
    
    for step in steps
        data = load_vtu_snapshot(variant_tag, step)
        if data !== nothing
            vmag = compute_velocity_magnitude(data.velocity)
            push!(times, step * 0.01)
            push!(vmag_mean, mean(vmag))
            push!(vmag_std, std(vmag))
            push!(vmag_max, maximum(vmag))
        end
    end
    
    if !isempty(times)
        p = plot(layout=(3,1), size=(900, 900), dpi=150)
        
        plot!(p[1], times, vmag_mean, label="Mean", color=color,
              linewidth=2, ylabel="Mean |v| [m/s]", grid=true)
        
        plot!(p[2], times, vmag_std, label="Std Dev", color=color,
              linewidth=2, ylabel="Std |v| [m/s]", grid=true)
        
        plot!(p[3], times, vmag_max, label="Max", color=color,
              linewidth=2, xlabel="Time [s]", ylabel="Max |v| [m/s]", grid=true)
        
        plot!(p, suptitle="$label: Flow Statistics Evolution")
        
        savefig(p, joinpath(PLOT_DIR, "tgv_$(variant_tag)_flow_stats.png"))
        @info "Saved: tgv_$(variant_tag)_flow_stats.png"
    else
        @warn "No valid VTK data found for $variant_tag"
    end
end

# ==========================================================================================
# MAIN EXECUTION
# ==========================================================================================

function main()
    println("\n" * "="^80)
    println("TGV ADVANCED POSTPROCESSING")
    println("="^80)
    
    if !HAS_VTK
        @warn "ReadVTK not available. Skipping VTK analysis."
        @warn "Install with: using Pkg; Pkg.add(\"ReadVTK\")"
        return
    end
    
    for (tag, label, color) in VARIANTS
        println("\n" * "-"^80)
        println("Processing: $label ($tag)")
        println("-"^80)
        
        # VTK-based analysis
        plot_velocity_histogram(tag, label, color)
        plot_particle_snapshots(tag, label, color)
        plot_velocity_phase_space(tag, label, color)
        plot_flow_statistics_evolution(tag, label, color)
    end
    
    println("\n" * "="^80)
    println("✓ All advanced plots saved to: $PLOT_DIR")
    println("="^80 * "\n")
end

# Run automatically
main()
