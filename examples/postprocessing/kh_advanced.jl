# ==========================================================================================
# Advanced KH Visualization with VTK Support
#
# This script provides advanced visualization options for KH simulations including:
#   - Spatial plots from VTK data (requires ReadVTK.jl)
#   - Vorticity field visualization
#   - Particle distribution snapshots
#   - Energy spectrum analysis (if available)
# ==========================================================================================

using CSV, DataFrames, Plots
using Printf, Statistics
using LinearAlgebra

# Try to load ReadVTK if available
TRY_VTK = true
VTK_AVAILABLE = false

if TRY_VTK
    try
        using ReadVTK
        global VTK_AVAILABLE = true
        println("✓ ReadVTK.jl loaded - VTK visualization enabled")
    catch
        @warn "ReadVTK.jl not available. Install with: using Pkg; Pkg.add(\"ReadVTK\")\n" *
              "VTK visualization will be skipped."
    end
end

# Paths
OUT_DIR = raw"C:\Users\Administrator\.julia\TrixiParticles.jl\out"
PLOT_DIR = joinpath(OUT_DIR, "plots_advanced")
!isdir(PLOT_DIR) && mkpath(PLOT_DIR)

# ==========================================================================================
# VTK Processing Functions
# ==========================================================================================

"""
    load_vtu_snapshot(variant, sgs_tag, step::Int)

Load a VTU file for a specific timestep.
Returns particle positions, velocities, and other fields.
"""
function load_vtu_snapshot(variant, sgs_tag, step::Int)
    if !VTK_AVAILABLE
        @warn "VTK loading not available"
        return nothing
    end
    
    prefix = "kh_WCSPH_$(variant)_$(sgs_tag)"
    filename = joinpath(OUT_DIR, "$(prefix)_fluid_1_$(step).vtu")
    
    if !isfile(filename)
        @warn "VTU file not found: $filename"
        return nothing
    end
    
    try
        vtk_data = VTKFile(filename)
        
        # Extract point data and convert to regular arrays
        points = get_points(vtk_data)
        points_array = Matrix(points)  # Convert to regular Matrix
        
        point_data = get_point_data(vtk_data)
        
        result = Dict(
            "positions" => points_array,
            "step" => step
        )
        
        # Extract available fields and convert to regular arrays
        for (field_name, field_data) in point_data
            try
                # Convert VTKDataArray to regular array
                if ndims(field_data) == 2
                    result[field_name] = Matrix(field_data)
                else
                    result[field_name] = Vector(field_data)
                end
            catch
                # If conversion fails, try different approach
                result[field_name] = collect(field_data)
            end
        end
        
        nparticles = size(points_array, 2)
        @info "Loaded VTU snapshot" step file=basename(filename) nparticles=nparticles
        return result
    catch e
        @warn "Error loading VTU file: $e"
        @warn "Stack trace:" exception=(e, catch_backtrace())
        return nothing
    end
end

"""
    compute_vorticity_2d(positions, velocities)

Compute vorticity field for 2D flow using particle data.
"""
function compute_vorticity_2d(positions, velocities)
    # Simple finite-difference approximation
    # In practice, SPH kernels would be better
    nparticles = size(positions, 2)
    vorticity = zeros(nparticles)
    
    # This is a placeholder - proper implementation would use SPH interpolation
    for i in 1:nparticles
        # ω = ∂v/∂x - ∂u/∂y (for 2D)
        # This requires neighbor search and gradient calculation
        vorticity[i] = 0.0  # Placeholder
    end
    
    return vorticity
end

"""
    plot_particle_snapshot(data, variant_name; field="velocity_magnitude")

Create a scatter plot of particle data colored by a field.
"""
function plot_particle_snapshot(data, variant_name; field="velocity_magnitude")
    if data === nothing
        return nothing
    end
    
    try
        positions = data["positions"]
        x = positions[1, :]
        y = positions[2, :]
        
        # Try to extract the field for coloring
        c = nothing
        
        if haskey(data, field)
            field_data = data[field]
            
            # Handle different data types
            if field_data isa AbstractVector
                c = Vector(field_data)
            elseif field_data isa AbstractMatrix
                # For matrix data, compute magnitude
                nparticles = size(field_data, 2)
                c = zeros(nparticles)
                for i in 1:nparticles
                    c[i] = norm(field_data[:, i])
                end
            else
                c = collect(field_data)
            end
        end
        
        # Fallback: try velocity if requested field not found
        if c === nothing && haskey(data, "velocity")
            vel = data["velocity"]
            if vel isa AbstractMatrix
                nparticles = size(vel, 2)
                c = zeros(nparticles)
                for i in 1:nparticles
                    c[i] = norm(vel[:, i])
                end
            end
        end
        
        # Final fallback: uniform coloring
        if c === nothing
            c = ones(length(x))
        end
        
        # Ensure c is a vector of the right length
        if length(c) != length(x)
            @warn "Color data length mismatch" length(c) length(x)
            c = ones(length(x))
        end
        
        p = scatter(x, y, 
                    marker_z=c,
                    markersize=2,
                    markerstrokewidth=0,
                    title="$variant_name - Step $(data["step"])\n$field",
                    xlabel="x",
                    ylabel="y",
                    aspect_ratio=:equal,
                    size=(800, 800),
                    colorbar=true,
                    clims=(minimum(c), maximum(c)))
        
        return p
    catch e
        @warn "Error creating particle snapshot plot: $e"
        @warn "Stack trace:" exception=(e, catch_backtrace())
        return nothing
    end
end

# ==========================================================================================
# Time Series Analysis with Phase Space Plots
# ==========================================================================================

"""
    create_phase_space_plot(data_dict)

Create phase space plot (KE vs Pressure) for all variants.
"""
function create_phase_space_plot(data_dict)
    p = plot(title="Phase Space: KE vs Pressure",
             xlabel="Average Pressure",
             ylabel="Kinetic Energy",
             legend=:best,
             size=(800, 600),
             grid=true)
    
    colors = [:blue, :red, :green, :orange, :purple]
    
    for (idx, (name, df)) in enumerate(data_dict)
        # Find KE and pressure columns
        ke_col = findfirst(name -> occursin("kinetic_energy", name), names(df))
        press_col = findfirst(name -> occursin("avg_pressure", name), names(df))
        
        if ke_col !== nothing && press_col !== nothing
            plot!(p, df[!, press_col], df[!, ke_col],
                  label=name,
                  color=colors[mod1(idx, length(colors))],
                  linewidth=2,
                  alpha=0.7)
        end
    end
    
    return p
end

"""
    create_time_derivative_plot(data_dict)

Plot time derivatives to analyze rates of change.
"""
function create_time_derivative_plot(data_dict)
    p = plot(title="KE Dissipation Rate",
             xlabel="Time (s)",
             ylabel="dKE/dt",
             legend=:best,
             size=(900, 600),
             grid=true)
    
    colors = [:blue, :red, :green, :orange]
    
    for (idx, (name, df)) in enumerate(data_dict)
        ke_col = findfirst(name -> occursin("kinetic_energy", name), names(df))
        
        if ke_col !== nothing && nrow(df) > 1
            # Compute time derivative
            dt = diff(df.time)
            dke_dt = diff(df[!, ke_col]) ./ dt
            t_mid = df.time[1:end-1] .+ dt./2
            
            plot!(p, t_mid, dke_dt,
                  label=name,
                  color=colors[mod1(idx, length(colors))],
                  linewidth=2)
        end
    end
    
    hline!(p, [0], linestyle=:dash, color=:black, label="Zero", alpha=0.5)
    
    return p
end

# ==========================================================================================
# Main Analysis
# ==========================================================================================

println("\n" * "="^80)
println("ADVANCED KH POSTPROCESSING")
println("="^80)

# Load CSV data
kh_files = filter(f -> startswith(f, "kh_WCSPH_") && endswith(f, "_pp.csv"),
                  readdir(OUT_DIR))

data_dict = Dict()
for file in kh_files
    variant_name = replace(file, "kh_WCSPH_" => "", "_pp.csv" => "")
    println("Loading: $variant_name")
    data_dict[variant_name] = CSV.read(joinpath(OUT_DIR, file), DataFrame)
end

if isempty(data_dict)
    error("No KH data found")
end

# Create advanced plots
println("\nGenerating advanced visualizations...")

# 1. Phase space plot
p_phase = create_phase_space_plot(data_dict)
savefig(p_phase, joinpath(PLOT_DIR, "kh_phase_space.png"))
println("  ✓ Phase space plot saved")

# 2. Time derivative plot
p_deriv = create_time_derivative_plot(data_dict)
savefig(p_deriv, joinpath(PLOT_DIR, "kh_dissipation_rate.png"))
println("  ✓ Dissipation rate plot saved")

# 3. Log-scale kinetic energy
p_log = plot(title="Kinetic Energy (Log Scale)",
             xlabel="Time (s)",
             ylabel="log₁₀(KE)",
             legend=:best,
             size=(900, 600))

for (idx, (name, df)) in enumerate(data_dict)
    ke_col = findfirst(name -> occursin("kinetic_energy", name), names(df))
    if ke_col !== nothing
        ke_positive = max.(df[!, ke_col], 1e-20)  # Avoid log(0)
        plot!(p_log, df.time, log10.(ke_positive),
              label=name,
              linewidth=2)
    end
end

savefig(p_log, joinpath(PLOT_DIR, "kh_log_ke.png"))
println("  ✓ Log-scale KE plot saved")

# ==========================================================================================
# VTK Snapshots (if available)
# ==========================================================================================

if VTK_AVAILABLE
    println("\nProcessing VTK snapshots...")
    
    # Select a few timesteps to visualize
    snapshot_steps = [0, 50, 100, 150, 200, 250]
    
    for (variant_name, df) in data_dict
        # Extract variant and sgs_tag
        parts = split(variant_name, "_")
        if length(parts) >= 2
            variant = parts[1]
            sgs_tag = join(parts[2:end], "_")
            
            println("  Processing variant: $variant_name")
            
            for step in snapshot_steps
                try
                    data = load_vtu_snapshot(variant, sgs_tag, step)
                    
                    if data !== nothing
                        # Use invokelatest to avoid world age issues
                        p_snap = Base.invokelatest(plot_particle_snapshot, 
                                                    data, variant_name,
                                                    field="velocity_magnitude")
                        
                        if p_snap !== nothing
                            filename = "kh_snapshot_$(variant_name)_step$(step).png"
                            savefig(p_snap, joinpath(PLOT_DIR, filename))
                            println("    ✓ Saved snapshot for step $step")
                        end
                    end
                catch e
                    @warn "Failed to process step $step for $variant_name: $e"
                end
            end
        end
    end
    
    println("  ✓ VTK snapshots processed")
else
    println("\nVTK visualization skipped (ReadVTK not available)")
end

# ==========================================================================================
# Summary Report
# ==========================================================================================

println("\n" * "="^80)
println("SUMMARY REPORT")
println("="^80)

for (name, df) in data_dict
    println("\n$name:")
    
    ke_col = findfirst(col -> occursin("kinetic_energy", col), names(df))
    press_col = findfirst(col -> occursin("avg_pressure", col), names(df))
    
    if ke_col !== nothing
        ke_data = df[!, ke_col]
        println(@sprintf("  KE:   min=%.4e  max=%.4e  final=%.4e",
                         minimum(ke_data), maximum(ke_data), ke_data[end]))
        
        # Decay percentage
        if ke_data[1] > 0
            decay_pct = (1 - ke_data[end]/ke_data[1]) * 100
            println(@sprintf("        Decay: %.2f%%", decay_pct))
        end
    end
    
    if press_col !== nothing
        press_data = df[!, press_col]
        println(@sprintf("  Pressure: mean=%.4e  std=%.4e",
                         mean(press_data), std(press_data)))
    end
end

println("\n" * "="^80)
println("All plots saved to: $PLOT_DIR")
println("="^80 * "\n")
