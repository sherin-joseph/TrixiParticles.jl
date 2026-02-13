# ==========================================================================================
# Kelvin-Helmholtz (KH) Instability Postprocessing Script
#
# This script processes and visualizes results from KH simulations with different variants:
#   - Variant A: noSGS (no subgrid scale model)
#   - Variant B: AdamiSGS (with Adami subgrid scale model)
#   - Variant C: noSGS (reduced stabilization)
#   - Variant D: AdamiSGS (reduced stabilization with SGS)
#
# Outputs:
#   - Time series plots (kinetic energy, average pressure)
#   - Comparison plots between variants
#   - Optional VTK visualization snapshots
# ==========================================================================================

using CSV
using DataFrames
using JSON
using Plots
using Printf
using Statistics

# Directory containing the output files
const OUT_DIR = raw"C:\Users\Administrator\.julia\TrixiParticles.jl\out"
const PLOT_DIR = joinpath(OUT_DIR, "plots")

# Create plots directory if it doesn't exist
!isdir(PLOT_DIR) && mkpath(PLOT_DIR)

# ==========================================================================================
# Helper Functions
# ==========================================================================================

"""
    load_kh_data(variant::String, sgs_tag::String)

Load KH postprocessing data for a specific variant.
Returns a DataFrame with time, kinetic_energy, and avg_pressure columns.
"""
function load_kh_data(variant::String, sgs_tag::String)
    prefix = "kh_WCSPH_$(variant)_$(sgs_tag)"
    csv_file = joinpath(OUT_DIR, "$(prefix)_pp.csv")
    
    if !isfile(csv_file)
        @warn "File not found: $csv_file"
        return nothing
    end
    
    data = CSV.read(csv_file, DataFrame)
    @info "Loaded data for Variant $variant ($sgs_tag)" nrows=nrow(data)
    return data
end

"""
    load_kh_json(variant::String, sgs_tag::String)

Load KH postprocessing data from JSON file.
Returns a dictionary with the parsed JSON content.
"""
function load_kh_json(variant::String, sgs_tag::String)
    prefix = "kh_WCSPH_$(variant)_$(sgs_tag)"
    json_file = joinpath(OUT_DIR, "$(prefix)_pp.json")
    
    if !isfile(json_file)
        @warn "File not found: $json_file"
        return nothing
    end
    
    file_content = read(json_file, String)
    data = JSON.parse(file_content)
    @info "Loaded JSON for Variant $variant ($sgs_tag)"
    return data
end

"""
    normalize_kinetic_energy(data::DataFrame)

Normalize kinetic energy by its initial value.
"""
function normalize_kinetic_energy(data::DataFrame)
    ke_col = names(data)[findfirst(name -> occursin("kinetic_energy", name), names(data))]
    ke0 = data[1, ke_col]
    return data[:, ke_col] ./ ke0
end

# ==========================================================================================
# Data Loading
# ==========================================================================================

println("\n" * "="^80)
println("Loading KH Simulation Data")
println("="^80)

# Define all variants
variants = [
    ("A", "noSGS"),
    ("B", "AdamiSGS"),
    ("C", "noSGS"),
    ("D", "AdamiSGS")
]

# Load all available data
kh_data = Dict()
for (variant, sgs_tag) in variants
    key = "$(variant)_$(sgs_tag)"
    data = load_kh_data(variant, sgs_tag)
    if data !== nothing
        kh_data[key] = data
    end
end

if isempty(kh_data)
    error("No KH data found in $OUT_DIR")
end

# ==========================================================================================
# Plot 1: Kinetic Energy Evolution (All Variants)
# ==========================================================================================

println("\nGenerating Plot 1: Kinetic Energy Evolution...")

p1 = plot(title="Kinetic Energy Evolution - KH Instability",
          xlabel="Time (s)",
          ylabel="Kinetic Energy",
          legend=:best,
          size=(1000, 600),
          linewidth=2,
          grid=true,
          framestyle=:box)

colors = Dict("A_noSGS" => :blue, "B_AdamiSGS" => :red, 
              "C_noSGS" => :green, "D_AdamiSGS" => :orange)
linestyles = Dict("A_noSGS" => :solid, "B_AdamiSGS" => :solid,
                  "C_noSGS" => :dash, "D_AdamiSGS" => :dash)

for (key, data) in kh_data
    ke_col = names(data)[findfirst(name -> occursin("kinetic_energy", name), names(data))]
    plot!(p1, data.time, data[:, ke_col],
          label=key,
          color=get(colors, key, :black),
          linestyle=get(linestyles, key, :solid))
end

savefig(p1, joinpath(PLOT_DIR, "kh_kinetic_energy_all.png"))
println("  Saved: kh_kinetic_energy_all.png")

# ==========================================================================================
# Plot 2: Normalized Kinetic Energy (Decay Analysis)
# ==========================================================================================

println("\nGenerating Plot 2: Normalized Kinetic Energy...")

p2 = plot(title="Normalized Kinetic Energy - KH Instability",
          xlabel="Time (s)",
          ylabel="KE / KE₀",
          legend=:best,
          size=(1000, 600),
          linewidth=2,
          grid=true,
          framestyle=:box)

for (key, data) in kh_data
    ke_normalized = normalize_kinetic_energy(data)
    plot!(p2, data.time, ke_normalized,
          label=key,
          color=get(colors, key, :black),
          linestyle=get(linestyles, key, :solid))
end

# Add reference line at KE/KE0 = 1.0
hline!(p2, [1.0], label="Initial Value", color=:black, linestyle=:dot, linewidth=1)

savefig(p2, joinpath(PLOT_DIR, "kh_normalized_ke.png"))
println("  Saved: kh_normalized_ke.png")

# ==========================================================================================
# Plot 3: Average Pressure Evolution
# ==========================================================================================

println("\nGenerating Plot 3: Average Pressure Evolution...")

p3 = plot(title="Average Pressure Evolution - KH Instability",
          xlabel="Time (s)",
          ylabel="Average Pressure",
          legend=:best,
          size=(1000, 600),
          linewidth=2,
          grid=true,
          framestyle=:box)

for (key, data) in kh_data
    pressure_col = names(data)[findfirst(name -> occursin("avg_pressure", name), names(data))]
    plot!(p3, data.time, data[:, pressure_col],
          label=key,
          color=get(colors, key, :black),
          linestyle=get(linestyles, key, :solid))
end

savefig(p3, joinpath(PLOT_DIR, "kh_avg_pressure_all.png"))
println("  Saved: kh_avg_pressure_all.png")

# ==========================================================================================
# Plot 4: Combined Dashboard (2x2 layout)
# ==========================================================================================

println("\nGenerating Plot 4: Combined Dashboard...")

# Create individual comparison plots
p4a = plot(title="KE: Baseline (A,B) vs Reduced Stab (C,D)",
           xlabel="Time", ylabel="Kinetic Energy",
           legend=:topright, size=(500, 400))

p4b = plot(title="Pressure: Baseline vs Reduced Stab",
           xlabel="Time", ylabel="Avg Pressure",
           legend=:topright, size=(500, 400))

p4c = plot(title="KE: SGS Effect (noSGS vs AdamiSGS)",
           xlabel="Time", ylabel="Kinetic Energy",
           legend=:topright, size=(500, 400))

p4d = plot(title="Pressure: SGS Effect",
           xlabel="Time", ylabel="Avg Pressure",
           legend=:topright, size=(500, 400))

# Add data to subplots
for (key, data) in kh_data
    ke_col = names(data)[findfirst(name -> occursin("kinetic_energy", name), names(data))]
    pressure_col = names(data)[findfirst(name -> occursin("avg_pressure", name), names(data))]
    
    # Baseline vs Reduced (A,B vs C,D)
    if occursin("A_", key) || occursin("B_", key)
        plot!(p4a, data.time, data[:, ke_col], label=key, color=get(colors, key, :black))
        plot!(p4b, data.time, data[:, pressure_col], label=key, color=get(colors, key, :black))
    end
    if occursin("C_", key) || occursin("D_", key)
        plot!(p4a, data.time, data[:, ke_col], label=key, 
              color=get(colors, key, :black), linestyle=:dash)
        plot!(p4b, data.time, data[:, pressure_col], label=key,
              color=get(colors, key, :black), linestyle=:dash)
    end
    
    # SGS effect (noSGS vs AdamiSGS)
    if occursin("noSGS", key)
        plot!(p4c, data.time, data[:, ke_col], label=key, color=get(colors, key, :black))
        plot!(p4d, data.time, data[:, pressure_col], label=key, color=get(colors, key, :black))
    end
    if occursin("AdamiSGS", key)
        plot!(p4c, data.time, data[:, ke_col], label=key,
              color=get(colors, key, :black), linestyle=:dashdot)
        plot!(p4d, data.time, data[:, pressure_col], label=key,
              color=get(colors, key, :black), linestyle=:dashdot)
    end
end

p4_combined = plot(p4a, p4b, p4c, p4d, layout=(2,2), size=(1200, 1000))
savefig(p4_combined, joinpath(PLOT_DIR, "kh_dashboard.png"))
println("  Saved: kh_dashboard.png")

# ==========================================================================================
# Statistical Summary
# ==========================================================================================

println("\n" * "="^80)
println("Statistical Summary")
println("="^80)

for (key, data) in kh_data
    ke_col = names(data)[findfirst(name -> occursin("kinetic_energy", name), names(data))]
    pressure_col = names(data)[findfirst(name -> occursin("avg_pressure", name), names(data))]
    
    ke_initial = data[1, ke_col]
    ke_final = data[end, ke_col]
    ke_decay = (ke_initial - ke_final) / ke_initial * 100
    
    pressure_mean = mean(data[:, pressure_col])
    pressure_std = std(data[:, pressure_col])
    
    println("\nVariant: $key")
    println("  Time range: $(data.time[1]) - $(data.time[end]) s")
    println("  Kinetic Energy:")
    println("    Initial: $(@sprintf("%.6e", ke_initial))")
    println("    Final:   $(@sprintf("%.6e", ke_final))")
    println("    Decay:   $(@sprintf("%.2f", ke_decay))%")
    println("  Pressure:")
    println("    Mean:    $(@sprintf("%.6e", pressure_mean))")
    println("    Std Dev: $(@sprintf("%.6e", pressure_std))")
end

# ==========================================================================================
# Optional: VTK File Information
# ==========================================================================================

println("\n" * "="^80)
println("VTK Output Files")
println("="^80)

for (variant, sgs_tag) in variants
    prefix = "kh_WCSPH_$(variant)_$(sgs_tag)"
    vtu_files = filter(f -> startswith(f, prefix) && endswith(f, ".vtu"), readdir(OUT_DIR))
    
    if !isempty(vtu_files)
        println("\nVariant $variant ($sgs_tag): $(length(vtu_files)) VTU files")
        println("  Pattern: $(prefix)_fluid_1_*.vtu")
        println("  Visualize with ParaView or similar VTK viewer")
    end
end

# ==========================================================================================
# Print Summary
# ==========================================================================================

println("\n" * "="^80)
println("Postprocessing Complete!")
println("="^80)
println("Output directory: $PLOT_DIR")
println("\nGenerated plots:")
println("  1. kh_kinetic_energy_all.png    - KE evolution for all variants")
println("  2. kh_normalized_ke.png          - Normalized KE (decay analysis)")
println("  3. kh_avg_pressure_all.png       - Pressure evolution")
println("  4. kh_dashboard.png              - Combined comparison dashboard")
println("="^80 * "\n")
