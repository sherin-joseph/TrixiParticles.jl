# ==========================================================================================
# TGV (Taylor-Green Vortex) POSTPROCESSING SCRIPT
# ==========================================================================================
# This script processes output data from 2D/3D TGV simulations comparing different
# SGS models and stabilization settings:
#
# Variants:
#   A: noSGS (baseline stabilization)
#   B: AdamiSGS (with SGS model)
#   C: noSGS (reduced stabilization)
#   D: AdamiSGS (reduced stabilization + SGS)
#
# Expected data files in out/:
#   - tgv_WCSPH_A_noSGS_pp.csv / .json
#   - tgv_WCSPH_B_AdamiSGS_pp.csv / .json
#   - tgv_WCSPH_C_noSGS_pp.csv / .json
#   - tgv_WCSPH_D_AdamiSGS_pp.csv / .json
#
# Key metrics:
#   - kinetic_energy: Total kinetic energy decay over time
#   - avg_pressure: Average fluid pressure evolution
# ==========================================================================================

using CSV
using DataFrames
using JSON
using Plots
using Printf
using Statistics

# Set output directory
const OUT_DIR = joinpath(@__DIR__, "..", "..", "out")
const PLOT_DIR = joinpath(OUT_DIR, "plots")
mkpath(PLOT_DIR)

# Variant definitions
const VARIANTS = [
    ("A_noSGS", "A: Baseline (noSGS)", :blue, :solid),
    ("B_AdamiSGS", "B: Baseline + SGS", :red, :dash),
    ("C_noSGS", "C: Reduced (noSGS)", :green, :dot),
    ("D_AdamiSGS", "D: Reduced + SGS", :orange, :dashdot)
]

# ==========================================================================================
# DATA LOADING FUNCTIONS
# ==========================================================================================

"""
    load_tgv_data(variant_tag::String) -> DataFrame

Load CSV time series data for a specific TGV variant.
"""
function load_tgv_data(variant_tag::String)
    csv_path = joinpath(OUT_DIR, "tgv_WCSPH_$(variant_tag)_pp.csv")
    
    if !isfile(csv_path)
        @warn "CSV file not found: $csv_path"
        return nothing
    end
    
    df = CSV.read(csv_path, DataFrame)
    @info "Loaded TGV data: $variant_tag" size(df)
    return df
end

"""
    load_tgv_json(variant_tag::String) -> Dict

Load JSON metadata for a specific TGV variant.
"""
function load_tgv_json(variant_tag::String)
    json_path = joinpath(OUT_DIR, "tgv_WCSPH_$(variant_tag)_pp.json")
    
    if !isfile(json_path)
        @warn "JSON file not found: $json_path"
        return nothing
    end
    
    return JSON.parsefile(json_path)
end

# ==========================================================================================
# ANALYSIS FUNCTIONS
# ==========================================================================================

"""
    compute_ke_decay_rate(t, ke)

Compute the energy decay rate -dE/dt using finite differences.
"""
function compute_ke_decay_rate(t, ke)
    n = length(t)
    decay_rate = zeros(n)
    
    # Central differences for interior points
    for i in 2:n-1
        dt_fwd = t[i+1] - t[i]
        dt_bwd = t[i] - t[i-1]
        decay_rate[i] = -(ke[i+1] - ke[i-1]) / (dt_fwd + dt_bwd)
    end
    
    # Forward/backward differences for endpoints
    decay_rate[1] = -(ke[2] - ke[1]) / (t[2] - t[1])
    decay_rate[n] = -(ke[n] - ke[n-1]) / (t[n] - t[n-1])
    
    return decay_rate
end

# ==========================================================================================
# PLOTTING FUNCTIONS
# ==========================================================================================

"""
    plot_kinetic_energy_decay()

Plot absolute kinetic energy decay on log scale.
"""
function plot_kinetic_energy_decay()
    p = plot(xlabel="Time [s]", ylabel="Kinetic Energy [J]",
             title="TGV: Kinetic Energy Decay",
             yscale=:log10,
             legend=:topright,
             grid=true,
             size=(900, 600),
             dpi=150)
    
    for (tag, label, color, style) in VARIANTS
        df = load_tgv_data(tag)
        if df !== nothing && "kinetic_energy_fluid_1" in names(df)
            plot!(p, df.time, df.kinetic_energy_fluid_1,
                  label=label, color=color, linestyle=style, linewidth=2)
        end
    end
    
    savefig(p, joinpath(PLOT_DIR, "tgv_ke_decay.png"))
    @info "Saved: tgv_ke_decay.png"
end

"""
    plot_normalized_ke()

Plot normalized kinetic energy E/E₀.
"""
function plot_normalized_ke()
    p = plot(xlabel="Time [s]", ylabel="E/E₀",
             title="TGV: Normalized Kinetic Energy",
             legend=:topright,
             grid=true,
             size=(900, 600),
             dpi=150)
    
    for (tag, label, color, style) in VARIANTS
        df = load_tgv_data(tag)
        if df !== nothing && "kinetic_energy_fluid_1" in names(df)
            ke = df.kinetic_energy_fluid_1
            ke_norm = ke ./ ke[1]
            plot!(p, df.time, ke_norm,
                  label=label, color=color, linestyle=style, linewidth=2)
        end
    end
    
    # Reference line at E/E₀ = 1
    plot!(p, [0, 10], [1, 1], label="", color=:black, linestyle=:dash, alpha=0.3)
    
    savefig(p, joinpath(PLOT_DIR, "tgv_normalized_ke.png"))
    @info "Saved: tgv_normalized_ke.png"
end

"""
    plot_pressure_evolution()

Plot average pressure evolution.
"""
function plot_pressure_evolution()
    p = plot(xlabel="Time [s]", ylabel="Average Pressure [Pa]",
             title="TGV: Average Pressure Evolution",
             legend=:topright,
             grid=true,
             size=(900, 600),
             dpi=150)
    
    for (tag, label, color, style) in VARIANTS
        df = load_tgv_data(tag)
        if df !== nothing && "avg_pressure_fluid_1" in names(df)
            plot!(p, df.time, df.avg_pressure_fluid_1,
                  label=label, color=color, linestyle=style, linewidth=2)
        end
    end
    
    # Reference line at P = 0
    plot!(p, [0, 10], [0, 0], label="", color=:black, linestyle=:dash, alpha=0.3)
    
    savefig(p, joinpath(PLOT_DIR, "tgv_pressure.png"))
    @info "Saved: tgv_pressure.png"
end

"""
    plot_ke_decay_rate()

Plot energy decay rate -dE/dt.
"""
function plot_ke_decay_rate()
    p = plot(xlabel="Time [s]", ylabel="-dE/dt [W]",
             title="TGV: Energy Decay Rate",
             legend=:topright,
             grid=true,
             size=(900, 600),
             dpi=150)
    
    for (tag, label, color, style) in VARIANTS
        df = load_tgv_data(tag)
        if df !== nothing && "kinetic_energy_fluid_1" in names(df)
            decay_rate = compute_ke_decay_rate(df.time, df.kinetic_energy_fluid_1)
            plot!(p, df.time, decay_rate,
                  label=label, color=color, linestyle=style, linewidth=2)
        end
    end
    
    savefig(p, joinpath(PLOT_DIR, "tgv_ke_decay_rate.png"))
    @info "Saved: tgv_ke_decay_rate.png"
end

"""
    create_dashboard()

Create a comprehensive 2×2 dashboard plot.
"""
function create_dashboard()
    # Subplot 1: KE decay (log scale)
    sp1 = plot(xlabel="Time [s]", ylabel="KE [J]",
               title="Kinetic Energy",
               yscale=:log10,
               legend=:topright,
               grid=true)
    
    # Subplot 2: Normalized KE
    sp2 = plot(xlabel="Time [s]", ylabel="E/E₀",
               title="Normalized Kinetic Energy",
               legend=:topright,
               grid=true)
    
    # Subplot 3: Average pressure
    sp3 = plot(xlabel="Time [s]", ylabel="P [Pa]",
               title="Average Pressure",
               legend=:topright,
               grid=true)
    
    # Subplot 4: Decay rate
    sp4 = plot(xlabel="Time [s]", ylabel="-dE/dt [W]",
               title="Energy Decay Rate",
               legend=:topright,
               grid=true)
    
    for (tag, label, color, style) in VARIANTS
        df = load_tgv_data(tag)
        if df === nothing
            continue
        end
        
        # KE decay
        if "kinetic_energy_fluid_1" in names(df)
            ke = df.kinetic_energy_fluid_1
            plot!(sp1, df.time, ke, label=label, color=color,
                  linestyle=style, linewidth=2)
            
            # Normalized KE
            ke_norm = ke ./ ke[1]
            plot!(sp2, df.time, ke_norm, label=label, color=color,
                  linestyle=style, linewidth=2)
            
            # Decay rate
            decay_rate = compute_ke_decay_rate(df.time, ke)
            plot!(sp4, df.time, decay_rate, label=label, color=color,
                  linestyle=style, linewidth=2)
        end
        
        # Average pressure
        if "avg_pressure_fluid_1" in names(df)
            plot!(sp3, df.time, df.avg_pressure_fluid_1, label=label, color=color,
                  linestyle=style, linewidth=2)
        end
    end
    
    # Reference lines
    plot!(sp2, [0, 10], [1, 1], label="", color=:black, linestyle=:dash, alpha=0.3)
    plot!(sp3, [0, 10], [0, 0], label="", color=:black, linestyle=:dash, alpha=0.3)
    
    dashboard = plot(sp1, sp2, sp3, sp4, layout=(2,2), size=(1400, 1000), dpi=150)
    savefig(dashboard, joinpath(PLOT_DIR, "tgv_dashboard.png"))
    @info "Saved: tgv_dashboard.png"
end

"""
    print_summary_statistics()

Print summary statistics for all variants.
"""
function print_summary_statistics()
    println("\n" * "="^80)
    println("TGV SUMMARY STATISTICS")
    println("="^80)
    
    for (tag, label, _, _) in VARIANTS
        df = load_tgv_data(tag)
        if df === nothing
            continue
        end
        
        println("\n$label ($tag):")
        println("  " * "-"^70)
        
        # KE statistics
        if "kinetic_energy_fluid_1" in names(df)
            ke = df.kinetic_energy_fluid_1
            ke_decay = 100 * (1 - ke[end] / ke[1])
            @printf("    Initial KE:    %.6e J\n", ke[1])
            @printf("    Final KE:      %.6e J\n", ke[end])
            @printf("    KE decay:      %.2f%%\n", ke_decay)
            
            # Decay rate statistics
            decay_rate = compute_ke_decay_rate(df.time, ke)
            @printf("    Mean |dE/dt|:  %.6e W\n", mean(abs.(decay_rate)))
            @printf("    Max |dE/dt|:   %.6e W\n", maximum(abs.(decay_rate)))
        end
        
        # Pressure statistics
        if "avg_pressure_fluid_1" in names(df)
            p_avg = df.avg_pressure_fluid_1
            @printf("    Avg pressure:  %.6e Pa\n", mean(p_avg))
            @printf("    Pressure std:  %.6e Pa\n", std(p_avg))
        end
    end
    
    println("\n" * "="^80)
end

# ==========================================================================================
# MAIN EXECUTION
# ==========================================================================================

function main()
    println("\n" * "="^80)
    println("TGV POSTPROCESSING")
    println("="^80)
    
    # Generate all plots
    @info "Generating plots..."
    plot_kinetic_energy_decay()
    plot_normalized_ke()
    plot_pressure_evolution()
    plot_ke_decay_rate()
    create_dashboard()
    
    # Print statistics
    print_summary_statistics()
    
    println("\n✓ All plots saved to: $PLOT_DIR")
    println("="^80 * "\n")
end

# Run automatically
main()
