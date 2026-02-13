# ==========================================================================================
# DHIT (Decaying Homogeneous Isotropic Turbulence) POSTPROCESSING SCRIPT
# ==========================================================================================
# This script processes output data from 3D DHIT simulations comparing different
# SGS models and stabilization settings:
#
# Variants:
#   A: noSGS (baseline stabilization)
#   B: AdamiSGS (with SGS model)
#   C: noSGS (reduced stabilization)
#   D: AdamiSGS (reduced stabilization + SGS)
#
# Expected data files in out/:
#   - dhit_WCSPH_A_noSGS_pp.csv / .json
#   - dhit_WCSPH_B_AdamiSGS_pp.csv / .json
#   - dhit_WCSPH_C_noSGS_pp.csv / .json
#   - dhit_WCSPH_D_AdamiSGS_pp.csv / .json
#
# Key metrics:
#   - kinetic_energy: Total kinetic energy decay over time
#   - avg_density: Average fluid density (compressibility indicator)
#   - max_pressure / min_pressure: Pressure range evolution
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
    load_dhit_data(variant_tag::String) -> DataFrame

Load CSV time series data for a specific DHIT variant.
"""
function load_dhit_data(variant_tag::String)
    csv_path = joinpath(OUT_DIR, "dhit_WCSPH_$(variant_tag)_pp.csv")
    
    if !isfile(csv_path)
        @warn "CSV file not found: $csv_path"
        return nothing
    end
    
    df = CSV.read(csv_path, DataFrame)
    @info "Loaded DHIT data: $variant_tag" size(df)
    return df
end

"""
    load_dhit_json(variant_tag::String) -> Dict

Load JSON metadata for a specific DHIT variant.
"""
function load_dhit_json(variant_tag::String)
    json_path = joinpath(OUT_DIR, "dhit_WCSPH_$(variant_tag)_pp.json")
    
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
    if n >= 2
        decay_rate[1] = -(ke[2] - ke[1]) / (t[2] - t[1])
        decay_rate[n] = -(ke[n] - ke[n-1]) / (t[n] - t[n-1])
    end
    
    return decay_rate
end

"""
    normalize_kinetic_energy(ke)

Normalize kinetic energy by its initial value.
"""
function normalize_kinetic_energy(ke)
    if isempty(ke) || ke[1] ≈ 0.0
        return ones(length(ke))
    end
    return ke ./ ke[1]
end

# ==========================================================================================
# PLOTTING FUNCTIONS
# ==========================================================================================

"""
    plot_kinetic_energy_decay()

Plot kinetic energy evolution for all variants (log scale).
"""
function plot_kinetic_energy_decay()
    plt = plot(xlabel="Time [s]", ylabel="Kinetic Energy [J]",
               title="DHIT: Kinetic Energy Decay",
               yscale=:log10,
               legend=:topright,
               grid=true,
               size=(800, 600),
               dpi=150)
    
    for (tag, label, color, style) in VARIANTS
        df = load_dhit_data(tag)
        if df !== nothing && "kinetic_energy_fluid_1" in names(df)
            plot!(plt, df.time, df.kinetic_energy_fluid_1,
                  label=label, color=color, linestyle=style, linewidth=2)
        end
    end
    
    savefig(plt, joinpath(PLOT_DIR, "dhit_kinetic_energy_decay.png"))
    @info "Saved: dhit_kinetic_energy_decay.png"
    return plt
end

"""
    plot_normalized_ke()

Plot normalized kinetic energy (E/E₀) for all variants.
"""
function plot_normalized_ke()
    plt = plot(xlabel="Time [s]", ylabel="E / E₀",
               title="DHIT: Normalized Kinetic Energy",
               legend=:topright,
               grid=true,
               size=(800, 600),
               dpi=150)
    
    for (tag, label, color, style) in VARIANTS
        df = load_dhit_data(tag)
        if df !== nothing && "kinetic_energy_fluid_1" in names(df)
            ke_norm = normalize_kinetic_energy(df.kinetic_energy_fluid_1)
            plot!(plt, df.time, ke_norm,
                  label=label, color=color, linestyle=style, linewidth=2)
        end
    end
    
    plot!(plt, [0, 2], [1, 1], label="Unity", color=:black, linestyle=:dash, alpha=0.3)
    
    savefig(plt, joinpath(PLOT_DIR, "dhit_normalized_ke.png"))
    @info "Saved: dhit_normalized_ke.png"
    return plt
end

"""
    plot_density_evolution()

Plot average density evolution (compressibility indicator).
"""
function plot_density_evolution()
    plt = plot(xlabel="Time [s]", ylabel="Average Density [kg/m³]",
               title="DHIT: Average Density Evolution",
               legend=:topright,
               grid=true,
               size=(800, 600),
               dpi=150)
    
    for (tag, label, color, style) in VARIANTS
        df = load_dhit_data(tag)
        if df !== nothing && "avg_density_fluid_1" in names(df)
            plot!(plt, df.time, df.avg_density_fluid_1,
                  label=label, color=color, linestyle=style, linewidth=2)
        end
    end
    
    # Reference line at ρ₀ = 1000
    plot!(plt, [0, 2], [1000, 1000], label="ρ₀ = 1000", 
          color=:black, linestyle=:dash, alpha=0.3)
    
    savefig(plt, joinpath(PLOT_DIR, "dhit_avg_density.png"))
    @info "Saved: dhit_avg_density.png"
    return plt
end

"""
    plot_pressure_range()

Plot max and min pressure evolution.
"""
function plot_pressure_range()
    plt = plot(xlabel="Time [s]", ylabel="Pressure [Pa]",
               title="DHIT: Pressure Range Evolution",
               legend=:topright,
               grid=true,
               size=(800, 600),
               dpi=150)
    
    for (tag, label, color, style) in VARIANTS
        df = load_dhit_data(tag)
        if df !== nothing
            has_max = "max_pressure_fluid_1" in names(df)
            has_min = "min_pressure_fluid_1" in names(df)
            
            if has_max
                plot!(plt, df.time, df.max_pressure_fluid_1,
                      label="$label (max)", color=color, 
                      linestyle=style, linewidth=2)
            end
            
            if has_min
                plot!(plt, df.time, df.min_pressure_fluid_1,
                      label="$label (min)", color=color, 
                      linestyle=:solid, linewidth=1, alpha=0.5)
            end
        end
    end
    
    savefig(plt, joinpath(PLOT_DIR, "dhit_pressure_range.png"))
    @info "Saved: dhit_pressure_range.png"
    return plt
end

"""
    plot_ke_decay_rate()

Plot kinetic energy decay rate -dE/dt (dissipation).
"""
function plot_ke_decay_rate()
    plt = plot(xlabel="Time [s]", ylabel="-dE/dt [W]",
               title="DHIT: Energy Dissipation Rate",
               legend=:topright,
               grid=true,
               size=(800, 600),
               dpi=150)
    
    for (tag, label, color, style) in VARIANTS
        df = load_dhit_data(tag)
        if df !== nothing && "kinetic_energy_fluid_1" in names(df)
            decay_rate = compute_ke_decay_rate(df.time, df.kinetic_energy_fluid_1)
            plot!(plt, df.time, decay_rate,
                  label=label, color=color, linestyle=style, linewidth=2)
        end
    end
    
    savefig(plt, joinpath(PLOT_DIR, "dhit_ke_decay_rate.png"))
    @info "Saved: dhit_ke_decay_rate.png"
    return plt
end

"""
    create_dashboard()

Create a 2×2 dashboard of key DHIT metrics.
"""
function create_dashboard()
    # Create 4 subplots
    p1 = plot(xlabel="Time [s]", ylabel="E [J]", 
              title="(a) Kinetic Energy Decay", yscale=:log10,
              legend=:topright, grid=true)
    p2 = plot(xlabel="Time [s]", ylabel="E / E₀",
              title="(b) Normalized Energy",
              legend=:topright, grid=true)
    p3 = plot(xlabel="Time [s]", ylabel="ρ [kg/m³]",
              title="(c) Average Density",
              legend=:topright, grid=true)
    p4 = plot(xlabel="Time [s]", ylabel="-dE/dt [W]",
              title="(d) Dissipation Rate",
              legend=:topright, grid=true)
    
    for (tag, label, color, style) in VARIANTS
        df = load_dhit_data(tag)
        if df !== nothing && "kinetic_energy_fluid_1" in names(df)
            ke = df.kinetic_energy_fluid_1
            ke_norm = normalize_kinetic_energy(ke)
            decay_rate = compute_ke_decay_rate(df.time, ke)
            
            # Plot for each subplot
            plot!(p1, df.time, ke, label=label, color=color, 
                  linestyle=style, linewidth=2)
            plot!(p2, df.time, ke_norm, label=label, color=color,
                  linestyle=style, linewidth=2)
            
            if "avg_density_fluid_1" in names(df)
                plot!(p3, df.time, df.avg_density_fluid_1, label=label,
                      color=color, linestyle=style, linewidth=2)
            end
            
            plot!(p4, df.time, decay_rate, label=label, color=color,
                  linestyle=style, linewidth=2)
        end
    end
    
    # Reference lines
    plot!(p2, [0, 2], [1, 1], label="", color=:black, linestyle=:dash, alpha=0.3)
    plot!(p3, [0, 2], [1000, 1000], label="", color=:black, linestyle=:dash, alpha=0.3)
    
    dashboard = plot(p1, p2, p3, p4, layout=(2,2), size=(1400, 1000), dpi=150)
    savefig(dashboard, joinpath(PLOT_DIR, "dhit_dashboard.png"))
    @info "Saved: dhit_dashboard.png"
    return dashboard
end

# ==========================================================================================
# STATISTICAL SUMMARY
# ==========================================================================================

"""
    print_summary_statistics()

Print statistical summary for all variants.
"""
function print_summary_statistics()
    println("\n" * "="^80)
    println("DHIT POSTPROCESSING: STATISTICAL SUMMARY")
    println("="^80)
    
    for (tag, label, _, _) in VARIANTS
        println("\n$label ($tag)")
        println("-"^80)
        
        df = load_dhit_data(tag)
        if df === nothing
            println("  ⚠ No data available")
            continue
        end
        
        # Time span
        t_start, t_end = extrema(df.time)
        @printf("  Time span: %.4f - %.4f s (Δt = %.4f s)\n", t_start, t_end, t_end - t_start)
        @printf("  Data points: %d\n", nrow(df))
        
        # Kinetic energy decay
        if "kinetic_energy_fluid_1" in names(df)
            ke = df.kinetic_energy_fluid_1
            ke_initial = ke[1]
            ke_final = ke[end]
            ke_decay_pct = 100 * (1 - ke_final / ke_initial)
            
            @printf("  Kinetic Energy:\n")
            @printf("    Initial: %.6e J\n", ke_initial)
            @printf("    Final:   %.6e J\n", ke_final)
            @printf("    Decay:   %.2f%%\n", ke_decay_pct)
            
            # Average decay rate
            avg_decay = mean(compute_ke_decay_rate(df.time, ke))
            @printf("    Avg dissipation: %.6e W\n", avg_decay)
        end
        
        # Density statistics
        if "avg_density_fluid_1" in names(df)
            rho = df.avg_density_fluid_1
            rho_target = 1000.0
            rho_mean = mean(rho)
            rho_std = std(rho)
            rho_error = abs(rho_mean - rho_target) / rho_target * 100
            
            @printf("  Average Density:\n")
            @printf("    Mean:   %.4f kg/m³\n", rho_mean)
            @printf("    Std:    %.4f kg/m³\n", rho_std)
            @printf("    Error:  %.4f%%\n", rho_error)
        end
        
        # Pressure statistics
        if "max_pressure_fluid_1" in names(df) && "min_pressure_fluid_1" in names(df)
            p_max = df.max_pressure_fluid_1
            p_min = df.min_pressure_fluid_1
            
            @printf("  Pressure Range:\n")
            @printf("    Max peak:  %.4e Pa\n", maximum(p_max))
            @printf("    Min trough: %.4e Pa\n", minimum(p_min))
            @printf("    Mean range: %.4e Pa\n", mean(p_max .- p_min))
        end
    end
    
    println("\n" * "="^80)
end

# ==========================================================================================
# MAIN EXECUTION
# ==========================================================================================

function main()
    println("\n" * "="^80)
    println("DHIT POSTPROCESSING")
    println("="^80)
    
    # Generate all plots
    @info "Generating plots..."
    plot_kinetic_energy_decay()
    plot_normalized_ke()
    plot_density_evolution()
    plot_pressure_range()
    plot_ke_decay_rate()
    create_dashboard()
    
    # Print statistics
    print_summary_statistics()
    
    println("\n✓ All plots saved to: $PLOT_DIR")
    println("="^80 * "\n")
end

# Run automatically
main()
