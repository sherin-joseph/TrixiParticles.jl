# ==========================================================================================
# TGV QUICK POSTPROCESSING
# ==========================================================================================
# Fast visualization script for Taylor-Green Vortex simulations.
# Auto-detects available variants and generates comparison plots.
#
# Usage:
#   julia tgv_quick.jl
#
# Requirements: CSV, DataFrames, Plots, Statistics
# ==========================================================================================

using CSV, DataFrames, Plots, Statistics

# Configuration
const OUT_DIR = joinpath(@__DIR__, "..", "..", "out")
const PLOT_DIR = joinpath(OUT_DIR, "plots")
mkpath(PLOT_DIR)

# Helper function to get column safely
function get_column(df, names_to_try)
    for name in names_to_try
        if name in names(df)
            return df[!, name]
        end
    end
    return nothing
end

# Auto-detect available TGV variants
function find_tgv_variants()
    variants = []
    pattern = r"tgv_WCSPH_([A-D]_.+?)_pp\.csv"
    
    for file in readdir(OUT_DIR)
        m = match(pattern, file)
        if m !== nothing
            tag = m.captures[1]
            push!(variants, tag)
        end
    end
    
    return sort(variants)
end

# Color and style mapping
function variant_style(tag)
    colors = Dict("A" => :blue, "B" => :red, "C" => :green, "D" => :orange)
    styles = Dict("noSGS" => :solid, "AdamiSGS" => :dash, "MorrisSGS" => :dashdot)
    
    variant = tag[1:1]  # First character (A, B, C, or D)
    color = get(colors, variant, :black)
    
    style = :solid
    if occursin("AdamiSGS", tag)
        style = :dash
    elseif occursin("MorrisSGS", tag)
        style = :dashdot
    end
    
    return color, style
end

# Main plotting function
function quick_plots()
    println("\n" * "="^80)
    println("TGV QUICK POSTPROCESSING")
    println("="^80)
    
    variants = find_tgv_variants()
    
    if isempty(variants)
        @warn "No TGV data files found in $OUT_DIR"
        println("Expected pattern: tgv_WCSPH_*_pp.csv")
        return
    end
    
    @info "Found variants:" variants
    
    # === PLOT 1: Kinetic Energy Decay (log scale) ===
    p1 = plot(xlabel="Time [s]", ylabel="Kinetic Energy [J]",
              title="TGV: Kinetic Energy Decay (Log Scale)",
              yscale=:log10,
              legend=:topright,
              grid=true,
              size=(900, 600),
              dpi=150)
    
    for tag in variants
        csv_path = joinpath(OUT_DIR, "tgv_WCSPH_$(tag)_pp.csv")
        df = CSV.read(csv_path, DataFrame)
        
        ke = get_column(df, ["kinetic_energy_fluid_1", "kinetic_energy"])
        if ke !== nothing
            color, style = variant_style(tag)
            plot!(p1, df.time, ke, label=tag, color=color, 
                  linestyle=style, linewidth=2)
        end
    end
    
    savefig(p1, joinpath(PLOT_DIR, "tgv_quick_ke_decay.png"))
    @info "Saved: tgv_quick_ke_decay.png"
    
    # === PLOT 2: Average Pressure Evolution ===
    p2 = plot(xlabel="Time [s]", ylabel="Average Pressure [Pa]",
              title="TGV: Average Pressure Evolution",
              legend=:topright,
              grid=true,
              size=(900, 600),
              dpi=150)
    
    for tag in variants
        csv_path = joinpath(OUT_DIR, "tgv_WCSPH_$(tag)_pp.csv")
        df = CSV.read(csv_path, DataFrame)
        
        p_avg = get_column(df, ["avg_pressure_fluid_1", "avg_pressure"])
        if p_avg !== nothing
            color, style = variant_style(tag)
            plot!(p2, df.time, p_avg, label=tag, color=color,
                  linestyle=style, linewidth=2)
        end
    end
    
    # Reference line at P = 0
    if !isempty(variants)
        csv_path = joinpath(OUT_DIR, "tgv_WCSPH_$(variants[1])_pp.csv")
        df_ref = CSV.read(csv_path, DataFrame)
        plot!(p2, [df_ref.time[1], df_ref.time[end]], [0, 0], label="P₀ = 0",
              color=:black, linestyle=:dash, alpha=0.3, linewidth=1)
    end
    
    savefig(p2, joinpath(PLOT_DIR, "tgv_quick_pressure.png"))
    @info "Saved: tgv_quick_pressure.png"
    
    # === PLOT 3: Combined Dashboard (2×1) ===
    sp1 = plot(xlabel="Time", ylabel="KE [J]", title="Kinetic Energy",
               yscale=:log10, legend=:topright, grid=true)
    sp2 = plot(xlabel="Time", ylabel="E/E₀", title="Normalized KE",
               legend=:topright, grid=true)
    
    for tag in variants
        csv_path = joinpath(OUT_DIR, "tgv_WCSPH_$(tag)_pp.csv")
        df = CSV.read(csv_path, DataFrame)
        color, style = variant_style(tag)
        
        # KE decay
        ke = get_column(df, ["kinetic_energy_fluid_1", "kinetic_energy"])
        if ke !== nothing
            plot!(sp1, df.time, ke, label=tag, color=color,
                  linestyle=style, linewidth=2)
            
            # Normalized KE
            ke_norm = ke ./ ke[1]
            plot!(sp2, df.time, ke_norm, label=tag, color=color,
                  linestyle=style, linewidth=2)
        end
    end
    
    # Reference line for normalized KE
    plot!(sp2, [0, 10], [1, 1], label="", color=:black, linestyle=:dash, alpha=0.3)
    
    dashboard = plot(sp1, sp2, layout=(2,1), size=(900, 800), dpi=150)
    savefig(dashboard, joinpath(PLOT_DIR, "tgv_quick_dashboard.png"))
    @info "Saved: tgv_quick_dashboard.png"
    
    # === Print Quick Stats ===
    println("\n" * "-"^80)
    println("QUICK STATISTICS")
    println("-"^80)
    
    for tag in variants
        csv_path = joinpath(OUT_DIR, "tgv_WCSPH_$(tag)_pp.csv")
        df = CSV.read(csv_path, DataFrame)
        
        println("\n$tag:")
        
        ke = get_column(df, ["kinetic_energy_fluid_1", "kinetic_energy"])
        if ke !== nothing
            ke_decay_pct = 100 * (1 - ke[end] / ke[1])
            println("  KE decay: $(round(ke_decay_pct, digits=2))%")
        end
        
        p_avg = get_column(df, ["avg_pressure_fluid_1", "avg_pressure"])
        if p_avg !== nothing
            p_mean = mean(p_avg)
            println("  Avg pressure: $(round(p_mean, digits=4)) Pa")
        end
    end
    
    println("\n" * "="^80)
    println("✓ All plots saved to: $PLOT_DIR")
    println("="^80 * "\n")
end

# Run automatically
quick_plots()
