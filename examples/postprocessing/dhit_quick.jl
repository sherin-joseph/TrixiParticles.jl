# ==========================================================================================
# DHIT QUICK POSTPROCESSING
# ==========================================================================================
# Fast visualization script for DHIT simulations.
# Auto-detects available variants and generates comparison plots.
#
# Usage:
#   julia dhit_quick.jl
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

# Auto-detect available DHIT variants
function find_dhit_variants()
    variants = []
    pattern = r"dhit_WCSPH_([A-D]_.+?)_pp\.csv"
    
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
    println("DHIT QUICK POSTPROCESSING")
    println("="^80)
    
    variants = find_dhit_variants()
    
    if isempty(variants)
        @warn "No DHIT data files found in $OUT_DIR"
        println("Expected pattern: dhit_WCSPH_*_pp.csv")
        return
    end
    
    @info "Found variants:" variants
    
    # === PLOT 1: Kinetic Energy Decay (log scale) ===
    p1 = plot(xlabel="Time [s]", ylabel="Kinetic Energy [J]",
              title="DHIT: Kinetic Energy Decay (Log Scale)",
              yscale=:log10,
              legend=:topright,
              grid=true,
              size=(900, 600),
              dpi=150)
    
    for tag in variants
        csv_path = joinpath(OUT_DIR, "dhit_WCSPH_$(tag)_pp.csv")
        df = CSV.read(csv_path, DataFrame)
        
        ke = get_column(df, ["kinetic_energy_fluid_1", "kinetic_energy"])
        if ke !== nothing
            color, style = variant_style(tag)
            plot!(p1, df.time, ke, label=tag, color=color, 
                  linestyle=style, linewidth=2)
        end
    end
    
    savefig(p1, joinpath(PLOT_DIR, "dhit_quick_ke_decay.png"))
    @info "Saved: dhit_quick_ke_decay.png"
    
    # === PLOT 2: Average Density (compressibility) ===
    p2 = plot(xlabel="Time [s]", ylabel="Average Density [kg/m³]",
              title="DHIT: Average Density Evolution",
              legend=:topright,
              grid=true,
              size=(900, 600),
              dpi=150)
    
    for tag in variants
        csv_path = joinpath(OUT_DIR, "dhit_WCSPH_$(tag)_pp.csv")
        df = CSV.read(csv_path, DataFrame)
        
        rho = get_column(df, ["avg_density_fluid_1", "avg_density"])
        if rho !== nothing
            color, style = variant_style(tag)
            plot!(p2, df.time, rho, label=tag, color=color,
                  linestyle=style, linewidth=2)
        end
    end
    
    # Reference line at ρ₀ = 1000 (use explicit time range)
    if !isempty(variants)
        # Get time range from first variant for reference line
        csv_path = joinpath(OUT_DIR, "dhit_WCSPH_$(variants[1])_pp.csv")
        df_ref = CSV.read(csv_path, DataFrame)
        plot!(p2, [df_ref.time[1], df_ref.time[end]], [1000, 1000], label="ρ₀ = 1000",
              color=:black, linestyle=:dash, alpha=0.3, linewidth=1)
    end
    
    savefig(p2, joinpath(PLOT_DIR, "dhit_quick_density.png"))
    @info "Saved: dhit_quick_density.png"
    
    # === PLOT 3: Combined Dashboard (2×2) ===
    sp1 = plot(xlabel="Time", ylabel="KE [J]", title="Kinetic Energy",
               yscale=:log10, legend=:topright, grid=true)
    sp2 = plot(xlabel="Time", ylabel="E/E₀", title="Normalized KE",
               legend=:topright, grid=true)
    sp3 = plot(xlabel="Time", ylabel="ρ [kg/m³]", title="Avg Density",
               legend=:topright, grid=true)
    sp4 = plot(xlabel="Time", ylabel="P [Pa]", title="Pressure Range",
               legend=:topright, grid=true)
    
    for tag in variants
        csv_path = joinpath(OUT_DIR, "dhit_WCSPH_$(tag)_pp.csv")
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
        
        # Density
        rho = get_column(df, ["avg_density_fluid_1", "avg_density"])
        if rho !== nothing
            plot!(sp3, df.time, rho, label=tag, color=color,
                  linestyle=style, linewidth=2)
        end
        
        # Pressure range
        p_max = get_column(df, ["max_pressure_fluid_1", "max_pressure"])
        p_min = get_column(df, ["min_pressure_fluid_1", "min_pressure"])
        if p_max !== nothing && p_min !== nothing
            plot!(sp4, df.time, p_max, label="$tag (max)", color=color,
                  linestyle=style, linewidth=2)
            plot!(sp4, df.time, p_min, label="", color=color,
                  linestyle=:solid, linewidth=1, alpha=0.5)
        end
    end
    
    # Reference lines
    plot!(sp2, [0, 2], [1, 1], label="", color=:black, linestyle=:dash, alpha=0.3)
    plot!(sp3, [0, 2], [1000, 1000], label="", color=:black, linestyle=:dash, alpha=0.3)
    
    dashboard = plot(sp1, sp2, sp3, sp4, layout=(2,2), size=(1400, 1000), dpi=150)
    savefig(dashboard, joinpath(PLOT_DIR, "dhit_quick_dashboard.png"))
    @info "Saved: dhit_quick_dashboard.png"
    
    # === Print Quick Stats ===
    println("\n" * "-"^80)
    println("QUICK STATISTICS")
    println("-"^80)
    
    for tag in variants
        csv_path = joinpath(OUT_DIR, "dhit_WCSPH_$(tag)_pp.csv")
        df = CSV.read(csv_path, DataFrame)
        
        println("\n$tag:")
        
        ke = get_column(df, ["kinetic_energy_fluid_1", "kinetic_energy"])
        if ke !== nothing
            ke_decay_pct = 100 * (1 - ke[end] / ke[1])
            println("  KE decay: $(round(ke_decay_pct, digits=2))%")
        end
        
        rho = get_column(df, ["avg_density_fluid_1", "avg_density"])
        if rho !== nothing
            rho_mean = mean(rho)
            rho_err = abs(rho_mean - 1000) / 1000 * 100
            println("  Avg density: $(round(rho_mean, digits=2)) kg/m³ (error: $(round(rho_err, digits=4))%)")
        end
        
        p_max = get_column(df, ["max_pressure_fluid_1", "max_pressure"])
        p_min = get_column(df, ["min_pressure_fluid_1", "min_pressure"])
        if p_max !== nothing && p_min !== nothing
            p_range = mean(p_max .- p_min)
            println("  Avg pressure range: $(round(p_range, sigdigits=4)) Pa")
        end
    end
    
    println("\n" * "="^80)
    println("✓ All plots saved to: $PLOT_DIR")
    println("="^80 * "\n")
end

# Run automatically
quick_plots()
