# ==========================================================================================
# POISEUILLE FLOW QUICK POSTPROCESSING
# ==========================================================================================
# Fast visualization script for Poiseuille flow simulations.
# Auto-detects available variants and generates comparison plots.
#
# Usage:
#   julia poiseuille_quick.jl
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

# Auto-detect available Poiseuille variants
function find_poiseuille_variants()
    variants = []
    pattern = r"poiseuille_WCSPH_([A-D]_.+?)_pp\.csv"
    
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
    println("POISEUILLE FLOW QUICK POSTPROCESSING")
    println("="^80)
    
    variants = find_poiseuille_variants()
    
    if isempty(variants)
        @warn "No Poiseuille data files found in $OUT_DIR"
        println("Expected pattern: poiseuille_WCSPH_*_pp.csv")
        return
    end
    
    @info "Found variants:" variants
    
    # === PLOT 1: Fluid Kinetic Energy ===
    p1 = plot(xlabel="Time [s]", ylabel="Kinetic Energy [J]",
              title="Poiseuille Flow: Fluid Kinetic Energy",
              legend=:topright,
              grid=true,
              size=(900, 600),
              dpi=150)
    
    for tag in variants
        csv_path = joinpath(OUT_DIR, "poiseuille_WCSPH_$(tag)_pp.csv")
        df = CSV.read(csv_path, DataFrame)
        
        ke = get_column(df, ["kinetic_energy_fluid_1", "kinetic_energy"])
        if ke !== nothing
            color, style = variant_style(tag)
            plot!(p1, df.time, ke, label=tag, color=color, 
                  linestyle=style, linewidth=2)
        end
    end
    
    savefig(p1, joinpath(PLOT_DIR, "poiseuille_quick_kinetic_energy.png"))
    @info "Saved: poiseuille_quick_kinetic_energy.png"
    
    # === PLOT 2: Average Pressure (Fluid) ===
    p2 = plot(xlabel="Time [s]", ylabel="Average Pressure [Pa]",
              title="Poiseuille Flow: Average Fluid Pressure",
              legend=:topright,
              grid=true,
              size=(900, 600),
              dpi=150)
    
    for tag in variants
        csv_path = joinpath(OUT_DIR, "poiseuille_WCSPH_$(tag)_pp.csv")
        df = CSV.read(csv_path, DataFrame)
        
        p_avg = get_column(df, ["avg_pressure_fluid_1", "avg_pressure"])
        if p_avg !== nothing
            color, style = variant_style(tag)
            plot!(p2, df.time, p_avg, label=tag, color=color,
                  linestyle=style, linewidth=2)
        end
    end
    
    savefig(p2, joinpath(PLOT_DIR, "poiseuille_quick_pressure.png"))
    @info "Saved: poiseuille_quick_pressure.png"
    
    # === PLOT 3: Open Boundary Kinetic Energy ===
    p3 = plot(xlabel="Time [s]", ylabel="Kinetic Energy [J]",
              title="Poiseuille Flow: Open Boundary Kinetic Energy",
              legend=:topright,
              grid=true,
              size=(900, 600),
              dpi=150)
    
    for tag in variants
        csv_path = joinpath(OUT_DIR, "poiseuille_WCSPH_$(tag)_pp.csv")
        df = CSV.read(csv_path, DataFrame)
        
        ke_ob = get_column(df, ["kinetic_energy_open_boundary_1"])
        if ke_ob !== nothing
            color, style = variant_style(tag)
            plot!(p3, df.time, ke_ob, label=tag, color=color,
                  linestyle=style, linewidth=2)
        end
    end
    
    savefig(p3, joinpath(PLOT_DIR, "poiseuille_quick_open_boundary_ke.png"))
    @info "Saved: poiseuille_quick_open_boundary_ke.png"
    
    # === PLOT 4: Combined Dashboard (2×2) ===
    sp1 = plot(xlabel="Time", ylabel="KE [J]", title="Fluid Kinetic Energy",
               legend=:topright, grid=true)
    sp2 = plot(xlabel="Time", ylabel="P [Pa]", title="Avg Fluid Pressure",
               legend=:topright, grid=true)
    sp3 = plot(xlabel="Time", ylabel="KE [J]", title="Open Boundary KE",
               legend=:topright, grid=true)
    sp4 = plot(xlabel="Time", ylabel="P [Pa]", title="Open Boundary Pressure",
               legend=:topright, grid=true)
    
    for tag in variants
        csv_path = joinpath(OUT_DIR, "poiseuille_WCSPH_$(tag)_pp.csv")
        df = CSV.read(csv_path, DataFrame)
        color, style = variant_style(tag)
        
        # Fluid KE
        ke = get_column(df, ["kinetic_energy_fluid_1", "kinetic_energy"])
        if ke !== nothing
            plot!(sp1, df.time, ke, label=tag, color=color,
                  linestyle=style, linewidth=2)
        end
        
        # Fluid Pressure
        p_avg = get_column(df, ["avg_pressure_fluid_1", "avg_pressure"])
        if p_avg !== nothing
            plot!(sp2, df.time, p_avg, label=tag, color=color,
                  linestyle=style, linewidth=2)
        end
        
        # Open Boundary KE
        ke_ob = get_column(df, ["kinetic_energy_open_boundary_1"])
        if ke_ob !== nothing
            plot!(sp3, df.time, ke_ob, label=tag, color=color,
                  linestyle=style, linewidth=2)
        end
        
        # Open Boundary Pressure
        p_ob = get_column(df, ["avg_pressure_open_boundary_1"])
        if p_ob !== nothing
            plot!(sp4, df.time, p_ob, label=tag, color=color,
                  linestyle=style, linewidth=2)
        end
    end
    
    dashboard = plot(sp1, sp2, sp3, sp4, layout=(2,2), size=(1400, 1000), dpi=150)
    savefig(dashboard, joinpath(PLOT_DIR, "poiseuille_quick_dashboard.png"))
    @info "Saved: poiseuille_quick_dashboard.png"
    
    # === Print Quick Stats ===
    println("\n" * "-"^80)
    println("QUICK STATISTICS")
    println("-"^80)
    
    for tag in variants
        csv_path = joinpath(OUT_DIR, "poiseuille_WCSPH_$(tag)_pp.csv")
        df = CSV.read(csv_path, DataFrame)
        
        println("\n$tag:")
        
        ke = get_column(df, ["kinetic_energy_fluid_1", "kinetic_energy"])
        if ke !== nothing
            ke_final = ke[end]
            ke_mean = mean(ke[max(1, end-10):end])  # Last 10 points average
            println("  Final KE: $(round(ke_final, sigdigits=4)) J")
            println("  Mean KE (last 10): $(round(ke_mean, sigdigits=4)) J")
        end
        
        p_avg = get_column(df, ["avg_pressure_fluid_1", "avg_pressure"])
        if p_avg !== nothing
            p_mean = mean(p_avg[max(1, end-10):end])
            println("  Mean pressure (last 10): $(round(p_mean, sigdigits=4)) Pa")
        end
        
        ke_ob = get_column(df, ["kinetic_energy_open_boundary_1"])
        if ke_ob !== nothing
            ke_ob_final = ke_ob[end]
            println("  Final open boundary KE: $(round(ke_ob_final, sigdigits=4)) J")
        end
    end
    
    println("\n" * "="^80)
    println("✓ All plots saved to: $PLOT_DIR")
    println("="^80 * "\n")
end

# Run automatically
quick_plots()
