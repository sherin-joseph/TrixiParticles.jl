# ==========================================================================================
# Quick KH Postprocessing - Compact Version
# 
# A simplified script for quick visualization of KH simulation results.
# Automatically detects available variants and creates comparison plots.
# ==========================================================================================

using CSV, DataFrames, Plots, Statistics

# Setup paths
OUT_DIR = raw"C:\Users\Administrator\.julia\TrixiParticles.jl\out"
PLOT_DIR = joinpath(OUT_DIR, "plots")
!isdir(PLOT_DIR) && mkpath(PLOT_DIR)

println("Scanning for KH simulation data in: $OUT_DIR")

# ==========================================================================================
# Auto-detect and load all available KH variants
# ==========================================================================================

kh_files = filter(f -> startswith(f, "kh_WCSPH_") && endswith(f, "_pp.csv"), 
                  readdir(OUT_DIR))

if isempty(kh_files)
    error("No KH postprocessing CSV files found in $OUT_DIR")
end

println("\nFound $(length(kh_files)) KH variant(s):")
data_dict = Dict()

for file in kh_files
    # Extract variant info from filename (e.g., "kh_WCSPH_B_AdamiSGS_pp.csv" -> "B_AdamiSGS")
    variant_name = replace(file, "kh_WCSPH_" => "", "_pp.csv" => "")
    println("  - $variant_name")
    
    try
        df = CSV.read(joinpath(OUT_DIR, file), DataFrame)
        data_dict[variant_name] = df
    catch e
        @warn "Failed to load $file: $e"
    end
end

# ==========================================================================================
# Quick visualization
# ==========================================================================================

# Define colors and line styles
colors = [:blue, :red, :green, :orange, :purple, :brown]
styles = [:solid, :dash, :dot, :dashdot]

# Function to get column safely
function get_column(df, pattern)
    col_idx = findfirst(name -> occursin(pattern, lowercase(name)), names(df))
    return col_idx !== nothing ? names(df)[col_idx] : nothing
end

# Plot 1: Kinetic Energy
p1 = plot(title="Kinetic Energy - KH Variants", 
          xlabel="Time (s)", ylabel="Kinetic Energy",
          legend=:best, size=(900, 600), dpi=150)

for (idx, (name, df)) in enumerate(data_dict)
    ke_col = get_column(df, "kinetic")
    if ke_col !== nothing
        plot!(p1, df.time, df[!, ke_col], 
              label=name, 
              color=colors[mod1(idx, length(colors))],
              lw=2)
    end
end

savefig(p1, joinpath(PLOT_DIR, "kh_quick_ke.png"))
println("\n✓ Saved: kh_quick_ke.png")

# Plot 2: Pressure
p2 = plot(title="Average Pressure - KH Variants",
          xlabel="Time (s)", ylabel="Pressure",
          legend=:best, size=(900, 600), dpi=150)

for (idx, (name, df)) in enumerate(data_dict)
    press_col = get_column(df, "pressure")
    if press_col !== nothing
        plot!(p2, df.time, df[!, press_col],
              label=name,
              color=colors[mod1(idx, length(colors))],
              lw=2)
    end
end

savefig(p2, joinpath(PLOT_DIR, "kh_quick_pressure.png"))
println("✓ Saved: kh_quick_pressure.png")

# Plot 3: Combined view
p3 = plot(p1, p2, layout=(2,1), size=(900, 900))
savefig(p3, joinpath(PLOT_DIR, "kh_quick_combined.png"))
println("✓ Saved: kh_quick_combined.png")

# ==========================================================================================
# Quick statistics
# ==========================================================================================

println("\n" * "="^70)
println("QUICK STATISTICS")
println("="^70)

for (name, df) in data_dict
    ke_col = get_column(df, "kinetic")
    press_col = get_column(df, "pressure")
    
    println("\n$name:")
    println("  Time span: $(df.time[1]) to $(df.time[end]) s")
    
    if ke_col !== nothing
        ke_start = df[1, ke_col]
        ke_end = df[end, ke_col]
        ke_change = ((ke_end - ke_start) / ke_start) * 100
        println("  KE:   start=$(round(ke_start, sigdigits=4))  " *
                "end=$(round(ke_end, sigdigits=4))  " *
                "change=$(round(ke_change, digits=2))%")
    end
    
    if press_col !== nothing
        p_mean = mean(df[!, press_col])
        p_std = std(df[!, press_col])
        println("  Pressure: mean=$(round(p_mean, sigdigits=4))  " *
                "std=$(round(p_std, sigdigits=4))")
    end
end

println("\n" * "="^70)
println("Done! Check plots in: $PLOT_DIR")
println("="^70)
