# DHIT (Decaying Homogeneous Isotropic Turbulence) Postprocessing

This directory contains postprocessing scripts for analyzing 3D DHIT (Decaying Homogeneous Isotropic Turbulence) simulations comparing different SGS models and stabilization techniques.

## Overview

The DHIT test case simulates freely-decaying turbulence in a periodic cubic domain with an initial divergence-free random velocity field. Four variants are tested:

- **Variant A**: Baseline stabilization, no SGS model (`A_noSGS`)
- **Variant B**: Baseline stabilization + AdamiSGS model (`B_AdamiSGS`)
- **Variant C**: Reduced stabilization, no SGS model (`C_noSGS`)
- **Variant D**: Reduced stabilization + AdamiSGS model (`D_AdamiSGS`)

## Scripts

### 1. `dhit_postprocessing.jl` (Comprehensive Analysis)

**Purpose**: Full statistical analysis and publication-quality plots.

**Features**:
- Kinetic energy decay curves (log scale)
- Normalized energy evolution (E/E₀)
- Average density tracking (compressibility indicator)
- Pressure range evolution (max/min)
- Energy dissipation rate (-dE/dt)
- Comprehensive 2×2 dashboard
- Detailed statistical summaries

**Usage**:
```julia
include("examples/postprocessing/dhit_postprocessing.jl")
```

**Outputs** (saved to `out/plots/`):
- `dhit_kinetic_energy_decay.png` - KE decay on log scale
- `dhit_normalized_ke.png` - Normalized energy E/E₀
- `dhit_avg_density.png` - Average density evolution
- `dhit_pressure_range.png` - Max/min pressure bounds
- `dhit_ke_decay_rate.png` - Dissipation rate
- `dhit_dashboard.png` - Combined 2×2 overview

---

### 2. `dhit_quick.jl` (Rapid Visualization)

**Purpose**: Fast comparison plots with auto-detection.

**Features**:
- Automatic variant detection (scans `out/` directory)
- Quick KE decay plot (log scale)
- Density evolution comparison
- Combined 2×2 dashboard
- Basic statistics summary

**Usage**:
```julia
include("examples/postprocessing/dhit_quick.jl")
```

**Outputs** (saved to `out/plots/`):
- `dhit_quick_ke_decay.png`
- `dhit_quick_density.png`
- `dhit_quick_dashboard.png`

**Advantages**:
- No configuration needed
- Automatically finds available variants
- Handles missing data gracefully

---

### 3. `dhit_advanced.jl` (Advanced VTK Analysis)

**Purpose**: 3D particle visualization and turbulence statistics from VTU snapshot files.

**Features**:
- 3D particle scatter plots colored by velocity magnitude
- Velocity magnitude histogram evolution
- Velocity phase space plots (Vx-Vy-Vz projections)
- Turbulence statistics (RMS velocity, TKE)
- Time evolution of turbulent quantities

**Requirements**:
- `ReadVTK.jl` package (install: `using Pkg; Pkg.add("ReadVTK")`)

**Usage**:
```julia
using Pkg
Pkg.add("ReadVTK")  # If not already installed

include("examples/postprocessing/dhit_advanced.jl")
```

**Outputs** (saved to `out/plots_advanced/`):
- `dhit_*_velocity_hist.png` - Velocity distribution evolution
- `dhit_*_3d_step*.png` - 3D particle snapshots
- `dhit_*_phase_space_step*.png` - Velocity component correlations
- `dhit_*_turbulence_stats.png` - RMS velocity and TKE evolution

**Note**: If `ReadVTK` is not available, the script will skip VTK analysis but still run CSV-based processing.

---

## Data Files

### Expected Input Files

The scripts expect the following files in the `out/` directory:

#### Time Series Data (CSV + JSON):
```
out/
├── dhit_WCSPH_A_noSGS_pp.csv
├── dhit_WCSPH_A_noSGS_pp.json
├── dhit_WCSPH_B_AdamiSGS_pp.csv
├── dhit_WCSPH_B_AdamiSGS_pp.json
├── dhit_WCSPH_C_noSGS_pp.csv
├── dhit_WCSPH_C_noSGS_pp.json
├── dhit_WCSPH_D_AdamiSGS_pp.csv
└── dhit_WCSPH_D_AdamiSGS_pp.json
```

#### VTK Snapshots (for advanced analysis):
```
out/
├── dhit_WCSPH_A_noSGS_fluid_1_0.vtu
├── dhit_WCSPH_A_noSGS_fluid_1_10.vtu
├── dhit_WCSPH_A_noSGS_fluid_1_20.vtu
└── ... (and so on for other variants)
```

### CSV Data Structure

Expected columns in CSV files:
- `time` - Simulation time [s]
- `kinetic_energy_fluid_1` - Total kinetic energy [J]
- `avg_density_fluid_1` - Average fluid density [kg/m³]
- `max_pressure_fluid_1` - Maximum pressure [Pa]
- `min_pressure_fluid_1` - Minimum pressure [Pa]

### JSON Metadata

Contains simulation parameters:
- Viscosity model (ν, AdamiSGS parameters)
- Domain size and resolution
- Time integration settings
- Physical parameters (ρ₀, c₀)

---

## Key Metrics & Interpretation

### 1. **Kinetic Energy Decay**

**What it shows**: Total kinetic energy dissipation over time

**Expected behavior**:
- **Power-law decay**: $E(t) \propto t^{-n}$ for fully developed turbulence
- **Variants with SGS**: Faster dissipation (more effective at small scales)
- **Reduced stabilization**: May show oscillations or instabilities

**Interpretation**:
```julia
# Typical decay rate: ~40-60% over 2 turnover times
KE_decay = 100 * (1 - E_final / E_initial)
```

---

### 2. **Average Density**

**What it shows**: Measure of compressibility / numerical errors

**Expected behavior**:
- **Should remain close to ρ₀ = 1000 kg/m³**
- Small oscillations (<1%) are acceptable
- Large deviations indicate:
  - Insufficient density diffusion
  - Acoustic wave reflections
  - Instability

**Interpretation**:
```julia
ρ_error = |mean(ρ) - ρ₀| / ρ₀ * 100  # Should be < 0.5%
```

---

### 3. **Pressure Range**

**What it shows**: Pressure fluctuations in the flow

**Expected behavior**:
- Reflects turbulent velocity fluctuations
- Range decreases as turbulence decays
- Max/min should remain bounded

**Red flags**:
- Unbounded growth → numerical instability
- Negative pressures → tensile instability

---

### 4. **Energy Dissipation Rate**

**What it shows**: $\epsilon = -dE/dt$ (rate of energy loss)

**Expected behavior**:
- **Initially high** (dissipating large eddies)
- **Decays over time** as turbulence weakens
- **SGS models** may show enhanced dissipation

**Theoretical connection**:
```
ε ≈ ν * ⟨(∂u/∂x)²⟩  (viscous dissipation)
```

---

### 5. **Turbulence Statistics** (from `dhit_advanced.jl`)

#### **RMS Velocity** ($v_{rms}$):
- Measure of turbulence intensity
- $v_{rms}^2 = \frac{1}{3}(\overline{u'^2} + \overline{v'^2} + \overline{w'^2})$
- Should decay monotonically

#### **Turbulent Kinetic Energy** (TKE):
- $k = \frac{1}{2}(\overline{u'^2} + \overline{v'^2} + \overline{w'^2})$
- Proportional to $v_{rms}^2$

#### **Isotropy Check**:
- For isotropic turbulence: $\overline{u'^2} \approx \overline{v'^2} \approx \overline{w'^2}$
- Check via velocity phase space plots

---

## Troubleshooting

### No CSV Files Found

**Problem**: Scripts report "No DHIT data files found"

**Solutions**:
1. Run the DHIT simulation first:
   ```julia
   include("examples/experimental examples(cheri)/3D Decaying Homogeneous Isotropic Turbulence.jl")
   ```

2. Check simulation completed:
   ```julia
   readdir(joinpath(@__DIR__, "..", "..", "out"))
   ```

3. Verify postprocessing callback is enabled in simulation script

---

### VTK Files Not Loading

**Problem**: `dhit_advanced.jl` skips VTK analysis or throws errors

**Solutions**:
1. Install ReadVTK:
   ```julia
   using Pkg
   Pkg.add("ReadVTK")
   ```

2. Verify VTU files exist:
   ```julia
   # Should show dhit_WCSPH_*_fluid_1_*.vtu files
   readdir(joinpath(@__DIR__, "..", "..", "out"))
   ```

3. Check file naming convention matches pattern:
   ```
   dhit_WCSPH_{VARIANT}_fluid_1_{STEP}.vtu
   ```

---

### Missing Data Columns

**Problem**: "Column X not found in DataFrame"

**Solutions**:
1. Update simulation to use `PostprocessCallback` with all required metrics:
   ```julia
   postprocess_callback = PostprocessCallback(
       dt=0.01,
       kinetic_energy=kinetic_energy,
       avg_density=avg_density,
       max_pressure=max_pressure,
       min_pressure=min_pressure
   )
   ```

2. Check CSV file headers match expected columns

3. Regenerate data by re-running simulation

---

### Memory Issues (Advanced Script)

**Problem**: `dhit_advanced.jl` runs out of memory with large VTU files

**Solutions**:
1. Increase subsampling in 3D plots:
   ```julia
   plot_3d_particle_snapshot(tag, step; subsample=10)  # Was 5
   ```

2. Analyze fewer time steps:
   ```julia
   snapshot_steps = [0, 20, 40]  # Instead of [0, 10, 20, 30, 40]
   ```

3. Process one variant at a time

---

## Customization Examples

### Change Time Range
```julia
# In dhit_postprocessing.jl, modify plotting functions:
plot!(plt, df.time, df.kinetic_energy_fluid_1,
      xlims=(0.0, 1.0))  # Focus on first second
```

### Add New Metrics
```julia
# Define custom metric
function compute_enstrophy(df)
    # Your calculation here
    return enstrophy_values
end

# Add to plotting
plot!(plt, df.time, compute_enstrophy(df), label="Enstrophy")
```

### Export Data to CSV
```julia
# After analysis, save processed data
using CSV
results = DataFrame(
    time = df.time,
    ke_normalized = normalize_kinetic_energy(df.kinetic_energy_fluid_1)
)
CSV.write("dhit_processed.csv", results)
```

---

## References

### DHIT Theory
- **Pope, S. B.** (2000). *Turbulent Flows*. Cambridge University Press.
- **Sagaut, P.** (2006). *Large Eddy Simulation for Incompressible Flows*. Springer.

### SPH Turbulence Modeling
- **Adami et al.** (2013). "A transport-velocity formulation for smoothed particle hydrodynamics."
- **Morris et al.** (1997). "Modeling Low Reynolds Number Incompressible Flows Using SPH."

---

## Support

If you encounter issues:

1. **Check simulation logs** for errors during DHIT simulation
2. **Verify data files** exist and are not empty/corrupted
3. **Update packages**: `using Pkg; Pkg.update()`
4. **Review simulation parameters** in JSON metadata files

---

## Status

**Current**: ✅ Scripts created and documented

**Note**: These scripts are ready to use once DHIT simulations are run. If CSV files don't exist yet, run the DHIT simulation first to generate the required output data.

**To run DHIT simulations**:
```julia
# In Julia REPL, from TrixiParticles.jl root:
include("examples/experimental examples(cheri)/3D Decaying Homogeneous Isotropic Turbulence.jl")
```

This will generate all required CSV, JSON, and VTU files for postprocessing.
