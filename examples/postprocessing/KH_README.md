# Kelvin-Helmholtz Instability Postprocessing Scripts

This directory contains postprocessing scripts for analyzing Kelvin-Helmholtz (KH) instability simulations from TrixiParticles.jl.

## Overview

The KH simulation produces results for 4 variants:
- **Variant A**: Baseline with stabilization, no SGS model
- **Variant B**: Baseline with stabilization, with Adami SGS model
- **Variant C**: Reduced stabilization, no SGS model
- **Variant D**: Reduced stabilization, with Adami SGS model

## Scripts

### 1. `kh_postprocessing.jl` - Main Comprehensive Analysis

The main postprocessing script that generates detailed visualizations and statistics.

**Features:**
- Kinetic energy evolution for all variants
- Normalized kinetic energy (decay analysis)
- Average pressure evolution
- Combined dashboard with multiple comparisons
- Statistical summary

**Usage:**
```julia
julia> include("examples/postprocessing/kh_postprocessing.jl")
```

**Outputs:**
- `plots/kh_kinetic_energy_all.png` - KE evolution comparison
- `plots/kh_normalized_ke.png` - Normalized KE (shows decay)
- `plots/kh_avg_pressure_all.png` - Pressure evolution
- `plots/kh_dashboard.png` - 2x2 comparison dashboard

### 2. `kh_quick.jl` - Quick Analysis

A simplified script for rapid visualization without detailed configuration.

**Features:**
- Automatic detection of available variants
- Quick KE and pressure plots
- Basic statistics

**Usage:**
```julia
julia> include("examples/postprocessing/kh_quick.jl")
```

**Outputs:**
- `plots/kh_quick_ke.png`
- `plots/kh_quick_pressure.png`
- `plots/kh_quick_combined.png`

### 3. `kh_advanced.jl` - Advanced Visualization

Advanced analysis including VTK support and phase space plots.

**Features:**
- Phase space plots (KE vs Pressure)
- Dissipation rate analysis (dKE/dt)
- Log-scale kinetic energy plots
- VTK snapshot visualization (requires ReadVTK.jl)
- Particle distribution plots

**Requirements:**
```julia
# Optional for VTK support
using Pkg
Pkg.add("ReadVTK")
```

**Usage:**
```julia
julia> include("examples/postprocessing/kh_advanced.jl")
```

**Outputs:**
- `plots_advanced/kh_phase_space.png`
- `plots_advanced/kh_dissipation_rate.png`
- `plots_advanced/kh_log_ke.png`
- `plots_advanced/kh_snapshot_*.png` (if VTK available)

## Data Files

The scripts expect the following files in the `out/` directory:

### CSV Files (Time Series Data)
- `kh_WCSPH_A_noSGS_pp.csv`
- `kh_WCSPH_B_AdamiSGS_pp.csv`
- `kh_WCSPH_C_noSGS_pp.csv`
- `kh_WCSPH_D_AdamiSGS_pp.csv`

**Columns:**
- `time` - Simulation time
- `kinetic_energy_fluid_1` - Total kinetic energy
- `avg_pressure_fluid_1` - Average pressure

### JSON Files (Metadata + Data)
- `kh_WCSPH_*_pp.json` - Contains metadata and time series data

### VTU Files (Particle Snapshots)
- `kh_WCSPH_*_fluid_1_*.vtu` - VTK files for particle visualization
- Can be viewed in ParaView or loaded with ReadVTK.jl

## Key Metrics

### Kinetic Energy
- **Initial Value**: Energy at t=0
- **Decay Rate**: How quickly energy dissipates
- **Final Value**: Energy at final time
- **Comparison**: SGS models should show different dissipation

### Average Pressure
- **Mean**: Time-averaged pressure
- **Fluctuations**: Standard deviation indicates instability growth

## Interpretation

### Comparing Variants A vs C (Stabilization Effect)
- A has stabilization ON, C has it OFF
- Compare their energy dissipation rates
- Stability should differ

### Comparing noSGS vs AdamiSGS (SGS Model Effect)
- AdamiSGS adds subgrid-scale turbulence modeling
- Should affect small-scale energy cascade
- Look for differences in dissipation patterns

### Phase Space (Advanced)
- KE vs Pressure trajectory shows system evolution
- Closed loops indicate periodic behavior
- Spirals indicate damped oscillations

## Example Workflow

1. **Run simulations** (from main KH script):
   ```julia
   include("examples/experimental examples(cheri)/Kelvin–Helmholtz Roll-up (Fully Periodic).jl")
   ```

2. **Quick check**:
   ```julia
   include("examples/postprocessing/kh_quick.jl")
   ```

3. **Detailed analysis**:
   ```julia
   include("examples/postprocessing/kh_postprocessing.jl")
   ```

4. **Advanced features**:
   ```julia
   include("examples/postprocessing/kh_advanced.jl")
   ```

## Customization

### Change Output Directory
Edit the `OUT_DIR` constant in any script:
```julia
const OUT_DIR = raw"C:\path\to\your\output"
```

### Select Specific Variants
In `kh_postprocessing.jl`, modify the `variants` list:
```julia
variants = [
    ("A", "noSGS"),
    ("B", "AdamiSGS"),
    # Comment out variants you don't want to plot
]
```

### Adjust Plot Appearance
Modify plot parameters:
```julia
plot(..., 
     size=(1200, 800),    # Figure size
     linewidth=3,         # Line thickness
     dpi=300,            # Resolution
     legend=:topright)   # Legend position
```

## Troubleshooting

### "No KH data found"
- Check that simulation output files exist in `OUT_DIR`
- Verify file naming matches expected pattern
- Run the simulation first

### "ReadVTK not available"
- VTK visualization is optional
- Install with: `using Pkg; Pkg.add("ReadVTK")`
- Or ignore if you only need time series plots

### Memory Issues with VTK
- Reduce `snapshot_steps` in `kh_advanced.jl`
- Process fewer variants at once
- Use lower resolution in simulation

## Output File Sizes

Typical file sizes for reference:
- CSV files: ~50-200 KB each
- JSON files: ~100-500 KB each
- VTU files: ~1-10 MB each (depends on particle count)
- PNG plots: ~100-500 KB each

## Further Analysis

For more advanced analysis, consider:
- Energy spectrum computation (FFT of velocity field)
- Enstrophy evolution (integrated vorticity²)
- Mixing metrics (scalar variance decay)
- Structure function analysis
- Time-frequency analysis (wavelets)

## References

- Kelvin-Helmholtz instability: https://en.wikipedia.org/wiki/Kelvin%E2%80%93Helmholtz_instability
- SPH method: Monaghan, J. J. (2005). "Smoothed particle hydrodynamics"
- SGS models: Adami et al. (2013), Morris et al. (1997)

---

**Questions or Issues?**
Please check the main TrixiParticles.jl documentation or open an issue on GitHub.
