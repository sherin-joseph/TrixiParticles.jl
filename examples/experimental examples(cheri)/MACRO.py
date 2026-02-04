# Generic SPH visualization macro for TrixiParticles outputs
# Works for 2D (TGV, Poiseuille, KH) and 3D (DHIT)
from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()

# ---------------------------------------------------------------------------
# Get active source & view
# ---------------------------------------------------------------------------
source = GetActiveSource()
if source is None:
    raise RuntimeError("No active source. Select a dataset before running the macro.")

renderView = GetActiveViewOrCreate('RenderView')

# Ensure we have display properties (Show if it wasn't shown yet)
display = GetDisplayProperties(source, view=renderView)
if display is None:
    display = Show(source, renderView)

# ---------------------------------------------------------------------------
# Representation: Point Gaussian for all SPH particle clouds
# ---------------------------------------------------------------------------
display.SetRepresentationType('Point Gaussian')

# ---------------------------------------------------------------------------
# Detect "2D vs 3D" from data bounds and pick a shader preset
#  - 2D: z-extent ~ 0 -> Plain circle
#  - 3D: z-extent > tol -> Sphere
# ---------------------------------------------------------------------------
bounds = source.GetDataInformation().GetBounds()
# bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
z_min, z_max = bounds[4], bounds[5]
is_2d = abs(z_max - z_min) < 1.0e-6

if is_2d:
    display.ShaderPreset = 'Plain circle'
else:
    display.ShaderPreset = 'Sphere'

# ---------------------------------------------------------------------------
# Scale by particle_spacing, if available
#   - If the point data has a 'particle_spacing' array, we:
#       * enable scaling by this array
#       * choose GaussianRadius based on its range
#   - Otherwise, we fall back to a fixed radius
# ---------------------------------------------------------------------------
pdi = source.GetPointDataInformation()
ps_array = pdi.GetArray("particle_spacing")

if ps_array is not None:
    # Enable scaling by array
    display.ScaleByArray = 1
    display.SetScaleArray = ["POINTS", "particle_spacing"]
    display.UseScaleFunction = 0

    # Use the minimum positive value of particle_spacing as a base radius
    r0, r1 = ps_array.GetRange()
    if r0 > 0.0:
        base_radius = r0
    elif r1 > 0.0:
        base_radius = r1
    else:
        base_radius = 0.01  # very small fallback

    # Factor to make particles visually "touch" but not explode
    display.GaussianRadius = 0.75 * base_radius
else:
    # No particle_spacing array -> use a generic small radius
    display.ScaleByArray = 0
    display.GaussianRadius = 0.01

# Optional: keep coloring “as is” (pressure, velocity, etc.)
# If you want to enforce a default, uncomment something like:
#
# color_array = pdi.GetArray("pressure")
# if color_array is not None:
#     ColorBy(display, ("POINTS", "pressure"))
#     display.RescaleTransferFunctionToDataRange(True, False)
#     GetColorTransferFunction("pressure").ApplyPreset("Cool to Warm", True)
#
# Render view to update
Render()
