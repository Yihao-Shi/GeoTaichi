# trace generated using paraview version 6.0.1
#import paraview
#paraview.compatibility.major = 6
#paraview.compatibility.minor = 0

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Render an MPM-LSDEM frame with ParaView.')
parser.add_argument('-i', '--input-dir', required=True)
parser.add_argument('-o', '--output-dir', required=True)
parser.add_argument('-f', '--frame', type=int, required=True)
parser.add_argument(
    '--case-root',
    default=str(Path(__file__).resolve().parent),
    help='Directory containing the input and output case folders',
)
args = parser.parse_args()

case_root = Path(args.case_root).expanduser().resolve()
dem = str(
    case_root / args.input_dir / 'vtks' / f'GraphicLSDEMSurface{args.frame:06d}.vtu'
)
mpm = str(
    case_root / args.input_dir / 'vtks' / f'GraphicMPMParticle{args.frame:06d}.vtu'
)
output_dir = case_root / args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)
output = str(output_dir / f'vel{args.frame}.png')

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Unstructured Grid Reader'
graphicLSDEMSurface000000vtu = XMLUnstructuredGridReader(registrationName=f'GraphicLSDEMSurface{args.frame:06d}.vtu', FileName=[dem])

# create a new 'XML Unstructured Grid Reader'
graphicMPMParticle000000vtu = XMLUnstructuredGridReader(registrationName=f'GraphicMPMParticle{args.frame:06d}.vtu', FileName=[mpm])

# Properties modified on graphicMPMParticle000000vtu
graphicMPMParticle000000vtu.TimeArray = 'None'

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
graphicMPMParticle000000vtuDisplay = Show(graphicMPMParticle000000vtu, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
graphicMPMParticle000000vtuDisplay.Representation = 'Surface'

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# show color bar/color legend
graphicMPMParticle000000vtuDisplay.SetScalarBarVisibility(renderView1, True)

# Properties modified on graphicLSDEMSurface000000vtu
graphicLSDEMSurface000000vtu.TimeArray = 'None'

# show data in view
graphicLSDEMSurface000000vtuDisplay = Show(graphicLSDEMSurface000000vtu, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
graphicLSDEMSurface000000vtuDisplay.Representation = 'Surface'

# show color bar/color legend
graphicLSDEMSurface000000vtuDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get color transfer function/color map for 'bodyID'
bodyIDLUT = GetColorTransferFunction('bodyID')

# get opacity transfer function/opacity map for 'bodyID'
bodyIDPWF = GetOpacityTransferFunction('bodyID')

# get 2D transfer function for 'bodyID'
bodyIDTF2D = GetTransferFunction2D('bodyID')

# hide color bar/color legend
graphicMPMParticle000000vtuDisplay.SetScalarBarVisibility(renderView1, False)

# change representation type
graphicMPMParticle000000vtuDisplay.SetRepresentationType('Points')

# Properties modified on graphicMPMParticle000000vtuDisplay
graphicMPMParticle000000vtuDisplay.PointSize = 5.0

# set scalar coloring
ColorBy(graphicMPMParticle000000vtuDisplay, ('POINTS', 'velocity', 'Magnitude'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(bodyIDLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
graphicMPMParticle000000vtuDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
graphicMPMParticle000000vtuDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'velocity'
velocityLUT = GetColorTransferFunction('velocity')

# get opacity transfer function/opacity map for 'velocity'
velocityPWF = GetOpacityTransferFunction('velocity')

# get 2D transfer function for 'velocity'
velocityTF2D = GetTransferFunction2D('velocity')

# hide color bar/color legend
graphicMPMParticle000000vtuDisplay.SetScalarBarVisibility(renderView1, False)
renderView1.UseColorPaletteForBackground = 0
renderView1.Background = [1.0, 1.0, 1.0]
renderView1.OrientationAxesVisibility = 0
renderView1.Update()

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(784, 516)


# current camera placement for renderView1
renderView1.Set(
    CameraPosition=[81.81483649644113, -46.760670313310875, 31.989085891604798],
    CameraFocalPoint=[25.44518596936002, 24.779344814624775, 6.402276273503613],
    CameraViewUp=[-0.2211128163815652, 0.16928475362160011, 0.9604435405702338],
    CameraParallelScale=35.849511851627774,
)

# save screenshot
SaveScreenshot(filename=output, viewOrLayout=renderView1, location=16, ImageResolution=[784, 516])

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(784, 516)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.Set(
    CameraPosition=[81.81483649644113, -46.760670313310875, 31.989085891604798],
    CameraFocalPoint=[25.44518596936002, 24.779344814624775, 6.402276273503613],
    CameraViewUp=[-0.2211128163815652, 0.16928475362160011, 0.9604435405702338],
    CameraParallelScale=35.849511851627774,
)


##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, usefull when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple splitted view)
# SaveScreenshot("path/to/screenshot.png", GetLayout())
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://www.paraview.org/paraview-docs/nightly/python/
##--------------------------------------------
