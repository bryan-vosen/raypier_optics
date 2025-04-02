import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

from raypier.tracer import RayTraceModel
from raypier.faces import CylindericalFace, PlanarFace, SphericalFace
from raypier.shapes import RectangleShape, CircleShape
from raypier.achromats import EdmundOptic45805
from raypier.diffraction_gratings import RectangularGrating
from raypier.mirrors import BaseMirror, PECMirror, RectMirror, CylindricalPECMirror, SphericalMirrorWithHole, PlanarDispersiveWindow, PlanarWindow
from raypier.sources import BroadbandRaySource
from raypier.beamstop import BeamStop, RectTarget
from raypier.results import GroupVelocityDispersion, MeanOpticalPathLength, evaluate_phase, evaluate_group_delay
from raypier.constraints import BaseConstraint
from raypier.chirp_result import ChirpResult
from raypier.materials import OpticalMaterial, AirMaterial
from raypier.gausslet_sources import CollimatedGaussletSource, BroadbandGaussletSource, GaussianPointSource
from raypier.api import GeneralLens
from raypier.probes import GaussletCapturePlane
from raypier.core.cfaces import ShapedSphericalFace, CircularFace
#from raypier.core.cshapes import CircleShape, RectangleShape
from raypier.core.ctracer import FaceList

from traits.api import Range, on_trait_change
from traitsui.api import View, Item

# source = GaussianPointSource(origin=(0,50,0),
#                           direction=(0,-1,0),
#                           E_vector=(1,0,0),
#                           E1_amp = 1.0,
#                           E2_amp = 0.0,
#                           working_dist = 0,
#                           numerical_aperture=0.1,
#                           wavelength = 1.0,
#                           beam_waist = 5.0,
#                           display="wires",
#                           opacity=0.1,
#                           max_ray_len= 2000.0,
#                           )
# sources = []
# for z_direction in np.arange(-.1, .1, .01):
#     for x_direction in np.arange(-.1, .1, .01):
#         source = BroadbandRaySource(origin=(0,50,0),
#                             direction=tuple((x_direction,-1,z_direction)),
#                             wavelength_start=.782,
#                             wavelength_end = .818,
#                             number=1,
#                             uniform_deltaf=True,
#                             max_ray_len= 2000.0,
#                             show_normals=True)
#         sources.append(source)
    
# source = BroadbandRaySource(origin=(0,50,0),
#                             direction=tuple((0,-1,0)),
#                             wavelength_start=.782,
#                             wavelength_end = .818,
#                             number=200,
#                             uniform_deltaf=True,
#                             max_ray_len= 2000.0,
#                             show_normals=True)

source = BroadbandRaySource(origin=(0,50,0),
                            direction=tuple((.09,-1,-.09)),
                            wavelength_start=.782,
                            wavelength_end = .818,
                            number=200,
                            uniform_deltaf=True,
                            max_ray_len= 2000.0,
                            show_normals=True)



# shape1 = RectangleShape(centre = (0, 0), width = 80, height = 60)
# mir = SphericalMirrorWithHole(centre = (0, 0 ,0), curvature = 100, shape = shape1, diameter = 500)

shape = RectangleShape(centre = (0, 0), width = 80, height = 60)
face1 = PlanarFace(z_height=0, mirror = True)
face2 = SphericalFace(z_height=0.0, curvature=-100, mirror = True)

mat = OpticalMaterial( refractive_index=1.00)
mir = GeneralLens(name = "Cylinderincal Lens",
                    centre = (0,0,0),
                    direction=(0,1,0),
                    shape=shape, 
                    surfaces=[face2], 
                    materials=[],

                    )

mir = GeneralLens(name = "Cylinderincal Lens",
                    centre = (0,0,0),
                    direction=(0,1,0),
                    shape=shape, 
                    surfaces=[face2], 
                    materials=[],

                    )

# bs = RectTarget(centre=(0,70,-0),
#                     direction=(0 , -1,  0),
#                     diameter = 50
#                     )

mir2 = GeneralLens(name = "Beam Stop?",
                    centre = (0, 70 ,0),
                    direction=(.5,-1,0),
                    shape=shape, 
                    surfaces=[face1], 
                    materials=[],

                    )
mir2.orientation = 45

bs = RectTarget(centre=(40,70,-0),
                    direction=(-1 , 0,  0),
                    diameter = 50
                    )

# bs2= RectTarget(centre=(0,71,-0),
#                     direction=(0 , -1,  0),
#                     diameter = 50
#                     )

probe =  GaussletCapturePlane(centre=(0,60,-0),
                    direction=(0 , -1,  0),
                    diameter = 50,
                    width=5.0,
                    height=5.0)

window = PlanarWindow(mat = AirMaterial(), centre = (0, 65, 0), direction = (0, -1, 0), n_inside = 1)

chirp = ChirpResult(source=source, target=bs.faces.faces[0])

model = RayTraceModel(optics=[mir, bs, mir2], 
                            sources=[source],
                            results=[chirp],
                                )
model.trace_all()

all_rays = source.traced_rays
all_wavelengths = source.wavelength_list
# for source in sources:
#     all_rays.append(source.traced_rays)
#     all_wavelengths.append(source.wavelength_list)

# freq, phase = evaluate_phase(all_wavelengths=np.asarray(all_wavelengths), traced_rays = all_rays, target_face = bs.faces.faces[0])
# wavelength = 299792.458 / freq 

print(bs.faces.faces)
target_face = bs.faces.faces[0]
# target_face = mir.faces.faces[0]
wavelength, group_delay = evaluate_group_delay(all_wavelengths=np.asarray(all_wavelengths), traced_rays = all_rays, target_face = target_face)
group_delay_fs = group_delay * 1000
wavelength = wavelength * 1000

# fig, ax = plt.subplots()
# ax.plot(wavelength, phase)
# ax.set(xlabel = "Wavelength (nm)", ylabel = "Phase", title = f"Phase Evalution for Test")
# ax.grid()
# now = time.strftime("%Y%m%d-%H%M%S")
# fig.savefig(f"D:\senior_design\Screenshots\phase_plots\phase_plot_Testing{now}.png")

meanSquaredError = ((group_delay_fs) ** 2).mean()
rmse = np.sqrt(meanSquaredError)
rmse = np.round(rmse, 2)

fig, ax = plt.subplots()
ax.plot(wavelength, group_delay_fs)
ax.set(xlabel = "Wavelength (nm)", ylabel = "Group Delay (fs)", title = f"Group Delay Test, rmse = {rmse} fs")
ax.grid()


# Get current date and time
now = time.strftime("%Y%m%d-%H%M%S")
fig.savefig(f"D:\senior_design\Screenshots\group_delay_plots\group_delay_plot_test{now}.png")

print('Failed to make plot')


model.configure_traits(kind="live")