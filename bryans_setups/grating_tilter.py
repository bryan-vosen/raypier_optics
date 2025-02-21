import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

from raypier.tracer import RayTraceModel
from raypier.faces import CylindericalFace, PlanarFace, SphericalFace
from raypier.shapes import RectangleShape, CircleShape
from raypier.achromats import EdmundOptic45805
from raypier.diffraction_gratings import RectangularGrating
from raypier.mirrors import BaseMirror, PECMirror, RectMirror
from raypier.sources import BroadbandRaySource
from raypier.beamstop import BeamStop
from raypier.results import GroupVelocityDispersion, MeanOpticalPathLength, evaluate_phase, evaluate_group_delay
from raypier.constraints import BaseConstraint
from raypier.chirp_result import ChirpResult
from raypier.materials import OpticalMaterial
from raypier.gausslet_sources import CollimatedGaussletSource, BroadbandGaussletSource
from raypier.api import GeneralLens
from raypier.core.cfaces import ShapedSphericalFace, CircularFace
#from raypier.core.cshapes import CircleShape, RectangleShape
from raypier.core.ctracer import FaceList

from traits.api import Range, on_trait_change
from traitsui.api import View, Item


direction = np.array([0.0,-1,-0.015])
grating_distance = 500
grating_init_rotation = -151.31
cylindrical_radius_curvature = 1000
source_bandwidth = 32
source_wavelength = 800


wavelength_start = source_wavelength / 1000 - source_bandwidth/2000
wavelength_end = source_wavelength / 1000 + source_bandwidth/2000
source = BroadbandRaySource(origin=(0,260,10),
                            direction=tuple(direction),
                            wavelength_start=wavelength_start,
                            wavelength_end = wavelength_end,
                            number=500,
                            uniform_deltaf=True,
                            max_ray_len= 2000.0)

# source = BroadbandGaussletSource(origin=(-29.685957,124.73850,10),
#                             direction=tuple(direction),
#                             wavelength = .8,
#                             wavelength_extent = 0.1,
#                             bandwidth_nm = 10,
#                             number=260,
#                             uniform_deltaf=True,
#                             max_ray_len=300.0,
#                             beam_waist = 2000)

# source = CollimatedGaussletSource(origin=(-29.685957,124.73850,1.5),
#                                 direction=tuple(direction),
#                                wavelength=1.0,
#                                radius=10.0,
#                                beam_waist=10.0,
#                                resolution=10,
#                                max_ray_len=200.0,
#                                display='wires',
#                                opacity=0.2
#                                )


grating = RectangularGrating(centre=(0,grating_distance,-4),
                             direction=(-1,0,0),
                             lines_per_mm=1200,
                             order=-1,
                             width = 20,
                             length = 30)


grating.orientation = grating_init_rotation

lens = EdmundOptic45805(centre=(0,75,0),
                        direction=(0,-1,0))
init_lens_rotation = lens.orientation


shape = RectangleShape(centre = (0, 0), width = 80, height = 60)

#face1 = PlanarFace(z_height=0.0)
face2 = CylindericalFace(z_height=4.0, curvature=cylindrical_radius_curvature, mirror = True)

mat = OpticalMaterial(refractive_index=1)


mir = GeneralLens(name = "Cylinderincal Lens",
                     centre = (0,0,0),
                     direction=(0,-1,0),
                     shape=shape, 
                     surfaces=[face2], 
                     materials=[mat])


# bs = BeamStop(centre=(0,grating_distance+60,-20),
#               direction=(0 , -1,  0),
#               diameter = 100
#               )

bs = BeamStop(centre=(0,200,-27.5),
              direction=(0 , -1,  0),
              diameter = 40
              )


gvd = GroupVelocityDispersion(source=source, target=bs.faces.faces[0])


class GVDConstraint(BaseConstraint):
    name = "Dispersion Compensator Adjustment"
    focus_adjust = Range(70.0,80.0, value=73.5)
    gdd_adjust = Range(-20.0,40.0, value=0.0)
    lens_rotation = Range(-20.0, 20.0, value=0.0)
    grating_angle = Range(-5.0,5.0, value=0.0)
    
    traits_view = View(Item("focus_adjust"),
                       Item("gdd_adjust"),
                       Item("lens_rotation"),
                       Item("grating_angle"),
                       resizable=True)
    
    @on_trait_change("focus_adjust, gdd_adjust")
    def on_new_values(self):
        lens.centre = (0.0, 75.0 + self.gdd_adjust, 0.0)
        mir.centre = (0.0, 75.0 + self.gdd_adjust + self.focus_adjust, 0.0)
        self.update = True
        
    def _lens_rotation_changed(self):
        lens.orientation = (init_lens_rotation + self.lens_rotation+180.0)%360.0 - 180.0
        self.update = True
        
    def _grating_angle_changed(self):
        grating.orientation = grating_init_rotation + self.grating_angle
        self.update = True
    
        
gvd_cstr = GVDConstraint()
    

chirp = ChirpResult(source=source, target=bs.faces.faces[0])

mean_optical_path = MeanOpticalPathLength(source = source, target = bs.faces.faces[0])

model = RayTraceModel(optics=[ grating, mir, bs], 
                      sources=[source,],
                      results=[gvd, chirp, mean_optical_path],
                    #   constraints=[gvd_cstr,]
                        )

try:
    freq, phase = evaluate_phase(all_wavelengths=np.asarray(source.wavelength_list), traced_rays = source.traced_rays, target_face = bs.faces.faces[0])
    # wavelength = 299792.458 / freq 

    wavelength, group_delay = evaluate_group_delay(all_wavelengths=np.asarray(source.wavelength_list), traced_rays = source.traced_rays, target_face = bs.faces.faces[0])
    group_delay_fs = group_delay * 1000
    wavelength = wavelength * 1000
    fig, ax = plt.subplots()
    ax.plot(wavelength, phase)
    ax.set(xlabel = "Wavelength (nm)", ylabel = "Phase", title = f"Phase Evalution, gd = {grating_distance}")
    ax.grid()
    now = time.strftime("%Y%m%d-%H%M%S")
    fig.savefig(f"D:\senior_design\Screenshots\phase_plots\phase_plot_{now}.png")

    meanSquaredError = ((group_delay_fs) ** 2).mean()
    rmse = np.sqrt(meanSquaredError)
    rmse = np.round(rmse, 2)

    fig, ax = plt.subplots()
    ax.plot(wavelength, group_delay_fs)
    ax.set(xlabel = "Wavelength (nm)", ylabel = "Group Delay (fs)", title = f"Group Delay, rmse = {rmse} fs")
    ax.grid()


    # Get current date and time
    now = time.strftime("%Y%m%d-%H%M%S")
    fig.savefig(f"D:\senior_design\Screenshots\group_delay_plots\group_delay_plot_{now}.png")
except:
    print("Error making plot")





model.configure_traits(kind="live")
