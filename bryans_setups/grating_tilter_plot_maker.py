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
from raypier.results import GroupVelocityDispersion, MeanOpticalPathLength, evaluate_phase
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

def generate_gd_distance_plot():
    #grating distances
    phase_ranges = []
    grating_distances = np.arange(43, 48, .05)
    for grating_distance in grating_distances:
        direction = np.array([0.0,-1,-0.1])
        source = BroadbandRaySource(origin=(0,70,10),
                                    direction=tuple(direction),
                                    wavelength_start=0.78,
                                    wavelength_end = 0.82,
                                    number=260,
                                    uniform_deltaf=True,
                                    max_ray_len=800.0)


        grating = RectangularGrating(centre=(0,grating_distance,-2),
                                    direction=(-1,0,0),
                                    lines_per_mm=1200,
                                    order=-1,
                                    width = 10)

        grating_init_rotation = -151
        grating.orientation = grating_init_rotation

        shape = RectangleShape(centre = (0, 0), width = 80, height = 60)

        face1 = PlanarFace(z_height=0.0)
        face2 = CylindericalFace(z_height=4.0, curvature=100.0, mirror = True)

        mat = OpticalMaterial(refractive_index=1)


        mir = GeneralLens(name = "Cylinderincal Lens",
                            centre = (0,0,0),
                            direction=(0,-1,0),
                            shape=shape, 
                            surfaces=[face2, 
                                    face1], 
                            materials=[mat])


        bs = BeamStop(centre=(0,50,-10),
                    direction=(0 , -1,  0),
                    diameter = 10
                    )


        model = RayTraceModel(optics=[ grating, mir, bs], 
                            sources=[source,],
                            results=[],
                            #   constraints=[gvd_cstr,]
                                )

        try:
            freq, phase = evaluate_phase(all_wavelengths=np.asarray(source.wavelength_list), traced_rays = source.traced_rays, target_face = bs.faces.faces[0])
            wavelength = 299792.458 / freq 

            phase_range = np.max(phase) - np.min(phase)
            phase_ranges.append(phase_range)
        except:
            print("Error making plot")
            phase_ranges.append(np.nan)

        print(grating_distance)

    fig, ax = plt.subplots()
    ax.plot(grating_distances, phase_ranges)
    ax.set(xlabel = "Grating Distance (cm)", ylabel = "Phase Range", title = f"Phase Evalution for varying Grating Distances")
    ax.grid()

    fig.savefig(f"D:\senior_design\Screenshots\phase_plots\phase_eval_gd.png")

def generate_grating_rotation_plot():
    phase_ranges = []
    grating_angles = np.arange(-155, -145, .05)
    grating_distance = 46
    for grating_angle in grating_angles:
        direction = np.array([0.0,-1,-0.1])
        source = BroadbandRaySource(origin=(0,70,10),
                                    direction=tuple(direction),
                                    wavelength_start=0.78,
                                    wavelength_end = 0.82,
                                    number=260,
                                    uniform_deltaf=True,
                                    max_ray_len=800.0)


        grating = RectangularGrating(centre=(0,grating_distance,-2),
                                    direction=(-1,0,0),
                                    lines_per_mm=1200,
                                    order=-1,
                                    width = 10)

        grating_init_rotation = grating_angle
        grating.orientation = grating_init_rotation

        shape = RectangleShape(centre = (0, 0), width = 80, height = 60)

        face1 = PlanarFace(z_height=0.0)
        face2 = CylindericalFace(z_height=4.0, curvature=100.0, mirror = True)

        mat = OpticalMaterial(refractive_index=1)


        mir = GeneralLens(name = "Cylinderincal Lens",
                            centre = (0,0,0),
                            direction=(0,-1,0),
                            shape=shape, 
                            surfaces=[face2, 
                                    face1], 
                            materials=[mat])


        bs = BeamStop(centre=(0,50,-10),
                    direction=(0 , -1,  0),
                    diameter = 10
                    )


        model = RayTraceModel(optics=[ grating, mir, bs], 
                            sources=[source,],
                            results=[],
                            #   constraints=[gvd_cstr,]
                                )

        try:
            freq, phase = evaluate_phase(all_wavelengths=np.asarray(source.wavelength_list), traced_rays = source.traced_rays, target_face = bs.faces.faces[0])
            wavelength = 299792.458 / freq 

            if np.isnan(np.min(phase)):
                raise
            phase_range = np.max(phase) - np.min(phase)
            phase_ranges.append(phase_range)
        except:
            print("Error making plot")
            phase_ranges.append(np.nan)

        print(grating_angle)

    fig, ax = plt.subplots()
    ax.plot(grating_angles, phase_ranges)
    ax.set(xlabel = "Grating Angle (degrees)", ylabel = "Phase Range", title = f"Phase Evalution for varying Grating Angles")
    ax.grid()

    fig.savefig(f"D:\senior_design\Screenshots\phase_plots\phase_eval_ga.png")

def generate_beam_lateral_pos_plot():
    phase_ranges = []
    beam_positions = np.arange(-1, 1, .005)
    grating_distance = 46
    grating_angle = -151.1
    for beam_pos in beam_positions:
        direction = np.array([0.0,-1,-0.1])
        source = BroadbandRaySource(origin=(beam_pos,70,10),
                                    direction=tuple(direction),
                                    wavelength_start=0.78,
                                    wavelength_end = 0.82,
                                    number=260,
                                    uniform_deltaf=True,
                                    max_ray_len=800.0)


        grating = RectangularGrating(centre=(0,grating_distance,-2),
                                    direction=(-1,0,0),
                                    lines_per_mm=1200,
                                    order=-1,
                                    width = 10)

        grating_init_rotation = grating_angle
        grating.orientation = grating_init_rotation

        shape = RectangleShape(centre = (0, 0), width = 80, height = 60)

        face1 = PlanarFace(z_height=0.0)
        face2 = CylindericalFace(z_height=4.0, curvature=100.0, mirror = True)

        mat = OpticalMaterial(refractive_index=1)


        mir = GeneralLens(name = "Cylinderincal Lens",
                            centre = (0,0,0),
                            direction=(0,-1,0),
                            shape=shape, 
                            surfaces=[face2, 
                                    face1], 
                            materials=[mat])


        bs = BeamStop(centre=(0,50,-10),
                    direction=(0 , -1,  0),
                    diameter = 10
                    )


        model = RayTraceModel(optics=[ grating, mir, bs], 
                            sources=[source,],
                            results=[],
                            #   constraints=[gvd_cstr,]
                                )

        try:
            freq, phase = evaluate_phase(all_wavelengths=np.asarray(source.wavelength_list), traced_rays = source.traced_rays, target_face = bs.faces.faces[0])
            wavelength = 299792.458 / freq 

            if np.isnan(np.min(phase)):
                raise
            phase_range = np.max(phase) - np.min(phase)
            phase_ranges.append(phase_range)
        except:
            print("Error making plot")
            phase_ranges.append(np.nan)

        print(grating_angle)

    fig, ax = plt.subplots()
    ax.plot(beam_positions, phase_ranges)
    ax.set(xlabel = "Beam Position (cm)", ylabel = "Phase Range", title = f"Phase Evalution for varying Lateral Beam Positions")
    ax.grid()

    fig.savefig(f"D:\senior_design\Screenshots\phase_plots\phase_eval_beam_pos_lateral.png")



generate_beam_lateral_pos_plot()