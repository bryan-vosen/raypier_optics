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
from raypier.results import GroupVelocityDispersion, MeanOpticalPathLength, evaluate_phase,evaluate_group_delay
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

def generate_grating_distance_group_delay_plot():
    all_rmse = []
    grating_distances = np.arange(400, 990, 1)

    direction = np.array([0.0,-1,-0.015])
    grating_distance = 500
    grating_init_rotation = -151.31
    cylindrical_radius_curvature = 1000
    source_bandwidth = 32
    source_wavelength = 800


    for grating_distance in grating_distances:

        wavelength_start = source_wavelength / 1000 - source_bandwidth/2000
        wavelength_end = source_wavelength / 1000 + source_bandwidth/2000
        source = BroadbandRaySource(origin=(0,260,10),
                                    direction=tuple(direction),
                                    wavelength_start=wavelength_start,
                                    wavelength_end = wavelength_end,
                                    number=500,
                                    uniform_deltaf=True,
                                    max_ray_len=1000000.0)



        grating = RectangularGrating(centre=(0,grating_distance,-4),
                                    direction=(-1,0,0),
                                    lines_per_mm=1200,
                                    order=-1,
                                    width = 10,
                                    length = 30)


        grating.orientation = grating_init_rotation

        shape = RectangleShape(centre = (0, 0), width = 80, height = 60)
        face2 = CylindericalFace(z_height=4.0, curvature=cylindrical_radius_curvature, mirror = True)
        mat = OpticalMaterial(refractive_index=1)
        mir = GeneralLens(name = "Cylinderincal Lens",
                            centre = (0,0,0),
                            direction=(0,-1,0),
                            shape=shape, 
                            surfaces=[face2], 
                            materials=[mat])


        bs = BeamStop(centre=(0,1000,-20),
                    direction=(0 , -1,  0),
                    diameter = 100
                    )

        gvd = GroupVelocityDispersion(source=source, target=bs.faces.faces[0])
        chirp = ChirpResult(source=source, target=bs.faces.faces[0])
        mean_optical_path = MeanOpticalPathLength(source = source, target = bs.faces.faces[0])

        model = RayTraceModel(optics=[ grating, mir, bs], 
                            sources=[source,],
                            results=[gvd, chirp, mean_optical_path],
                            #   constraints=[gvd_cstr,]
                                )

        try:
            wavelength, group_delay = evaluate_group_delay(all_wavelengths=np.asarray(source.wavelength_list), traced_rays = source.traced_rays, target_face = bs.faces.faces[0])
            group_delay_fs = group_delay * 1000
            wavelength = wavelength * 1000


            meanSquaredError = ((group_delay_fs) ** 2).mean()
            rmse = np.sqrt(meanSquaredError)
            all_rmse.append(rmse)

            # fig, ax = plt.subplots()
            # ax.plot(wavelength, group_delay_fs)
            # ax.set(xlabel = "Wavelength (nm)", ylabel = "Group Delay (fs)", title = f"Group Delay, rmse = {rmse} fs")
            # ax.grid()


            # Get current date and time
            # now = time.strftime("%Y%m%d-%H%M%S")
            # fig.savefig(f"D:\senior_design\Screenshots\group_delay_plots\group_delay_plot_{now}.png")
        except:
            print("Error making plot")
    
    fig, ax = plt.subplots()
    ax.plot(grating_distances, all_rmse)
    ax.set(xlabel = "Grating Distance (mm)", ylabel = "Group Delay RMSE (fs)", title = f"Group Delay RMSE for varying grating distances, min = {grating_distances[np.argmin(all_rmse)]}")
    ax.grid()
    now = time.strftime("%Y%m%d-%H%M%S")
    fig.savefig(f"D:\senior_design\Screenshots\group_delay_plots\group_delay_plot_grating_distances_{now}.png")


generate_grating_distance_group_delay_plot()