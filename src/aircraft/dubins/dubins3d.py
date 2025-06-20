"""
3D Dubins Maneuver Implementation

This module implements 3D Dubins path planning by combining horizontal and vertical
Dubins maneuvers. It provides functionality to compute feasible 3D paths between
two configurations while respecting minimum turning radius and pitch angle constraints.
"""
import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .dubins2d import DubinsManeuver2D
from .vertical import Vertical

@dataclass
class DubinsManeuver3D:
    """
    DubinsManeuver3D struct

    This struct contains all necessary information about the maneuver.
    * qi - initial configuration (x, y, z, heading, pitch)
    * qf - final configuration (x, y, z, heading, pitch)
    * rhomin - minimum turning radius
    * pitchlims - limits of the pitch angle [pitch_min, pitch_max] 
      where pitch_min < 0.0   
    * path - array containing horizontal and vertical Dubins paths
    * length - total length of the 3D maneuver 
    """
    qi: np.ndarray
    qf: np.ndarray
    rhomin: float
    pitchlims: np.ndarray
    path: List[DubinsManeuver2D]
    length: float

def DubinsManeuver3D_constructor(qi: List[float], qf: List[float], 
                                rhomin: float, pitchlims: List[float]) -> DubinsManeuver3D:
    """
    Create 3D Dubins path between two configurations qi, qf
    * qi - initial configuration (x, y, z, heading, pitch)
    * qf - final configuration (x, y, z, heading, pitch)
    * rhomin - minimum turning radius
    * pitchlims - limits of the pitch angle [pitch_min, pitch_max] 
    """
    maneuver = DubinsManeuver3D(
        qi=np.array(qi), 
        qf=np.array(qf), 
        rhomin=rhomin, 
        pitchlims=np.array(pitchlims), 
        path=[], 
        length=-1.0
    )

    # Delta Z (height)
    zi = maneuver.qi[2]
    zf = maneuver.qf[2]
    dz = zf - zi
    
    # Multiplication factor of rhomin in [1, 1000]
    a = 1.0
    b = 1.0

    fa = try_to_construct(maneuver, maneuver.rhomin * a)
    fb = try_to_construct(maneuver, maneuver.rhomin * b)

    while len(fb) < 2:
        b *= 2.0
        fb = try_to_construct(maneuver, maneuver.rhomin * b)

    if len(fa) > 0:
        maneuver.path = fa
    else:
        if len(fb) < 2:
            raise ValueError("No maneuver exists")

    # Binary search (commented out in original)
    # while abs(b-a) > 1e-5:
    #     c = (a+b) / 2.0
    #     fc = try_to_construct(maneuver, maneuver.rhomin * c)
    #     if len(fc) > 0:
    #         b = c
    #         fb = fc
    #     else:
    #         a = c

    # Local optimization between horizontal and vertical radii
    step = 0.1
    while abs(step) > 1e-10:
        c = b + step
        if c < 1.0:
            c = 1.0
        fc = try_to_construct(maneuver, maneuver.rhomin * c)
        if len(fc) > 0:
            if fc[1].maneuver.length < fb[1].maneuver.length:
                b = c
                fb = fc
                step *= 2.0
                continue
        step *= -0.1
    
    maneuver.path = fb
    Dlat, Dlon = fb
    maneuver.length = Dlon.maneuver.length
    return maneuver

def compute_sampling(maneuver: DubinsManeuver3D, numberOfSamples: int = 1000) -> List[List[float]]:
    """Compute sampling points along the 3D path"""
    Dlat, Dlon = maneuver.path
    # Sample points on the final path
    points = []
    lena = Dlon.maneuver.length
    rangeLon = lena * np.arange(numberOfSamples) / (numberOfSamples - 1)

    for ran in rangeLon:   
        offsetLon = ran
        qSZ = Dlon.getCoordinatesAt(offsetLon)
        qXY = Dlat.getCoordinatesAt(qSZ[0])
        points.append([qXY[0], qXY[1], qSZ[1], qXY[2], qSZ[2]])
    
    return points

def try_to_construct(maneuver: DubinsManeuver3D, horizontal_radius: float) -> List[DubinsManeuver2D]:
    """Try to construct a 3D maneuver with given horizontal radius"""
    qi2D = [maneuver.qi[0], maneuver.qi[1], maneuver.qi[3]]
    qf2D = [maneuver.qf[0], maneuver.qf[1], maneuver.qf[3]]

    Dlat = DubinsManeuver2D(qi2D, qf2D, rhomin=horizontal_radius)    
    
    # After finding a long enough 2D curve, calculate the Dubins path on SZ axis
    qi3D = [0.0, maneuver.qi[2], maneuver.qi[4]]
    qf3D = [Dlat.maneuver.length, maneuver.qf[2], maneuver.qf[4]]

    vertical_curvature = math.sqrt(1.0 / maneuver.rhomin**2 - 1.0 / horizontal_radius**2)
    if vertical_curvature < 1e-5:
        return []

    vertical_radius = 1.0 / vertical_curvature
    # Dlon = Vertical(qi3D, qf3D, vertical_radius, maneuver.pitchlims)
    Dlon = DubinsManeuver2D(qi3D, qf3D, rhomin=vertical_radius)

    if Dlon.maneuver.case == "RLR" or Dlon.maneuver.case == "LRL":
        return []

    if Dlon.maneuver.case[0] == 'R':
        if maneuver.qi[4] - Dlon.maneuver.t < maneuver.pitchlims[0]:
            return []
    else:
        if maneuver.qi[4] + Dlon.maneuver.t > maneuver.pitchlims[1]:
            return []
    
    # Final 3D path is formed by the two curves (Dlat, Dlon)
    return [Dlat, Dlon]

def getLowerBound(qi: List[float], qf: List[float], rhomin: float = 1, 
                  pitchlims: List[float] = None) -> DubinsManeuver3D:
    """Get lower bound estimate for 3D maneuver"""
    if pitchlims is None:
        pitchlims = [-math.pi/4, math.pi/2]
    
    maneuver = DubinsManeuver3D(
        qi=np.array(qi), 
        qf=np.array(qf), 
        rhomin=rhomin, 
        pitchlims=np.array(pitchlims), 
        path=[], 
        length=-1.0
    )

    spiral_radius = rhomin * (math.cos(max(-pitchlims[0], pitchlims[1])) ** 2)

    qi2D = [maneuver.qi[i] for i in [0, 1, 3]]
    qf2D = [maneuver.qf[i] for i in [0, 1, 3]]
    Dlat = DubinsManeuver2D(qi2D, qf2D, rhomin=spiral_radius)  

    qi3D = [0, maneuver.qi[2], maneuver.qi[4]]
    qf3D = [Dlat.maneuver.length, maneuver.qf[2], maneuver.qf[4]]

    Dlon = Vertical(qi3D, qf3D, maneuver.rhomin, maneuver.pitchlims)

    if Dlon.maneuver.case == "XXX":
        # TODO - update Vertical such that it computes the shortest prolongation
        maneuver.length = 0.0
        return maneuver

    maneuver.path = [Dlat, Dlon]
    maneuver.length = Dlon.maneuver.length
    return maneuver

def getUpperBound(qi: List[float], qf: List[float], rhomin: float = 1, 
                  pitchlims: List[float] = None) -> DubinsManeuver3D:
    """Get upper bound estimate for 3D maneuver"""
    if pitchlims is None:
        pitchlims = [-math.pi/4, math.pi/2]
    
    maneuver = DubinsManeuver3D(
        qi=np.array(qi), 
        qf=np.array(qf), 
        rhomin=rhomin, 
        pitchlims=np.array(pitchlims), 
        path=[], 
        length=-1.0
    )

    safeRadius = math.sqrt(2) * maneuver.rhomin

    p1 = maneuver.qi[:2]
    p2 = maneuver.qf[:2]
    diff = p2 - p1
    dist = math.sqrt(diff[0]**2 + diff[1]**2)
    if dist < 4.0 * safeRadius:
        maneuver.length = float('inf')
        return maneuver    

    qi2D = [maneuver.qi[i] for i in [0, 1, 3]]
    qf2D = [maneuver.qf[i] for i in [0, 1, 3]]
    Dlat = DubinsManeuver2D(qi2D, qf2D, rhomin=safeRadius)  

    qi3D = [0, maneuver.qi[2], maneuver.qi[4]]
    qf3D = [Dlat.maneuver.length, maneuver.qf[2], maneuver.qf[4]]

    Dlon = Vertical(qi3D, qf3D, safeRadius, maneuver.pitchlims)

    if Dlon.maneuver.case == "XXX":
        # TODO - update Vertical such that it computes the shortest prolongation
        maneuver.length = float('inf')
        return maneuver

    maneuver.path = [Dlat, Dlon]
    maneuver.length = Dlon.maneuver.length
    return maneuver

# Convenience function to create DubinsManeuver3D
def create_dubins_3d(qi: List[float], qf: List[float], rhomin: float, 
                     pitchlims: List[float]) -> DubinsManeuver3D:
    """Create a 3D Dubins maneuver"""
    return DubinsManeuver3D_constructor(qi, qf, rhomin, pitchlims)
