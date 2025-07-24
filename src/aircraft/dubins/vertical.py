"""
Vertical Dubins Maneuver with Pitch Constraints
"""
import numpy as np
import math
from typing import List, Tuple
from .dubins2d import DubinsManeuver2D, DubinsStruct

def Vertical(qi: List[float], qf: List[float], rhomin: float, pitchmax: Tuple[float, float]) -> DubinsManeuver2D:
    """
    Compute vertical Dubins maneuver with pitch constraints
    
    Args:
        qi: Initial configuration [x, y, z, theta]
        qf: Final configuration [x, y, z, theta]
        rhomin: Minimum turning radius
        pitchmax: Pitch constraints (min_pitch, max_pitch)
    
    Returns:
        DubinsManeuver2D object with computed maneuver
    """
    maneuver = DubinsManeuver2D(qi, qf, rhomin)
    maneuver.maneuver = DubinsStruct(0.0, 0.0, 0.0, float('inf'), "")

    dx = maneuver.qf[0] - maneuver.qi[0]
    dy = maneuver.qf[1] - maneuver.qi[1]
    D = math.sqrt(dx**2 + dy**2)

    # Distance normalization
    d = D / maneuver.rhomin

    # Normalize the problem using rotation
    rotationAngle = math.atan2(dy, dx) % (2 * math.pi)
    a = (maneuver.qi[2] - rotationAngle) % (2 * math.pi)
    b = (maneuver.qf[2] - rotationAngle) % (2 * math.pi)
    
    # CSC paths
    pathLSL = _LSL(maneuver)
    pathRSR = _RSR(maneuver)
    pathLSR = _LSR(maneuver, pitchmax)
    pathRSL = _RSL(maneuver, pitchmax)
    _paths = [pathLSR, pathLSL, pathRSR, pathRSL]
    
    # Sort by length
    _paths.sort(key=lambda x: x.length)
        
    for p in _paths:
        # Check if the turns are too long (do not meet pitch constraint)
        if abs(p.t) < math.pi and abs(p.q) < math.pi:
            # Check the inclination based on pitch constraint
            center_angle = maneuver.qi[2] + (p.t if p.case[0] == 'L' else -p.t)
            if not (center_angle < pitchmax[0] or center_angle > pitchmax[1]):
                maneuver.maneuver = p
                break
    
    if maneuver.maneuver.case == "":
        maneuver.maneuver = DubinsStruct(float('inf'), float('inf'), float('inf'), float('inf'), "XXX")

    return maneuver

def _LSL(maneuver: DubinsManeuver2D) -> DubinsStruct:
    """LSL path computation for vertical maneuver"""
    theta1 = maneuver.qi[2]
    theta2 = maneuver.qf[2]

    if theta1 <= theta2:
        # Start/end points
        p1 = maneuver.qi[:2]
        p2 = maneuver.qf[:2]

        radius = maneuver.rhomin

        c1, s1 = radius * math.cos(theta1), radius * math.sin(theta1)
        c2, s2 = radius * math.cos(theta2), radius * math.sin(theta2)

        # Origins of the turns
        o1 = p1 + np.array([-s1, c1])
        o2 = p2 + np.array([-s2, c2])

        diff = o2 - o1
        center_distance = math.sqrt(diff[0]**2 + diff[1]**2)
        centerAngle = math.atan2(diff[1], diff[0])
                
        t = (-theta1 + centerAngle) % (2 * math.pi)
        p = center_distance / radius
        q = (theta2 - centerAngle) % (2 * math.pi)

        if t > math.pi:
            t = 0.0
            q = theta2 - theta1
            turn_end_y = o2[1] - radius * math.cos(theta1)
            diff_y = turn_end_y - p1[1]
            if abs(theta1) > 1e-5 and (diff_y < 0) == (theta1 < 0):
                p = diff_y / math.sin(theta1) / radius
            else:
                t = p = q = float('inf')
                
        if q > math.pi:
            t = theta2 - theta1
            q = 0.0
            turn_end_y = o1[1] - radius * math.cos(theta2)
            diff_y = p2[1] - turn_end_y
            if abs(theta2) > 1e-5 and (diff_y < 0) == (theta2 < 0):
                p = diff_y / math.sin(theta2) / radius
            else:
                t = p = q = float('inf')
    else:
        t = p = q = float('inf')
    
    length = (t + p + q) * maneuver.rhomin
    case = "LSL"
    
    return DubinsStruct(t, p, q, length, case)

def _RSR(maneuver: DubinsManeuver2D) -> DubinsStruct:
    """RSR path computation for vertical maneuver"""
    theta1 = maneuver.qi[2]
    theta2 = maneuver.qf[2]

    if theta2 <= theta1:
        # Start/end points
        p1 = maneuver.qi[:2]
        p2 = maneuver.qf[:2]

        radius = maneuver.rhomin

        c1, s1 = radius * math.cos(theta1), radius * math.sin(theta1)
        c2, s2 = radius * math.cos(theta2), radius * math.sin(theta2)

        # Origins of the turns
        o1 = p1 + np.array([s1, -c1])
        o2 = p2 + np.array([s2, -c2])

        diff = o2 - o1
        center_distance = math.sqrt(diff[0]**2 + diff[1]**2)
        centerAngle = math.atan2(diff[1], diff[0])
                
        t = (theta1 - centerAngle) % (2 * math.pi)
        p = center_distance / radius
        q = (-theta2 + centerAngle) % (2 * math.pi)

        if t > math.pi:
            t = 0.0
            q = -theta2 + theta1
            turn_end_y = o2[1] + radius * math.cos(theta1)
            diff_y = turn_end_y - p1[1]
            if abs(theta1) > 1e-5 and (diff_y < 0) == (theta1 < 0):
                p = diff_y / math.sin(theta1) / radius
            else:
                t = p = q = float('inf')
                
        if q > math.pi:
            t = -theta2 + theta1
            q = 0.0
            turn_end_y = o1[1] + radius * math.cos(theta2)
            diff_y = p2[1] - turn_end_y
            if abs(theta2) > 1e-5 and (diff_y < 0) == (theta2 < 0):
                p = diff_y / math.sin(theta2) / radius
            else:
                t = p = q = float('inf')
    else:
        t = p = q = float('inf')
    
    length = (t + p + q) * maneuver.rhomin
    case = "RSR"
    
    return DubinsStruct(t, p, q, length, case)

def _LSR(maneuver: DubinsManeuver2D, pitchmax: Tuple[float, float]) -> DubinsStruct:
    """LSR path computation for vertical maneuver with pitch constraints"""
    theta1 = maneuver.qi[2]
    theta2 = maneuver.qf[2]

    # Start/end points
    p1 = maneuver.qi[:2]
    p2 = maneuver.qf[:2]

    radius = maneuver.rhomin

    c1, s1 = radius * math.cos(theta1), radius * math.sin(theta1)
    c2, s2 = radius * math.cos(theta2), radius * math.sin(theta2)

    # Origins of the turns
    o1 = p1 + np.array([-s1, c1])
    o2 = p2 + np.array([s2, -c2])

    diff = o2 - o1
    center_distance = math.sqrt(diff[0]**2 + diff[1]**2)

    # Not constructible
    if center_distance < 2 * radius:
        diff[0] = math.sqrt(4.0 * radius * radius - diff[1] * diff[1])
        alpha = math.pi / 2.0
    else:
        alpha = math.asin(2.0 * radius / center_distance)
        
    centerAngle = math.atan2(diff[1], diff[0]) + alpha

    if centerAngle < pitchmax[1]:
        t = (-theta1 + centerAngle) % (2 * math.pi)
        p = math.sqrt(max(0.0, center_distance * center_distance - 4.0 * radius * radius)) / radius
        q = (-theta2 + centerAngle) % (2 * math.pi)
    else:
        centerAngle = pitchmax[1]
        t = (-theta1 + centerAngle) % (2 * math.pi)
        q = (-theta2 + centerAngle) % (2 * math.pi)

        # Points on boundary between C and S segments
        c1, s1 = radius * math.cos(centerAngle), radius * math.sin(centerAngle)
        c2, s2 = radius * math.cos(centerAngle), radius * math.sin(centerAngle)
        w1 = o1 - np.array([-s1, c1])
        w2 = o2 - np.array([s2, -c2])

        p = (w2[1] - w1[1]) / math.sin(centerAngle) / radius

    length = (t + p + q) * maneuver.rhomin
    case = "LSR"
    
    return DubinsStruct(t, p, q, length, case)

def _RSL(maneuver: DubinsManeuver2D, pitchmax: Tuple[float, float]) -> DubinsStruct:
    """RSL path computation for vertical maneuver with pitch constraints"""
    theta1 = maneuver.qi[2]
    theta2 = maneuver.qf[2]

    # Start/end points
    p1 = maneuver.qi[:2]
    p2 = maneuver.qf[:2]

    radius = maneuver.rhomin

    c1, s1 = radius * math.cos(theta1), radius * math.sin(theta1)
    c2, s2 = radius * math.cos(theta2), radius * math.sin(theta2)

    # Origins of the turns
    o1 = p1 + np.array([s1, -c1])
    o2 = p2 + np.array([-s2, c2])

    diff = o2 - o1
    center_distance = math.sqrt(diff[0]**2 + diff[1]**2)

    # Not constructible
    if center_distance < 2 * radius:
        diff[0] = math.sqrt(4.0 * radius * radius - diff[1] * diff[1])
        alpha = math.pi / 2.0
    else:
        alpha = math.asin(2.0 * radius / center_distance)
        
    centerAngle = math.atan2(diff[1], diff[0]) - alpha

    if centerAngle > pitchmax[0]:
        t = (theta1 - centerAngle) % (2 * math.pi)
        p = math.sqrt(max(0.0, center_distance * center_distance - 4.0 * radius * radius)) / radius
        q = (theta2 - centerAngle) % (2 * math.pi)
    else:
        centerAngle = pitchmax[0]
        t = (theta1 - centerAngle) % (2 * math.pi)
        q = (theta2 - centerAngle) % (2 * math.pi)

        # Points on boundary between C and S segments
        c1, s1 = radius * math.cos(centerAngle), radius * math.sin(centerAngle)
        c2, s2 = radius * math.cos(centerAngle), radius * math.sin(centerAngle)
        w1 = o1 - np.array([s1, -c1])
        w2 = o2 - np.array([-s2, c2])

        p = (w2[1] - w1[1]) / math.sin(centerAngle) / radius

    length = (t + p + q) * maneuver.rhomin
    case = "RSL"
    
    return DubinsStruct(t, p, q, length, case)
