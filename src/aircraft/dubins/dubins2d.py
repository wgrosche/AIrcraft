"""
Classical 2D Dubins Curve
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

@dataclass
class DubinsStruct:
    t: float
    p: float
    q: float
    length: float
    case: str

class DubinsManeuver2D:
    """
    Classical 2D Dubins Curve
    """
    def __init__(self, qi: List[float], qf: List[float], rhomin: float = 1.0, 
                 minLength: Optional[float] = None, disable_CCC: bool = False):
        self.qi = np.array(qi)
        self.qf = np.array(qf)
        self.rhomin = rhomin
        self.maneuver = DubinsStruct(0.0, 0.0, 0.0, float('inf'), "")
        
        self._compute_maneuver(minLength, disable_CCC)
    
    def _compute_maneuver(self, minLength: Optional[float], disable_CCC: bool):
        dx = self.qf[0] - self.qi[0]
        dy = self.qf[1] - self.qi[1]
        D = math.sqrt(dx**2 + dy**2)
        
        # Distance normalization
        d = D / self.rhomin
        
        # Normalize the problem using rotation
        rotationAngle = math.atan2(dy, dx) % (2 * math.pi)
        a = (self.qi[2] - rotationAngle) % (2 * math.pi)
        b = (self.qf[2] - rotationAngle) % (2 * math.pi)
        
        sa, ca = math.sin(a), math.cos(a)
        sb, cb = math.sin(b), math.cos(b)
        
        # CSC paths
        pathLSL = self._LSL(a, b, d, sa, ca, sb, cb)
        pathRSR = self._RSR(a, b, d, sa, ca, sb, cb)
        pathLSR = self._LSR(a, b, d, sa, ca, sb, cb)
        pathRSL = self._RSL(a, b, d, sa, ca, sb, cb)
        
        if disable_CCC:
            _paths = [pathLSL, pathRSR, pathLSR, pathRSL]
        else:
            # CCC paths
            pathRLR = self._RLR(a, b, d, sa, ca, sb, cb)
            pathLRL = self._LRL(a, b, d, sa, ca, sb, cb)
            _paths = [pathLSL, pathRSR, pathLSR, pathRSL, pathRLR, pathLRL]
        
        # Special case for very small distances and angles
        if (abs(d) < self.rhomin * 1e-5 and abs(a) < self.rhomin * 1e-5 and 
            abs(b) < self.rhomin * 1e-5):
            dist_2D = max(abs(self.qi[0] - self.qf[0]), abs(self.qi[1] - self.qf[1]))
            if dist_2D < self.rhomin * 1e-5:
                pathC = self._C()
                _paths = [pathC]
        
        # Sort paths by length
        _paths.sort(key=lambda x: x.length)
        
        if minLength is None:
            self.maneuver = _paths[0]
        else:
            self.maneuver = None
            for p in _paths:
                if p.length >= minLength:
                    self.maneuver = p
                    break
            
            if self.maneuver is None:
                self.maneuver = DubinsStruct(float('inf'), float('inf'), 
                                           float('inf'), float('inf'), "XXX")

    def _LSL(self, a: float, b: float, d: float, sa: float, ca: float, 
             sb: float, cb: float) -> DubinsStruct:
        """LSL path computation"""
        aux = math.atan2(cb - ca, d + sa - sb)
        t = (-a + aux) % (2 * math.pi)
        p = math.sqrt(2 + d**2 - 2*math.cos(a-b) + 2*d*(sa-sb))
        q = (b - aux) % (2 * math.pi)
        length = (t + p + q) * self.rhomin
        return DubinsStruct(t, p, q, length, "LSL")

    def _RSR(self, a: float, b: float, d: float, sa: float, ca: float, 
             sb: float, cb: float) -> DubinsStruct:
        """RSR path computation"""
        aux = math.atan2(ca - cb, d - sa + sb)
        t = (a - aux) % (2 * math.pi)
        p = math.sqrt(2 + d**2 - 2*math.cos(a-b) + 2*d*(sb-sa))
        q = ((-b) % (2 * math.pi) + aux) % (2 * math.pi)
        length = (t + p + q) * self.rhomin
        return DubinsStruct(t, p, q, length, "RSR")

    def _LSR(self, a: float, b: float, d: float, sa: float, ca: float, 
             sb: float, cb: float) -> DubinsStruct:
        """LSR path computation"""
        aux1 = -2 + d**2 + 2*math.cos(a-b) + 2*d*(sa+sb)
        if aux1 > 0:
            p = math.sqrt(aux1)
            aux2 = math.atan2(-ca-cb, d+sa+sb) - math.atan2(-2, p)
            t = (-a + aux2) % (2 * math.pi)
            q = ((-b) % (2 * math.pi) + aux2) % (2 * math.pi)
        else:
            t = p = q = float('inf')
        length = (t + p + q) * self.rhomin
        return DubinsStruct(t, p, q, length, "LSR")

    def _RSL(self, a: float, b: float, d: float, sa: float, ca: float, 
             sb: float, cb: float) -> DubinsStruct:
        """RSL path computation"""
        aux1 = d**2 - 2 + 2*math.cos(a-b) - 2*d*(sa+sb)
        if aux1 > 0:
            p = math.sqrt(aux1)
            aux2 = math.atan2(ca+cb, d-sa-sb) - math.atan2(2, p)
            t = (a - aux2) % (2 * math.pi)
            q = (b - aux2) % (2 * math.pi)
        else:
            t = p = q = float('inf')
        length = (t + p + q) * self.rhomin
        return DubinsStruct(t, p, q, length, "RSL")

    def _RLR(self, a: float, b: float, d: float, sa: float, ca: float, 
             sb: float, cb: float) -> DubinsStruct:
        """RLR path computation"""
        aux = (6 - d**2 + 2*math.cos(a-b) + 2*d*(sa-sb)) / 8
        if abs(aux) <= 1:
            p = (-math.acos(aux)) % (2 * math.pi)
            t = (a - math.atan2(ca-cb, d-sa+sb) + p/2) % (2 * math.pi)
            q = (a - b - t + p) % (2 * math.pi)
        else:
            t = p = q = float('inf')
        length = (t + p + q) * self.rhomin
        return DubinsStruct(t, p, q, length, "RLR")

    def _LRL(self, a: float, b: float, d: float, sa: float, ca: float, 
             sb: float, cb: float) -> DubinsStruct:
        """LRL path computation"""
        aux = (6 - d**2 + 2*math.cos(a-b) + 2*d*(-sa+sb)) / 8
        if abs(aux) <= 1:
            p = (-math.acos(aux)) % (2 * math.pi)
            t = (-a + math.atan2(-ca+cb, d+sa-sb) + p/2) % (2 * math.pi)
            q = (b - a - t + p) % (2 * math.pi)
        else:
            t = p = q = float('inf')
        length = (t + p + q) * self.rhomin
        return DubinsStruct(t, p, q, length, "LRL")

    def _C(self) -> DubinsStruct:
        """Circle path computation"""
        t = 0.0
        p = 2 * math.pi
        q = 0.0
        length = (t + p + q) * self.rhomin
        return DubinsStruct(t, p, q, length, "RRR")

    def getCoordinatesAt(self, offset: float) -> List[float]:
        """Get coordinates at a given offset along the path"""
        # Normalized offset
        noffset = offset / self.rhomin
        
        # Translation to origin
        qi = [0.0, 0.0, self.qi[2]]
        
        # Generate intermediate configurations
        l1 = self.maneuver.t
        l2 = self.maneuver.p
        q1 = self.getPositionInSegment(l1, qi, self.maneuver.case[0])  # End of segment 1
        q2 = self.getPositionInSegment(l2, q1, self.maneuver.case[1])  # End of segment 2
        
        # Get remaining configurations
        if noffset < l1:
            q = self.getPositionInSegment(noffset, qi, self.maneuver.case[0])
        elif noffset < (l1 + l2):
            q = self.getPositionInSegment(noffset - l1, q1, self.maneuver.case[1])
        else:
            q = self.getPositionInSegment(noffset - l1 - l2, q2, self.maneuver.case[2])
        
        # Translation to previous position
        q[0] = q[0] * self.rhomin + self.qi[0]
        q[1] = q[1] * self.rhomin + self.qi[1]
        q[2] = q[2] % (2 * math.pi)
        
        return q

    def getPositionInSegment(self, offset: float, qi: List[float], case: str) -> List[float]:
        """Get position in a specific segment"""
        q = [0.0, 0.0, 0.0]
        if case == 'L':
            q[0] = qi[0] + math.sin(qi[2] + offset) - math.sin(qi[2])
            q[1] = qi[1] - math.cos(qi[2] + offset) + math.cos(qi[2])
            q[2] = qi[2] + offset
        elif case == 'R':
            q[0] = qi[0] - math.sin(qi[2] - offset) + math.sin(qi[2])
            q[1] = qi[1] + math.cos(qi[2] - offset) - math.cos(qi[2])
            q[2] = qi[2] - offset
        elif case == 'S':
            q[0] = qi[0] + math.cos(qi[2]) * offset
            q[1] = qi[1] + math.sin(qi[2]) * offset
            q[2] = qi[2]
        return q

    def getSamplingPoints(self, res: float = 0.1) -> List[List[float]]:
        """Get sampling points along the path"""
        points = []
        offset = 0.0
        while offset <= self.maneuver.length:
            points.append(self.getCoordinatesAt(offset))
            offset += res
        return points
