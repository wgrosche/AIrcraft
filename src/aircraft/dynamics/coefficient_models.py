"""
Coefficient models to be used with the aircraft class.
"""

from __future__ import annotations
import casadi as ca
from typing import Union, TYPE_CHECKING, Dict, Callable, Optional, Any, Protocol
import numpy as np
from pathlib import Path
import pandas as pd
import pickle

from aircraft.utils.utils import load_model

if TYPE_CHECKING:
    from aircraft.dynamics.aircraft import Aircraft 

__all__ = ['CoefficientModel', 'DefaultModel', 'LinearModel', 'NeuralModel', 'PolynomialModel', 'COEFF_MODEL_REGISTRY']

class CoefficientModel:
    def __call__(self, inputs: ca.DM | ca.MX) -> ca.MX:
        raise NotImplementedError
    
class CoeffModelFactory(Protocol):
    def __call__(
        self,
        path: Union[str, Path],
        aircraft: Aircraft,
        **kwargs: Any
    ) -> CoefficientModel: ...

COEFF_MODEL_REGISTRY: Dict[str, CoeffModelFactory] = {
    "linear": lambda path, aircraft, **kwargs: LinearModel(path, aircraft),
    "poly": lambda path, aircraft, **kwargs: PolynomialModel(path, aircraft),
    "nn": lambda path, aircraft, **kwargs: NeuralModel(path, aircraft, realtime=kwargs.get("realtime", False)),
    "default": lambda path, aircraft, **kwargs: DefaultModel(aircraft),
}


    
class DefaultModel(CoefficientModel):
    def __init__(self, aircraft:Aircraft):
        self.aircraft = aircraft

    def __call__(self, inputs: ca.DM | ca.MX) -> ca.MX:

        _, alpha, beta, aileron, elevator = ca.vertsplit(inputs)
        
        p, q, r = ca.vertsplit(self.aircraft._omega_frd_ned)
        
        # Constants for control surface effectiveness
        CD0 = 0.02
        CD_alpha = 0.3

        CL0 = 0.0
        CL_alpha = 5.0  # lift per rad

        CY_beta = -0.98

        Cl_aileron = 0.08
        Cl_p = -0.05  # roll damping

        Cm_elevator = -1.2
        Cm_q = -.5  # pitch damping

        Cn_rudder = -0.1
        Cn_r = -0.05  # yaw damping

        # Core coefficient calculations
        CD = CD0 + CD_alpha * alpha**2
        CL = CL0 + CL_alpha * alpha
        CY = CY_beta * beta

        Cl = Cl_aileron * 4 * aileron  * np.pi / 180+ Cl_p * p
        Cm = Cm_elevator * 5 * elevator  * np.pi / 180 + Cm_q * q
        Cn = Cn_rudder * 6 * self.aircraft._rudder * np.pi / 180 + Cn_r * r

        return ca.MX(ca.vertcat(-CD, CY, -CL, Cl, Cm, Cn))
    
class LinearModel(CoefficientModel):
    def __init__(self, coeff_path: Union[str, Path], aircraft:Aircraft):
        self.linear_coeffs = ca.DM(np.array(pd.read_csv(coeff_path)))
        self.aircraft = aircraft

    def __call__(self, inputs: ca.DM | ca.MX) -> ca.MX:
        outputs = ca.MX(ca.mtimes(self.linear_coeffs, ca.vertcat(inputs, 1)))
        Cn_rudder = -0.1
        outputs[5] += Cn_rudder * 6 * self.aircraft._rudder * np.pi / 180
        return outputs
    
class NeuralModel(CoefficientModel):
    def __init__(self, model_path: Union[str, Path], aircraft:Aircraft, realtime: bool = False):
        from l4casadi import L4CasADi
        from l4casadi.realtime import RealTimeL4CasADi
        self.aircraft = aircraft
        model = load_model(filepath=model_path)
        self.model = RealTimeL4CasADi(model, approximation_order=1) if realtime else L4CasADi(model)

    def __call__(self, inputs: ca.DM | ca.MX) -> ca.MX:
        outputs = self.model(ca.reshape(inputs, 1, -1))
        assert isinstance(outputs, Union[ca.DM, ca.MX])
        Cn_rudder = -0.1
        outputs[5] += Cn_rudder * 6 * self.aircraft._rudder * np.pi / 180
        return ca.MX(ca.vertcat(outputs.T))
    
class PolynomialModel(CoefficientModel):
    def __init__(self, poly_path: Union[str, Path], aircraft:Aircraft):
        with open(poly_path, 'rb') as file:
            self.fitted_models = pickle.load(file)
        self.aircraft = aircraft  # to access internal states

    def __call__(self, inputs: ca.DM | ca.MX) -> ca.MX:
        fm = self.fitted_models['casadi_functions']
        outputs = ca.vertcat(*[fm[k](inputs) for k in fm.keys()])

        a = self.aircraft
        lw_inputs = ca.vertcat(a._left_wing_qbar, a._left_wing_alpha, 0, 0, 0)
        rw_inputs = ca.vertcat(a._right_wing_qbar, a._right_wing_alpha, 0, 0, 0)
        lw_lift = ca.vertcat(*[fm[k](lw_inputs) for k in fm])
        rw_lift = ca.vertcat(*[fm[k](rw_inputs) for k in fm])
        outputs[3] += a.b / 4 * (rw_lift[2]/2 - lw_lift[2]/2)

        el_inputs = ca.vertcat(a._qbar, a._elevator_alpha, a._beta, a._aileron, a._elevator)
        outputs[4] = ca.vertcat(*[fm[k](el_inputs) for k in fm])[4]

        rud_inputs = ca.vertcat(a._qbar, a._alpha, a._rudder_beta, a._aileron, a._elevator)
        outputs[5] = ca.vertcat(*[fm[k](rud_inputs) for k in fm])[5]

        Cn_rudder = -0.1
        outputs[5] += Cn_rudder * 6 * a._rudder * np.pi / 180
        return ca.MX(outputs)