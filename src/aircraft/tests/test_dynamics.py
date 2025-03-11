import unittest
import casadi as ca
import numpy as np
from aircraft.dynamics.dynamics import Aircraft 
from aircraft.utils.utils import load_model
from liecasadi import Quaternion
import torch
import os
from aircraft.config import DEVICE, BASEPATH, NETWORKPATH


class TestAircraftInitialization(unittest.TestCase):
    
    def setUp(self):
        # Example parameters for aircraft
        params = {
            'Ixx': 1000, 'Iyy': 1200, 'Izz': 1500, 'Ixz': 100,
            'reference_area': 20.0, 'span': 10.0, 'chord': 2.0,
            'mass': 1500.0
        }

        model = load_model(filepath = os.path.join(NETWORKPATH,'model-dynamics.pth'), device = DEVICE)  # Placeholder for NN model
        self.aircraft = Aircraft(params, model)

    def test_initialization(self):
        self.assertEqual(self.aircraft.state.size()[0], 13)
        self.assertEqual(self.aircraft.control.size()[0], 15)
        # self.assertAlmostEqual(self.aircraft.mass, 1500.0)



class TestQuaternionIntegration(unittest.TestCase):
    
    def setUp(self):
        # Example parameters for aircraft
        params = {
            'Ixx': 1000, 'Iyy': 1200, 'Izz': 1500, 'Ixz': 100,
            'reference_area': 20.0, 'span': 10.0, 'chord': 2.0,
            'mass': 1500.0
        }
        model = load_model(filepath = os.path.join(NETWORKPATH,'model-dynamics.pth'), device = DEVICE)  # Placeholder for NN model
        self.aircraft = Aircraft(params, model)

    def test_quaternion_integration_zero_angular_velocity(self):
        # Test for zero angular velocity
        state = ca.DM([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        dt = 0.1
        new_quaternion = self.aircraft.integrate_quaternion(state, dt)
        np.testing.assert_array_almost_equal(new_quaternion.full().flatten(), [1, 0, 0, 0], decimal=6)
    
    def test_quaternion_integration_nonzero_angular_velocity(self):
        # Test for non-zero angular velocity
        state = ca.DM([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1])
        dt = 0.1
        new_quaternion = self.aircraft.integrate_quaternion(state, dt)
        self.assertAlmostEqual(ca.norm_2(new_quaternion).full()[0], 1.0, places=6)


class TestRelativeVelocity(unittest.TestCase):

    def setUp(self):
        # Example parameters for aircraft
        params = {
            'Ixx': 1000, 'Iyy': 1200, 'Izz': 1500, 'Ixz': 100,
            'reference_area': 20.0, 'span': 10.0, 'chord': 2.0,
            'mass': 1500.0
        }
        model = load_model(filepath = os.path.join(NETWORKPATH,'model-dynamics.pth'), device = DEVICE)  # Placeholder for NN model
        self.aircraft = Aircraft(params, model)

    def test_relative_velocity_no_wind(self):
        state = ca.DM([1, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0])
        control = ca.DM([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        v_frd_rel = self.aircraft._v_frd_rel
        relative_velocity = v_frd_rel(state, control)
        np.testing.assert_array_almost_equal(relative_velocity.full().flatten(), [10, 0, 0], decimal=6)

    def test_relative_velocity_with_wind(self):
        state = ca.DM([1, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0])
        control = ca.DM([0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0])
        v_frd_rel = self.aircraft._v_frd_rel
        relative_velocity = v_frd_rel(state, control)
        np.testing.assert_array_almost_equal(relative_velocity.full().flatten(), [5, 0, 0], decimal=6)


class TestForcesAndMoments(unittest.TestCase):

    def setUp(self):
        # Example parameters for aircraft
        params = {
            'Ixx': 1000, 'Iyy': 1200, 'Izz': 1500, 'Ixz': 100,
            'reference_area': 20.0, 'span': 10.0, 'chord': 2.0,
            'mass': 1500.0
        }

        model = load_model(filepath = os.path.join(NETWORKPATH,'model-dynamics.pth'), device = DEVICE)  # Placeholder for NN model
        self.aircraft = Aircraft(params, model)

    def test_forces_calculation(self):
        state = ca.DM([1, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0])
        control = ca.DM([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        forces_frd = self.aircraft._forces_frd
        forces = forces_frd(state, control)
        # Add expected forces comparison based on model coefficients
        
    def test_moments_calculation(self):
        state = ca.DM([1, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0])
        control = ca.DM([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        moments_frd = self.aircraft._moments_frd
        moments = moments_frd(state, control)
        # Add expected moments comparison based on model coefficients
