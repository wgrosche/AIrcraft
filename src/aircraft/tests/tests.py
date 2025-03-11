# # """
# # Unit tests for the application.

# # """


# import unittest
# import casadi as ca
# import numpy as np

# # class TestCasadiJacobian(unittest.TestCase):
# #     def test_jacobian(self):
# #         # Define the symbolic variables
# #         x = ca.MX.sym('x')
# #         y = ca.MX.sym('y')
        
# #         # Define the test function f(x, y) = [x^2 + y^2, x*y]
# #         f = ca.vertcat(x**2 + y**2, x * y)
# #         # f = ca.atan2(y, x)
        
# #         # Compute the Jacobian using CasADi
# #         jacobian_f = ca.jacobian(f, ca.vertcat(x, y))
        
# #         # Create a CasADi function to evaluate the Jacobian
# #         jacobian_func = ca.Function('jacobian_func', [ca.vertcat(x, y)], [jacobian_f])
        
# #         print(jacobian_f)
# #         # Define the point at which to evaluate the Jacobian
# #         point = ca.DM([1.0, 2.0])
        
# #         # Evaluate the Jacobian at the point
# #         jacobian_eval = jacobian_func(point)
        
# #         # Expected Jacobian at the point (1, 2)
# #         # df/dx = [2x, y] => [2*1, 2] => [2, 2]
# #         # df/dy = [2y, x] => [2*2, 1] => [4, 1]
# #         # expected_jacobian = np.array([[2.0, 4.0],
# #         #                               [2.0, 1.0]])
# #         expected_jacobian = np.array([[-0.4, 0.2]])

# #         print(jacobian_eval)
        
# #         # Compare the evaluated Jacobian with the expected Jacobian
# #         np.testing.assert_array_almost_equal(jacobian_eval.full(), expected_jacobian, decimal=5)

# # # class Aircraft:
# # #     def __init__(self):
# # #         # Define the state variables symbolically
# # #         self.state = ca.MX.sym('state', 12)

# # #     @property
# # #     def alpha(self) -> ca.MX:
# # #         u = self.state[3]
# # #         w = self.state[5]
# # #         alpha = ca.atan2(w, u)
# # #         self._alpha = alpha
# # #         return self._alpha

# # #     @property
# # #     def alpha_function(self) -> ca.Function:
# # #         return ca.Function('compute_alpha', [self.state], [self.alpha])



# # if __name__ == '__main__':
# #     unittest.main()

# # #     # Create an instance of the Aircraft class
# # #     aircraft = Aircraft()

# # #     # Define the trim state (example values, replace with actual trim state)
# # #     trim_state_control = ca.DM([0, 0, 0, 50, 0, 5, 0, 0, 0, 0, 0, 0])

# # #     # Symbolic state variable
# # #     state_sym = aircraft.state

# # #     # Evaluate alpha at the trim state
# # #     alpha_equil = aircraft.alpha_function(trim_state_control[:12])
# # #     print("Alpha:", alpha_equil)

# # #     # Compute the Jacobian of alpha with respect to the state
# # #     alpha_jac = ca.jacobian(alpha_equil, state_sym)

# # #     # Print symbolic expressions for debugging
# # #     print("Symbolic alpha expression:", aircraft.alpha)
# # #     print("Symbolic alpha Jacobian expression:", alpha_jac)

# # #     # Create a function to evaluate the Jacobian
# # #     alpha_jac_func = ca.Function('alpha_jac_func', [state_sym], [alpha_jac])

# # #     # Evaluate the Jacobian at the trim state
# # #     alpha_jac_eval = alpha_jac_func(trim_state_control[:12])
# # #     print("Alpha Jacobian Eval:", alpha_jac_eval)


# # import casadi as ca
# # import numpy as np

# # class Aircraft:
# #     def __init__(self):
# #         # Define the state variables symbolically
# #         self.state = ca.MX.sym('state', 12)
# #         self.u = ca.MX.sym('u')
# #         self.w = ca.MX.sym('w')

# #     @property
# #     def alpha(self) -> ca.MX:
# #         # u = ca.MX.sym('u') #self.state[3]
# #         # w = ca.MX.sym('w') #self.state[5]
# #         alpha = ca.atan2(self.w, self.u)
# #         return alpha

# #     @property
# #     def alpha_function(self) -> ca.Function:
# #         # u = ca.MX.sym('u') #self.state[3]
# #         # w = ca.MX.sym('w')
# #         return ca.Function('compute_alpha', [self.u, self.w], [self.alpha])

# # # Create an instance of the Aircraft class
# # aircraft = Aircraft()

# # # Define the trim state (example values, replace with actual trim state)
# # trim_state_control = ca.DM([0, 0, 0, 50, 0, 5, 0, 0, 0, 0, 0, 0])

# # # Symbolic state variable
# # state_sym = aircraft.state
# # u_sym  = ca.MX.sym('u')
# # w_sym  = ca.MX.sym('w')

# # # Evaluate alpha at the trim state
# # # alpha_equil = aircraft.alpha_function(trim_state_control[:12])
# # alpha_equil = aircraft.alpha_function(50, 5)
# # alpha_pedestrian = ca.Function('compute_alpha', [u_sym, w_sym], [ca.atan2(w_sym, u_sym)])(50, 5)
# # print("Alpha:", alpha_equil)

# # # Compute the Jacobian of alpha with respect to the state
# # # alpha_jac = ca.jacobian(alpha_equil, state_sym)
# # alpha_jac = ca.jacobian(alpha_equil, ca.vertcat(u_sym, w_sym))
# # alpha_jac_pedestrian = ca.jacobian(alpha_pedestrian, ca.vertcat(u_sym, w_sym))

# # # Print symbolic expressions for debugging
# # print("Symbolic alpha expression:", aircraft.alpha)
# # print("Symbolic alpha Jacobian expression:", alpha_jac)

# # # Print the pedestrian's jacobian
# # print("Pedestrian's Jacobian:", alpha_jac_pedestrian)


# # # Create a function to evaluate the Jacobian
# # alpha_jac_func = ca.Function('alpha_jac_func', [u_sym, w_sym], [alpha_jac])
# # # alpha_jac_func = ca.Function('alpha_jac_func', [state_sym], [alpha_jac])

# # # Evaluate the Jacobian at the trim state
# # alpha_jac_eval = alpha_jac_func(50, 5)
# # # alpha_jac_eval = alpha_jac_func(trim_state_control[:12])
# # print("Alpha Jacobian Eval:", alpha_jac_eval)

# # # Manually check the Jacobian
# # u_val = 50
# # w_val = 5
# # d_alpha_du = -w_val / (u_val**2 + w_val**2)
# # d_alpha_dw = u_val / (u_val**2 + w_val**2)

# # expected_jacobian = np.zeros((1, 12))
# # expected_jacobian[0, 3] = d_alpha_du
# # expected_jacobian[0, 5] = d_alpha_dw

# # print("Expected Jacobian:", expected_jacobian)



# import casadi as ca
# import numpy as np

# class Aircraft:
#     def __init__(self):
#         # Define the state variables symbolically
#         self.state = ca.MX.sym('state', 12)

#     @property
#     def alpha(self) -> ca.MX:
#         u = self.state[3]
#         w = self.state[5]
#         alpha = ca.atan2(w, u)
#         return alpha

#     @property
#     def alpha_function(self) -> ca.Function:
#         return ca.Function('compute_alpha', [self.state], [ca.atan2(self.state[5], self.state[3])] )

# # Create an instance of the Aircraft class
# aircraft = Aircraft()

# # Define the trim state (example values, replace with actual trim state)
# trim_state_control = ca.DM([0, 0, 0, 50, 0, 5, 0, 0, 0, 0, 0, 0])

# # Symbolic state variable
# state_sym = ca.MX.sym('state', 12)#aircraft.state
# # Evaluate alpha at the trim state
# # alpha_equil = aircraft.alpha_function(trim_state_control[:12])

# alpha_equil = ca.Function('atan', [state_sym], [ca.atan2(state_sym[5], state_sym[3])])(state_sym)#(trim_state_control)
# print("Alpha:", type(alpha_equil))

# # Compute the Jacobian of alpha with respect to the state
# alpha_jac = ca.jacobian(alpha_equil, state_sym)

# # Print symbolic expressions for debugging
# print("Symbolic alpha expression:", aircraft.alpha)
# print("Symbolic alpha Jacobian expression:", alpha_jac)

# # Create a function to evaluate the Jacobian
# alpha_jac_func = ca.Function('alpha_jac_func', [state_sym], [alpha_jac])

# # Evaluate the Jacobian at the trim state
# alpha_jac_eval = alpha_jac_func(trim_state_control[:12])
# print("Alpha Jacobian Eval:", alpha_jac_eval)

# # Manually check the Jacobian
# u_val = 50
# w_val = 5
# d_alpha_du = -w_val / (u_val**2 + w_val**2)
# d_alpha_dw = u_val / (u_val**2 + w_val**2)

# expected_jacobian = np.zeros((1, 12))
# expected_jacobian[0, 3] = d_alpha_du
# expected_jacobian[0, 5] = d_alpha_dw

# print("Expected Jacobian:", expected_jacobian)

# # Verify with unittest
# import unittest

# class TestAircraftJacobian(unittest.TestCase):
#     def test_alpha_jacobian(self):
#         # Evaluate alpha at the trim state
#         alpha_equil = aircraft.alpha_function(trim_state_control[:12])
        
#         # Compute the Jacobian of alpha with respect to the state
#         alpha_jac = ca.jacobian(alpha_equil, state_sym)
#         alpha_jac_func = ca.Function('alpha_jac_func', [state_sym], [alpha_jac])
        
#         # Evaluate the Jacobian at the trim state
#         alpha_jac_eval = alpha_jac_func(trim_state_control[:12])
        
#         # Manually check the Jacobian
#         u_val = 50
#         w_val = 5
#         d_alpha_du = -w_val / (u_val**2 + w_val**2)
#         d_alpha_dw = u_val / (u_val**2 + w_val**2)
        
#         expected_jacobian = np.zeros((1, 12))
#         expected_jacobian[0, 3] = d_alpha_du
#         expected_jacobian[0, 5] = d_alpha_dw
        
#         # Compare the evaluated Jacobian with the expected Jacobian
#         np.testing.assert_array_almost_equal(alpha_jac_eval.full(), expected_jacobian, decimal=5)

# if __name__ == '__main__':
#     unittest.main(argv=[''], exit=False)
