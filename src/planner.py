from casadi import MX, DM, vertcat, horzcat, veccat, norm_2, dot, mtimes, nlpsol, diag, repmat, sum1
import numpy as np
import inspect
from scipy.spatial.transform import Rotation as R
import dubins

class Track:
  def __init__(self, filename = ""):
    self.init_pos = [0, 0, -200]
    self.init_att = [0, 0, 0, 1]
    self.init_vel = [35, 0, 0]
    self.init_omega = [0, 0, 0]
    self.end_pos = [200, 200, -180]
    self.end_att = None
    self.end_vel = None
    self.end_omega = None
    self.gates = [[100, 0, -190], [150, 200, -170]]
    self.waypoint_indices = [0, 1]

import numpy as np

def generate_dubins_path(waypoints, r_min):
    """
    Generates a Dubins path for a given list of waypoints, ensuring the minimum turn radius is respected.

    :param waypoints: List of waypoints [(x, y, theta), ...] where theta is in radians.
    :param r_min: Minimum turn radius.
    :return: List of (x, y, theta) points forming the Dubins path.
    """
    path_points = []

    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]

        # Generate Dubins path between two waypoints
        path, _ = dubins.shortest_path(start, end, r_min).sample_many(0.1)  # Sample every 0.1m
        path_points.extend(path)

    return path_points

# Example usage:
# waypoints = [(0, 0, 0), (10, 10, np.pi/4), (20, 5, np.pi/2)]  # (x, y, heading in radians)
# r_min = 2.0  # Minimum turn radius
# trajectory = generate_dubins_path(waypoints, r_min)

# # Print or visualize trajectory
# for point in trajectory:
#     print(point)



class Planner:
  def __init__(self, aircraft, track = None, options = {'tolerance': 1.0, 'nodes_per_gate': 30, 'vel_guess': 35.0}):
    # Essentials
    # self.quad = quad
    self.track = Track()
    track = self.track

    self.options = options

    # Track
    self.wp = DM(track.gates).T
    if track.end_pos is not None:
      if len(track.gates) > 0:
        self.wp = horzcat(self.wp, DM(track.end_pos))
      else:
        self.wp = DM(track.end_pos)

    if track.init_pos is not None:
      self.p_init = DM(track.init_pos)
    else:
      self.p_init = self.wp[:,-1]
    if track.init_att is not None:
      self.q_init = DM(track.init_att)
    else:
      self.q_init = DM([0, 0, 0, 1]).T

    self.aircraft = aircraft

    # Dynamics
    self.dynamics = aircraft.state_update.expand()

    # Sizes
    self.NX = self.dynamics.size1_in(0)
    self.NU = self.dynamics.size1_in(1)
    self.NW = self.wp.shape[1]

    dist = [np.linalg.norm(self.wp[:,0] - self.p_init)]
    for i in range(self.NW-1):
      dist += [dist[i] + np.linalg.norm(self.wp[:,i+1] - self.wp[:,i])]

    if 'nodes_per_gate' in options:
      self.NPW = options['nodes_per_gate']
    else:
      self.NPW = 30

    if 'tolerance' in options:
      self.tol = options['tolerance']
    else:
      self.tol = 0.3

    self.N = self.NPW * self.NW
    self.dpn = dist[-1] / self.N
    if self.dpn < self.tol:
      suff_str = "sufficient"
    else:
      suff_str = "insufficient"
    print("Discretization over %d nodes and %1.1fm" % (self.N, dist[-1]))
    print("results in %1.3fm per node, %s for tolerance of %1.3fm." % (self.dpn, suff_str, self.tol))

    self.i_switch = np.array(self.N * np.array(dist) / dist[-1], dtype=int)

    # Problem variables
    self.x = []
    self.xg = []
    self.g = []
    self.lb = []
    self.ub = []
    self.J = []

    # Solver
    if 'solver_options' in options:
      self.solver_options = options['solver_options']
    else:
      self.solver_options = {}
      ipopt_options = {}
      ipopt_options['max_iter'] = 10000
      self.solver_options['ipopt'] = ipopt_options
      # self.solver_options['ipopt']['hessian_approximation']= 'limited-memory'
      # self.solver_options['expand'] = True
    if 'solver_type' in options:
      self.solver_type = options['solver_type']
    else:
      self.solver_type = 'ipopt'
    self.iteration_plot = []

    # Guesses
    if 't_guess' in options:
      self.t_guess = options['t_guess']
      self.vel_guess = dist[-1] / self.t_guess
    elif 'vel_guess' in options:
      self.vel_guess = options['vel_guess']
      self.t_guess = dist[-1] / self.vel_guess
    else:
      self.vel_guess = 2.5
      self.t_guess = dist[-1] / self.vel_guess

    if 'legacy_init' in options:
      if options['legacy_init']:
        self.i_switch = range(self.NPW,self.N+1,self.NPW)

  def set_iteration_callback(self, callback):
    self.iteration_callback = callback

  def set_init_pos(self, position):
    self.p_init = DM(position)

  def set_init_att(self, quaternion):
    self.q_init = DM(quaternion)

  def set_t_guess(self, t):
    self.t_guess

  def set_initial_guess(self, x_guess):
    if len(x_guess) > 0 and (self.xg.shape[0] == len(x_guess) or self.xg.shape[0] == 0):
      self.xg = veccat(*x_guess)

  def setup(self):
    x = []
    xg = []
    g = []
    lb = []
    ub = []
    J = 0

    # Total time variable
    t = MX.sym('t', 1)
    x += [t]
    xg += [self.t_guess]
    g += [t]
    lb += [0.1]
    ub += [1000]
    J = t

    # Bound initial state to x0
    xk = MX.sym('x_init', self.NX)
    vel_guess = (self.wp[:,0] - self.p_init)
    vel_guess = self.vel_guess * vel_guess / norm_2(vel_guess)
    x0 = [self.p_init[0], self.p_init[1], self.p_init[2], vel_guess[0], vel_guess[1], vel_guess[2], self.q_init[0], self.q_init[1], self.q_init[2], self.q_init[3], 0, 0, 0]
    pos_guess = self.p_init
    if self.track.init_vel is not None:
      x0[3:6] = self.track.init_vel
    x += [xk]
    xg += x0
    if self.track.init_pos is not None:
    # if self.track.init_pos is not None and not self.track.ring:
      print('Using start position constraint')
      g += [xk[0:3]]
      ub += [self.track.init_pos]
      lb += [self.track.init_pos]
    if self.track.init_vel is not None:
      print('Using start velocity constraint')
      g += [xk[3:6]]
      ub += [self.track.init_vel]
      lb += [self.track.init_vel]
    if self.track.init_att is not None:
      print('Using start attitude constraint')
      g += [xk[6:10]]
      ub += [self.track.init_att]
      lb += [self.track.init_att]
    else:
      g += [dot(xk[6:10], xk[6:10])]
      ub += [1.0]
      lb += [1.0]
    if self.track.init_omega is not None:
      print('Using start bodyrate constraint')
      g += [xk[10:13]]
      ub += [self.track.init_omega]
      lb += [self.track.init_omega]
    x_init = xk

    # Bound inital progress variable to 1
    muk = MX.sym('mu_init', self.NW)
    x += [muk]
    xg += [1]*(self.NW)
    g += [muk]
    ub += [1]*(self.NW)
    lb += [1]*(self.NW)

    # For each node ...
    i_wp = 0
    for i in range(self.N):
      # ... add inputs

      vk = MX.sym('v'+str(i), self.NU)  # Unconstrained control variable
      x += [vk]
      uk = -5 + 10 * (1 / (1 + MX.exp(-vk))) # Reparametrise to [-5, 5]
      xg += [0]*self.NU

      # uk = MX.sym('u'+str(i), self.NU)
      # x += [uk]
      # xg += [0]*self.NU
      # g += [uk]
      # lb += [-5]*self.NU
      # ub += [5]*self.NU

      # ... add next state
      xn = self.dynamics(xk, uk, t/self.N)

      # angle of attack, side slip and speed constraints
      g += [self.aircraft.alpha(xk, uk)]
      lb += [np.deg2rad(-20)]
      ub += [np.deg2rad(20)]

      g += [self.aircraft.beta(xk, uk)]
      lb += [np.deg2rad(-20)]
      ub += [np.deg2rad(20)]

      g += [self.aircraft.airspeed(xk, uk)]
      lb += [20]
      ub += [80]

      xk = MX.sym('x'+str(i), self.NX)
      x += [xk]
      g += [xk - xn]
      lb += [0] * self.NX
      ub += [0] * self.NX

      # if i >= (1 + i_wp) * self.NPW:
      if i > self.i_switch[i_wp]:
        i_wp += 1
      if i_wp == 0:
        wp_last = self.p_init
      else:
        wp_last = self.wp[:, i_wp-1]
      wp_next = self.wp[:, i_wp]
      if i_wp > 0:
        interp = (i - self.i_switch[i_wp-1]) / (self.i_switch[i_wp] - self.i_switch[i_wp-1])
      else:
        interp = i / self.i_switch[0]
      pos_guess = (1-interp) * wp_last + interp * wp_next
      vel_guess = self.vel_guess * (wp_next - wp_last)/norm_2(wp_next - wp_last)
      # direction = (wp_next - wp_last)/norm_2(wp_next - wp_last)
      # if interp < 0.5:
      #   vel_guess = interp * 4 * self.vel_guess * direction
      # else:
      #   vel_guess = (1 - interp) * 4 * self.vel_guess * direction
      # pos_guess += self.t_guess / self.N * vel_guess
      direction = (wp_next - wp_last)/norm_2(wp_next - wp_last)
      rotation, _ = R.align_vectors(np.array(direction).reshape(1, -1), [[1, 0, 0]])
      if np.dot(direction.T, [1, 0, 0]) < 0:
                flip_y = R.from_euler('y', 180, degrees=True)
                rotation = rotation * flip_y

      orientation_guess = (rotation).as_quat()

      xg += [pos_guess, vel_guess, orientation_guess, [0]*3]

      # Progress Variables
      lam = MX.sym('lam'+str(i), self.NW)
      x += [lam]
      g += [lam]
      lb += [0]*self.NW
      ub += [1]*self.NW
      if ((i_wp == 0) and (i + 1 >= self.i_switch[0])) or i + 1 - self.i_switch[i_wp-1] >= self.i_switch[i_wp]:
        lamg = [0] * self.NW
        lamg[i_wp] = 1.0
        xg += lamg
      else:
        xg += [0] * self.NW

      tau = MX.sym('tau'+str(i), self.NW)
      x += [tau]
      g += [tau]
      lb += [0]*self.NW
      ub += [self.tol**2]*(self.NW)
      xg += [0] * self.NW

      for j in range(self.NW):
        # diff = xk[0:3] - self.wp[:,j]
        diff = xk[self.track.waypoint_indices] - self.wp[self.track.waypoint_indices,j]
        g += [lam[j] * (dot(diff, diff)-tau[j])]
      lb += [0]*self.NW
      ub += [0.01]*self.NW

      mul = muk
      muk = MX.sym('mu'+str(i), self.NW)
      x += [muk]
      g += [mul - lam - muk]
      lb += [0]*self.NW
      ub += [0]*self.NW

      for j in range(self.NW):
        # if i >= ((j+1)*self.NPW - 1):
        if i+1 >= self.i_switch[j]:
          xg += [0]
        else:
          xg += [1]

      # # Bind rates
      # g += [xk[10:13]]
      # lb += [-omega_max_xy, -omega_max_xy, -self.quad.omega_max_z]
      # ub += [omega_max_xy, omega_max_xy, self.quad.omega_max_z]

      # z constraint
      g += [xk[2]]
      lb += [-200.0] 
      ub += [-0.5]           # infinity

      for j in range(self.NW-1):
        g += [muk[j+1]-muk[j]]
        lb += [0]
        ub += [1]
    # end for loop #############################################################

    g += [muk]
    lb += [0]*self.NW
    ub += [0]*self.NW

    # if self.track.ring:
    #   print('Using ring constraint')
    #   g += [xk[3:6] - x_init[3:6]]
    #   lb += [0] * 3 #self.NX
    #   ub += [0] * 3 #self.NX

    # if self.track.end_att is not None:
    #   print('Using end attitude constraint')
    #   g += [xk[6:10]]
    #   lb += [self.track.end_att]
    #   ub += [self.track.end_att]

    # if self.track.end_vel is not None:
    #   print('Using end velocity constraint')
    #   g += [xk[3:6]]
    #   lb += [self.track.end_vel]
    #   ub += [self.track.end_vel]

    # if self.track.end_omega is not None:
    #   print('Using end bodyrate constraint')
    #   g += [xk[10:13]]
    #   lb += [self.track.end_omega]
    #   ub += [self.track.end_omega]

    # Reformat
    self.x = vertcat(*x)
    if not self.xg:
      self.xg = xg
    self.xg = veccat(*self.xg)
    self.g = vertcat(*g)
    self.lb = veccat(*lb)
    self.ub = veccat(*ub)
    self.J = J

    # Construct Non-Linear Program
    self.nlp = {'f': self.J, 'x': self.x, 'g': self.g}

  def solve(self, x_guess = []):
    self.set_initial_guess(x_guess)

    if hasattr(self, 'iteration_callback') and self.iteration_callback is not None:
      if inspect.isclass(self.iteration_callback):
        callback = self.iteration_callback("IterationCallback")
      elif self.iteration_callback:
        callback = self.iteration_callback

      callback.set_size(self.x.shape[0], self.g.shape[0], self.NPW)
      callback.set_wp(self.wp)
      self.solver_options['iteration_callback'] = callback
      # self.solver_options['expand'] = True


    self.solver = nlpsol('solver', self.solver_type, self.nlp, self.solver_options)

    self.solution = self.solver(x0=self.xg, lbg=self.lb, ubg=self.ub)
    self.x_sol = self.solution['x'].full().flatten()
    return self.x_sol

  def interpolate(self, x1, y1, x2, y2, x):
    if (abs(x2 - x1) < 1e-5):
      return 0

    return y1 + (y2 - y1)/(x2 - x1) * (x - x1)

