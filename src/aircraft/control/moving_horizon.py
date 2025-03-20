import numpy as np
from dataclasses import dataclass
import casadi as ca
from aircraft.control.base import ControlProblem

class MHE:
    def __init__(self):
        self.nodes = 20
        self.max_dt = 1.0
        self.waypoints = []
        self.dynamics = None


        pass

    def loss(self):
        pass

    def instance(self, X_init):
        opti = ca.Opti()
        max_dist = (X_init[7:10] * self.nodes * self.max_dt)
        waypoints = self.waypoints
        state_dim = X_init.shape[0]

        initial_state = scale_state * opti.variable(scale_state.shape[0])
        opti.subject_to(initial_state == X_init)

        if np.linalg.norm(waypoints[0] - X_init[3:7]) < max_dist:
            # if waypoint is reachable within horizon
            self.current_waypoint = waypoints[0]
            self.next_waypoint = waypoints[1]

            # add waypoint constraint to opti
            for i in range(self.nodes):
                node_list.append(Node.create(opti, i))
                opti.subject_to(node_list[i].state 
                                == self.dynamics(node_list[i-1].state, node_list[i].control))

                opti.subject


                state_list.append(ca.DM(scale_state) * 
                                opti.variable(self.state_dim))
                waypoint_reached.append(opti.variable())

                if i < self.nodes:
                    control_list.append(ca.DM(scale_control) *          
                                opti.variable(self.control_dim))

        else:
             # formulate as minimising distance to next waypoint
            
        X_init
        


def setup_interval(self):
        self.current_waypoint
        self.next_waypoint
        num_nodes = 20

        # determine waypoint reachability
        if np.linalg.norm(self.current_waypoint - X) < 
        # choose number of control nodes
        # setup nodes
        for i, node in enumerate(range(num_nodes)):
             # setup node

        
        pass


def state_constraint(self, node:Node, dt:ca.MX):
    
    state_envelope = self.trajectory.state
    opti = self.opti
    dynamics = self.dynamics

    alpha = self.aircraft.alpha
    beta = self.aircraft.beta
    airspeed = self.aircraft.airspeed

    opti.subject_to(opti.bounded(state_envelope.alpha.lb,
        alpha(node.state, node.control), state_envelope.alpha.ub))

    opti.subject_to(opti.bounded(state_envelope.beta.lb,
        beta(node.state, node.control), state_envelope.beta.ub))

    opti.subject_to(opti.bounded(state_envelope.airspeed.lb,
        airspeed(node.state, node.control), state_envelope.airspeed.ub))
    
    opti.subject_to(node.state_next == dynamics(node.state, node.control, dt))



for index in range(nodes):

            node_data = self.Node(
                index=index,
                state_next = state[:, index + 1],
                state=state[:, index],
                control = control[:, index],
                lam=lam[:, index],
                lam_next=lam[:, index + 1],
                mu=mu[:, index],
                nu=nu[:, index]
            )
                
            self.state_constraint(node_data, dt)
            
            self.control_constraint(node_data)
            self.waypoint_constraint(node_data)#, waypoint_node)


def setup_opti_vars(self, 
                        scale_state = ca.vertcat(
                            [1, 1, 1, 1],
                            [1e3, 1e3, 1e3],
                            [1e2, 1e2, 1e2],
                            [1, 1, 1]
                            ), 
                        scale_control = ca.vertcat(
                            5, 5, 5,
                            [1e2, 1e2, 1e2],
                            [1, 1, 1],
                            [1e2, 1e2, 1e2]
                            ), 
                        scale_time = 1,
                        ):
        
        opti = self.opti
        self.time = scale_time * opti.variable()
        self.dt = self.time / self.nodes

        
        state_list = []
        control_list = []
        lam_list = []
        mu_list = []
        nu_list = []

        for i in range(self.nodes + 1):

            state_list.append(ca.DM(scale_state) * 
                              opti.variable(self.state_dim))
            lam_list.append(opti.variable(self.num_waypoints))

            if i < self.nodes:
                control_list.append(ca.DM(scale_control) *          
                            opti.variable(self.control_dim))
                mu_list.append(opti.variable(self.num_waypoints))
                nu_list.append(opti.variable(self.num_waypoints))
                

        self.state = ca.hcat(state_list)
        self.control = ca.hcat(control_list)
        self.lam = ca.hcat(lam_list)
        self.mu = ca.hcat(mu_list)
        self.nu = ca.hcat(nu_list)

class MHE(ControlProblem):
    def __init__(self, reference_trajectory):
        super().__init__()
        self.reference_trajectory = reference_trajectory

    def progress(self, node):
        """
        Progress along the reference trajectory for a given node. Since we won't be tracking exactly we use a tube constraint with distance along the tube 
        """

    def _setup_objective(self, nodes):
        opti = self.opti

        return None

class MHE:
    def __init__(self, N, dt, state_dim, control_dim):
        # Horizon length and time step
        self.N = N
        self.dt = dt
        
        # State, control, and progress dimensions
        self.state_dim = state_dim
        self.control_dim = control_dim
        
        # Progress variable (scalar)
        self.s = ca.SX.sym('s', N + 1)
        
        # State and control variables
        self.x = ca.SX.sym('x', state_dim, N + 1)
        self.u = ca.SX.sym('u', control_dim, N)
        
        # Reference waypoint positions
        self.wp = ca.SX.sym('wp', state_dim, N + 1)

        # Weights for cost function
        self.R = ca.SX.sym('R', control_dim, control_dim)

        # Initial state and progress
        self.x0 = ca.SX.sym('x0', state_dim)
        self.s0 = ca.SX.sym('s0')

        # Objective and constraints
        self.cost = 0
        self.g = []

        # Model and cost definition
        self.define_model()
        self.define_cost_function()
        self.setup_solver()

    def define_model(self):
        """Define the system dynamics and progress update."""
        def dynamics(x, u):
            # Example dynamics: x_dot = u (replace with actual dynamics)
            return x + self.dt * u

        for k in range(self.N):
            # Dynamics constraint
            x_next = dynamics(self.x[:, k], self.u[:, k])
            self.g.append(self.x[:, k + 1] - x_next)

            # Progress update (distance traveled along trajectory)
            delta_s = ca.norm_2(self.x[:, k + 1] - self.x[:, k])
            self.g.append(self.s[k + 1] - (self.s[k] + delta_s))

    def define_cost_function(self):
        """Define the objective function."""
        # Maximize cumulative progress at the final step (equivalent to minimizing -s[N])
        self.cost = -self.s[-1]

        for k in range(self.N):
            # Control effort penalty to keep inputs reasonable
            control_dev = self.u[:, k]
            self.cost += ca.mtimes([control_dev.T, self.R, control_dev])

    def setup_solver(self):
        """Configure and create the solver."""
        # Concatenate variables
        vars = ca.vertcat(ca.reshape(self.x, -1, 1), ca.reshape(self.u, -1, 1), self.s)
        g = ca.vertcat(*self.g)
        
        # Optimization problem
        nlp = {'x': vars, 'f': self.cost, 'g': g}
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    def solve(self, x0, s0, wp, R):
        """Solve the MHE problem given current state, progress, and waypoints."""
        # Initial guess for states and controls
        x_init = np.tile(x0, (self.N + 1, 1)).T
        u_init = np.zeros((self.control_dim, self.N))
        s_init = np.linspace(s0, s0 + 1, self.N + 1)  # Linear progress initialization
        var_init = np.concatenate((x_init.flatten(), u_init.flatten(), s_init))

        # Constraints (dynamics consistency and progress update)
        lbx, ubx = -np.inf * np.ones_like(var_init), np.inf * np.ones_like(var_init)
        lbg, ubg = np.zeros(len(self.g)), np.zeros(len(self.g))

        # Set parameter values
        p = np.concatenate([wp.flatten(), R.flatten(), x0.flatten(), [s0]])

        # Solve the problem
        sol = self.solver(x0=var_init, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        x_opt = sol['x'].full()

        # Extract optimized states, controls, and progress
        n_x = self.state_dim * (self.N + 1)
        n_u = self.control_dim * self.N

        x_traj = x_opt[:n_x].reshape((self.state_dim, self.N + 1))
        u_traj = x_opt[n_x:n_x + n_u].reshape((self.control_dim, self.N))
        s_traj = x_opt[n_x + n_u:]

        return x_traj, u_traj, s_traj