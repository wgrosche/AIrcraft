import numpy as np
from dataclasses import dataclass
import casadi as ca

scale_state = ca.DM(ca.vertcat(
    [1, 1, 1, 1],
    [1e3, 1e3, 1e3],
    [1e2, 1e2, 1e2],
    [1, 1, 1]
    ))
scale_control = ca.DM(ca.vertcat(
    5, 5, 5,
    [1e2, 1e2, 1e2],
    [1, 1, 1],
    [1e2, 1e2, 1e2]
    ))
scale_time = 1

opts = {'scale_state': scale_state, 
        'scale_control': scale_control, 
        'scale_time': scale_time, 
        'aircraft':aircraft,
        'traj_dict': traj_dict}

class ControlNode:
    def __init__(self, opti, opts = {}):
        self.state = opts['scale_state'] * opti.variable(scale_state.shape[0])
        self.control = opts['scale_control'] * opti.variable(scale_control.shape[0])
        self.reached = opti.variable()
        self.aircraft = opts['aircraft']
        self.opti = opti
        self.traj_dict = opts['traj_dict']

        self.constrain_control()
        self.constrain_state()


    def constrain_state(self):
        opti = self.opti
        aircraft = self.aircraft
        dynamics = aircraft.state_update
        state_envelope = self.traj_dict.state_envelope

        alpha = aircraft.alpha
        beta = aircraft.beta
        airspeed = aircraft.airspeed

        opti.subject_to(opti.bounded(state_envelope.alpha.lb,
        alpha(self.state, self.control), state_envelope.alpha.ub))

        opti.subject_to(opti.bounded(state_envelope.beta.lb,
            beta(self.state, self.control), state_envelope.beta.ub))

        opti.subject_to(opti.bounded(state_envelope.airspeed.lb,
            airspeed(self.state, self.control), state_envelope.airspeed.ub))
        
    def constrain_control(self):
        control_envelope = self.trajectory.control
        opti = self.opti
        com = self.trajectory.aircraft.aero_centre_offset

        opti.subject_to(opti.bounded(control_envelope.lb[:6],
                self.control[:6], control_envelope.ub[:6]))
        
        opti.subject_to(opti.bounded(np.zeros(self.control[9:].shape),
                self.control[9:], np.zeros(self.control[9:].shape)))
        
        opti.subject_to(self.control[6:9]==com)

        

    
    opti.subject_to(node2.state_next == dynamics(node1.state, node2.control, dt))



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

        node_list = [Node(index = 0, 
                          state=scale_state * opti.variable(scale_state.shape[0]), 
                          reached=opti.variable())]

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