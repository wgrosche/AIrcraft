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

@dataclass
class Node:
    index:int = None
    state:ca.MX = None
    state_next:ca.MX = None
    control:ca.MX = None
    reached:ca.MX = None

    def create(opti, index, state_dim, control_dim):
        return Node(
            index=index,
            state = scale_state * opti.variable(scale_state.shape[0]),
            control = scale_control * opti.variable(scale_control.shape[0]),
            reached = opti.variable()
        )
    
@dataclass
class Envelope:
    alpha:ca.MX = None
    beta:ca.MX = None
    airspeed:ca.MX = None


def state_constraint(opti, node1:Node, node2:Node, dt:ca.MX, aircraft:Aircraft, 
                     state_envelope:Envelope):
    
    dynamics = aircraft.state_update

    alpha = aircraft.alpha
    beta = aircraft.beta
    airspeed = aircraft.airspeed

    opti.subject_to(opti.bounded(state_envelope.alpha.lb,
        alpha(node2.state, node2.control), state_envelope.alpha.ub))

    opti.subject_to(opti.bounded(state_envelope.beta.lb,
        beta(node2.state, node2.control), state_envelope.beta.ub))

    opti.subject_to(opti.bounded(state_envelope.airspeed.lb,
        airspeed(node2.state, node2.control), state_envelope.airspeed.ub))
    
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