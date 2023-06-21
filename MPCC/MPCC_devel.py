import numpy as np 
import matplotlib.pyplot as plt

import casadi as ca
import trajectory_planning_helpers as tph
from ReferencePath import ReferencePath


SPEED = 2 # m/s
VERBOSE = False
L = 0.33
WIDTH = 0.5 # m on each side

WEIGHT_PROGRESS = 10
WEIGHT_LAG = 100
WEIGHT_CONTOUR = 1


def update_state(x, u, dt):
    """updates the state for simple dynamics

    Args:
        x (ndarray(3)): pos_x, pos_y, theta
        u (float): delta
        dt (floar): timestep

    Returns:
        ndarray(3): new_state
    """
    dx = np.array([np.cos(x[2])*SPEED, np.sin(x[2])*SPEED, SPEED/L * np.tan(u)])
    x_next = x + dx * dt
    return x_next


class PlannerMPC:
    def __init__(self, dt, N):
        self.rp = ReferencePath(WIDTH)
        self.dt = dt
        self.N = N # number of steps to predict
        self.nx = 4
        self.nu = 2
        
        self.x_min, self.y_min = np.min(self.rp.path, axis=0)
        self.psi_min, self.s_min = -np.pi, 0
        self.x_max, self.y_max = np.max(self.rp.path, axis=0)
        self.psi_max, self.s_max = np.pi, self.rp.s_track[-1]

        self.theta_min, self.p_min = -0.4, 0
        self.theta_max, self.p_max = 0.4, 4

        self.u0 = np.zeros((self.N, self.nu))
        self.X0 = np.zeros((self.N + 1, self.nx))
        self.warm_start = False

        self.init_optimisation()
        self.init_constraints()
        self.init_objective()
        self.init_solver()
       
    def init_optimisation(self):
        # States
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        psi = ca.MX.sym('psi')
        s = ca.MX.sym('s')
        # Controls
        v = ca.MX.sym('v')
        theta = ca.MX.sym('theta')
        p = ca.MX.sym('p')

        states = ca.vertcat(x, y, psi, s)
        controls = ca.vertcat(theta, p)
        rhs = ca.vertcat(SPEED * ca.cos(psi), SPEED * ca.sin(psi), (SPEED / L) * ca.tan(theta), p)  # dynamic equations of the states
        self.f = ca.Function('f', [states, controls], [rhs])  # nonlinear mapping function f(x,u)
        self.U = ca.MX.sym('U', self.nu, self.N)
        self.X = ca.MX.sym('X', self.nx, (self.N + 1))
        self.P = ca.MX.sym('P', self.nx + 2 * self.N) # init state and boundaries of the reference path

        self.Q = ca.MX.zeros(2, 2)
        self.Q[0, 0] = WEIGHT_CONTOUR  # cross track error
        self.Q[1, 1] = WEIGHT_LAG  # lag error

    def init_constraints(self):
        '''Initialize constraints for states, dynamic model state transitions and control inputs of the system'''
        self.lbg = np.zeros((self.nx * (self.N + 1) + self.N, 1))
        self.ubg = np.zeros((self.nx * (self.N + 1) + self.N, 1))
        self.lbx = np.zeros((self.nx + (self.nx + self.nu) * self.N, 1))
        self.ubx = np.zeros((self.nx + (self.nx + self.nu) * self.N, 1))
        # Upper and lower bounds for the state optimization variables
        for k in range(self.N + 1):
            self.lbx[self.nx * k:self.nx * (k + 1), 0] = np.array(
                [[self.x_min, self.y_min, self.psi_min, self.s_min]])
            self.ubx[self.nx * k:self.nx * (k + 1), 0] = np.array(
                [[self.x_max, self.y_max, self.psi_max, self.s_max]])
        state_count = self.nx * (self.N + 1)
        # Upper and lower bounds for the control optimization variables
        for k in range(self.N):
            self.lbx[state_count:state_count + self.nu, 0] = np.array(
                [[self.theta_min, self.p_min]])  # v and theta lower bound
            self.ubx[state_count:state_count + self.nu, 0] = np.array(
                [[self.theta_max, self.p_max]])  # v and theta upper bound
            state_count += self.nu

    def init_objective(self):
        self.obj = 0  # Objective function
        self.g = []  # constraints vector

        st = self.X[:, 0]  # initial state
        self.g = ca.vertcat(self.g, st - self.P[:self.nx])  # initial condition constraints
        for k in range(self.N):
            st = self.X[:, k]
            st_next = self.X[:, k + 1]
            con = self.U[:, k]
            t_angle = self.rp.angle_lut_t(st_next[3])
            ref_x, ref_y = self.rp.center_lut_x(st_next[3]), self.rp.center_lut_y(st_next[3])
            #Contouring error
            e_c = ca.sin(t_angle) * (st_next[0] - ref_x) - ca.cos(t_angle) * (st_next[1] - ref_y)
            #Lag error
            e_l = -ca.cos(t_angle) * (st_next[0] - ref_x) - ca.sin(t_angle) * (st_next[1] - ref_y)

            self.obj = self.obj + e_c **2 * WEIGHT_CONTOUR  
            self.obj = self.obj + e_l **2 * WEIGHT_LAG
            self.obj = self.obj - con[1] * WEIGHT_PROGRESS #! add a cost for the speed

            k1 = self.f(st, con)
            st_next_euler = st + (self.dt * k1)
            self.g = ca.vertcat(self.g, st_next - st_next_euler)  # compute constraints

            # path boundary constraints
            self.g = ca.vertcat(self.g, self.P[self.nx + 2 * k] * st_next[0] - self.P[self.nx + 2 * k + 1] * st_next[1])  # LB<=ax-by<=UB  --represents half space planes

    def init_solver(self):
        opts = {}
        opts["ipopt"] = {}
        opts["ipopt"]["max_iter"] = 2000
        opts["ipopt"]["print_level"] = 0
        opts["print_time"] = 0
        
        OPT_variables = ca.vertcat(ca.reshape(self.X, self.nx * (self.N + 1), 1),
                                ca.reshape(self.U, self.nu * self.N, 1))

        nlp_prob = {'f': self.obj, 'x': OPT_variables, 'g': self.g, 'p': self.P}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def plan(self, x0):
        s_current = self.rp.calculate_s(x0[0:2])
        x0 = np.append(x0, s_current)

        p, pts_l, pts_r = self.generate_constraints_and_parameters(x0)
        states, controls = self.solve(p)

        c_pts = [[self.rp.center_lut_x(states[k, 3]), self.rp.center_lut_y(states[k, 3])] for k in range(self.N + 1)]

        plt.figure(2)
        plt.plot(states[:, 0], states[:, 1], 'r--')
        for i in range(self.N + 1):
            xs = [states[i, 0], c_pts[i][0]]
            ys = [states[i, 1], c_pts[i][1]]
            plt.plot(xs, ys, '--', color='orange')

        first_control = controls[0, :]
        
        return first_control[0] # return the first control action

    def generate_constraints_and_parameters(self, x0_in):
        if not self.warm_start:
            self.construct_warm_start_soln(x0_in)

        p = np.zeros(self.nx + 2 * self.N)
        p[:self.nx] = x0_in

        points_left = np.zeros((self.N, 2))
        points_right = np.zeros((self.N, 2))
        for k in range(self.N):  # set the reference controls and path boundary conditions to track
            s = self.X0[k, 3]
            right_point = [self.rp.right_lut_x(s).full()[0, 0], self.rp.right_lut_y(s).full()[0, 0]]
            left_point = [self.rp.left_lut_x(s).full()[0, 0], self.rp.left_lut_y(s).full()[0, 0]]
            points_left[k, :] = left_point
            points_right[k, :] = right_point

            delta_x_path = right_point[0] - left_point[0]
            delta_y_path = right_point[1] - left_point[1]
            p[self.nx + 2 * k:self.nx + 2 * k + 2] = [-delta_x_path, delta_y_path]

            up_bound = max(-delta_x_path * right_point[0] - delta_y_path * right_point[1],
                           -delta_x_path * left_point[0] - delta_y_path * left_point[1])
            low_bound = min(-delta_x_path * right_point[0] - delta_y_path * right_point[1],
                            -delta_x_path * left_point[0] - delta_y_path * left_point[1])
            self.lbg[self.nx - 1 + (self.nx + 1) * (k + 1), 0] = low_bound # check this, there could be an error
            self.ubg[self.nx - 1 + (self.nx + 1) * (k + 1), 0] = up_bound

        return p, points_left, points_right

    def solve(self, p):
        x_init = ca.vertcat(ca.reshape(self.X0.T, self.nx * (self.N + 1), 1),
                         ca.reshape(self.u0.T, self.nu * self.N, 1))

        sol = self.solver(x0=x_init, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)

        # Get state and control solution
        self.X0 = ca.reshape(sol['x'][0:self.nx * (self.N + 1)], self.nx, self.N + 1).T  # get soln trajectory
        u = ca.reshape(sol['x'][self.nx * (self.N + 1):], self.nu, self.N).T  # get controls solution

        trajectory = self.X0.full()  # size is (N+1,n_states)
        inputs = u.full()

        # Shift trajectory and control solution to initialize the next step
        self.X0 = ca.vertcat(self.X0[1:, :], self.X0[self.X0.size1() - 1, :])
        self.u0 = ca.vertcat(u[1:, :], u[u.size1() - 1, :])

        return trajectory, inputs
        
    def construct_warm_start_soln(self, initial_state):
        # Construct an initial estimated solution to warm start the optimization problem with valid path constraints
        #! this will break for multiple laps
        # if initial_state[3] >= self.arc_lengths_orig_l:
        #     initial_state[3] -= self.arc_lengths_orig_l
        p_initial = 2
        initial_state[2] = self.rp.angle_lut_t(initial_state[3])
        self.X0[0, :] = initial_state
        for k in range(1, self.N + 1):
            s_next = self.X0[k - 1, 3] + p_initial * self.dt
            psi_next = self.rp.angle_lut_t(s_next)
            x_next, y_next = self.rp.center_lut_x(s_next), self.rp.center_lut_y(s_next)

            self.X0[k, :] = np.array([x_next.full()[0, 0], y_next.full()[0, 0], psi_next.full()[0, 0], s_next])

        self.warm_start = True



        
    
def run_simulation():
    planner = PlannerMPC(0.2, 10)
    planner.rp.plot_path()

    x0 = np.array([0, 0, 0])
    x = x0
    states = []
    for i in range(70):
        planner.rp.plot_path()
        u = planner.plan(x)
        x = update_state(x, u, 0.2)
    
        states.append(x)
        
        plt.figure(2)
        plt.plot(x[0], x[1], "ro")
        plt.pause(0.0001)
        plt.pause(0.5)

        # plt.show()
        
    plt.title("Vehicle Path")
    plt.savefig("Imgs/FO_mpcc_vehicle_path.svg")
    plt.show()
    

def test_path():
    p = ReferencePath()
    p.plot_path()
    
if __name__ == "__main__":
    run_simulation()


    # test_path()
