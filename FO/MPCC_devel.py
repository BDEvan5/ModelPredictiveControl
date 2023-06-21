import numpy as np 
import matplotlib.pyplot as plt

import casadi as ca
import trajectory_planning_helpers as tph

SPEED = 2 # m/s
VERBOSE = False
L = 0.33
WIDTH = 0.8 # m on each side


def create_test_path():
    n = 20
    ones, zeros = [1] * n, [-1] * n
    controls = []
    for i in range(3):
        controls.append(zeros)
        controls.append(ones)
    controls = np.array(controls).reshape(-1)
    
    states = [[0, 0, np.pi/4]]
    for i in range(len(controls)):
        x_next = update_state(states[-1], controls[i], 0.1)
        states.append(x_next)
        
    return np.array(states)[:, 0:2]

class ReferencePath:
    def __init__(self):
        self.path = None
        self.el_lengths = None 
        self.psi = None
        self.nvecs = None
        self.init_path()

        self.center_lut_x, self.center_lut_y = None, None
        self.left_lut_x, self.left_lut_y = None, None
        self.right_lut_x, self.right_lut_y = None, None

        self.center_lut_x, self.center_lut_y = self.get_interpolated_path_casadi('lut_center_x', 'lut_center_y', self.path, self.s_track)
        self.angle_lut_t = self.get_interpolated_heading_casadi('lut_angle_t', self.psi, self.s_track)

        left_path = self.path + self.nvecs * WIDTH
        right_path = self.path - self.nvecs * WIDTH
        self.left_lut_x, self.left_lut_y = self.get_interpolated_path_casadi('lut_left_x', 'lut_left_y', left_path, self.s_track)
        self.right_lut_x, self.right_lut_y = self.get_interpolated_path_casadi('lut_right_x', 'lut_right_y', right_path, self.s_track)

    def init_path(self):
        self.path = create_test_path()
        track = np.concatenate((self.path, np.ones_like(self.path)*WIDTH), axis=1)
        self.el_lengths = np.linalg.norm(np.diff(track[:, :2], axis=0), axis=1)
        self.s_track = np.insert(np.cumsum(self.el_lengths), 0, 0)
        self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(track, self.el_lengths, False)
        self.nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(self.psi-np.pi/2)

    def get_interpolated_path_casadi(self, label_x, label_y, pts, arc_lengths_arr):
        u = arc_lengths_arr
        V_X = pts[:, 0]
        V_Y = pts[:, 1]
        lut_x = ca.interpolant(label_x, 'bspline', [u], V_X)
        lut_y = ca.interpolant(label_y, 'bspline', [u], V_Y)
        return lut_x, lut_y
    
    def get_interpolated_heading_casadi(self, label, pts, arc_lengths_arr):
        u = arc_lengths_arr
        V = pts
        lut = ca.interpolant(label, 'bspline', [u], V)
        return lut

    def plot_path(self):
        plt.figure(2)
        # plt.plot(self.path[:, 0], self.path[:, 1], label="path")

        plt.plot(self.center_lut_x(self.s_track), self.center_lut_y(self.s_track), label="center")
        plt.plot(self.left_lut_x(self.s_track), self.left_lut_y(self.s_track), label="left")
        plt.plot(self.right_lut_x(self.s_track), self.right_lut_y(self.s_track), label="right")

        plt.show()

def update_state(x, u, dt):
    """updates the state for simple dynamics

    Args:
        x (ndarray(3)): pos_x, pos_y, theta
        u (float): theta_dot
        dt (floar): timestep

    Returns:
        ndarray(3): new_state
    """
    dx = np.array([np.cos(x[2])*SPEED, np.sin(x[2])*SPEED, u])
    x_next = x + dx * dt
    return x_next


class PlannerMPC:
    def __init__(self, path, dt, N):
        self.rp = ReferencePath()
        self.dt = dt
        self.N = N # number of steps to predict
        
        self.nx = 4
        self.nu = 2
        self.T_V = self.nx + self.nu
        
        self.x_min, self.y_min = np.min(self.rp.path, axis=0)
        self.psi_min, self.s_min = -np.pi, 0
        self.x_max, self.y_max = np.max(self.rp.path, axis=0)
        self.psi_max, self.s_max = np.pi, self.rp.s_track[-1]

        self.theta_min, self.p_min = -0.4, 0
        self.theta_max, self.p_max = 0.4, 4

        self.u0 = np.zeros((self.N, self.nu))
        self.X0 = np.zeros((self.N + 1, self.nx))

    def generate_reference_path(self, x0):
        nearest_idx = np.argmin(np.linalg.norm(self.path - x0[:2], axis=1))
        
        reference_path = self.path[nearest_idx:nearest_idx+self.N+2]
        
        reference_theta = np.arctan2(reference_path[1:, 1] - reference_path[:-1, 1], reference_path[1:, 0] - reference_path[:-1, 0])
        u0_estimated = np.diff(reference_theta) / self.dt
        u0_estimated[0] += (reference_theta[0]- x0[2]) / self.dt
        
        return reference_path, u0_estimated
        
    def plan(self, x0):
        reference_path, u0_estimated = self.generate_reference_path(x0)
        
        u_bar = self.generate_optimal_path(x0, reference_path[:-1].T, u0_estimated)
        
        return u_bar[0] # return the first control action
        
    def generate_optimal_path(self, x0_in, u_init):
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
        self.Q[0, 0] = 50  # cross track error
        self.Q[1, 1] = 1000  # lag error

        self.obj = 0  # Objective function
        self.g = []  # constraints vector

        st = self.X[:, 0]  # initial state
        self.g = ca.vertcat(self.g, st - x0_in)  # initial condition constraints
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
            error = ca.vertcat(e_c, e_l)

            self.obj = self.obj + ca.mtimes(ca.mtimes(error.T, self.Q), error) #! add a cost for the centerline velocity

            k1 = self.f(st, con)
            st_next_euler = st + (self.dt * k1)
            self.g = ca.vertcat(self.g, st_next - st_next_euler)  # compute constraints

            # path boundary constraints
            self.g = ca.vertcat(self.g, self.P[self.nx + 2 * k] * st_next[0] - self.P[self.nx + 2 * k + 1] * st_next[1])  # LB<=ax-by<=UB  --represents half space planes


        # setup solver with bounds, and initial value
        p = np.zeros(self.nx + 2 * self.N)
        p[:self.nx] = x0_in

        right_points, left_points = self.get_path_constraints_points(self.X0)
        for k in range(self.N):  # set the reference controls and path boundary conditions to track
            delta_x_path = right_points[k, 0] - left_points[k, 0]
            delta_y_path = right_points[k, 1] - left_points[k, 1]
            p[self.nx + 2 * k:self.nx + 2 * k + 2] = [-delta_x_path, delta_y_path]
            up_bound = max(-delta_x_path * right_points[k, 0] - delta_y_path * right_points[k, 1],
                           -delta_x_path * left_points[k, 0] - delta_y_path * left_points[k, 1])
            low_bound = min(-delta_x_path * right_points[k, 0] - delta_y_path * right_points[k, 1],
                            -delta_x_path * left_points[k, 0] - delta_y_path * left_points[k, 1])
            
            self.lbg[self.nx - 1 + (self.nx + 1) * (k + 1), 0] = low_bound # check this, there could be an error
            self.ubg[self.nx - 1 + (self.nx + 1) * (k + 1), 0] = up_bound

        self.construct_warm_start_soln(x0_in)
        x_init = ca.vertcat(ca.reshape(self.X0.T, self.nx * (self.N + 1), 1),
                         ca.reshape(self.u0.T, self.nx * self.N, 1))
        OPT_variables = vertcat(reshape(self.X, self.n_states * (self.N + 1), 1),
                                reshape(self.U, self.n_controls * self.N, 1))
        
        opts = {}
        opts["ipopt"] = {}
        opts["ipopt"]["max_iter"] = 2000
        opts["ipopt"]["print_level"] = 0
        nlp_prob = {'f': self.obj, 'x': OPT_variables, 'g': self.g, 'p': self.P}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
        sol = self.solver(x0=x_init, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)


        # Get state and control solution
        self.X0 = ca.reshape(sol['x'][0:self.nx * (self.N + 1)], self.nu, self.N + 1).T  # get soln trajectory
        u = ca.reshape(sol['x'][self.nx * (self.N + 1):], self.nu, self.N).T  # get controls solution

        # Get the first control
        con_first = u[0, :].T
        trajectory = self.X0.full()  # size is (N+1,n_states)
        inputs = u.full()

        # Shift trajectory and control solution to initialize the next step
        self.X0 = ca.vertcat(self.X0[1:, :], self.X0[self.X0.size1() - 1, :])
        self.u0 = ca.vertcat(u[1:, :], u[u.size1() - 1, :])
        # return con_first, trajectory, inputs

        return con_first
        
        # return np.array(u_bar)[:, 0]
    
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
            x_next, y_next = self.get_point_at_centerline(s_next)
            self.X0[k, :] = [x_next, y_next, psi_next, s_next]

    def get_point_at_centerline(self, s):
        x, y = self.rp.center_lut_xy(s), self.rp.center_lut_y(s)
        return x, y

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

    def get_path_constraints_points(self, prev_soln):
        right_points = np.zeros((self.N, 2))
        left_points = np.zeros((self.N, 2))
        for k in range(1, self.N + 1):
            right_points[k - 1, :] = [self.rp.right_lut_x(prev_soln[k, 3]),
                                      self.rp.right_lut_y(prev_soln[k, 3])]  # Right boundary
            left_points[k - 1, :] = [self.rp.left_lut_x(prev_soln[k, 3]),
                                     self.rp.left_lut_y(prev_soln[k, 3])]  # Left boundary
        return right_points, left_points

def f(x, u):
    # define the dynamics as a casadi array
    xdot = ca.vertcat(
        ca.cos(x[2])*SPEED,
        ca.sin(x[2])*SPEED,
        u[0]
    )
    return xdot

        
    
def run_simulation():
    path = create_test_path()
    planner = PlannerMPC(path, 0.1, 10)
    plt.figure(2)
    plt.plot(path[:, 0], path[:, 1], label="path")
    
    x0 = np.array([0, 0, 0])
    x = x0
    states = []
    for i in range(100):
        u = planner.plan(x)
        x = update_state(x, u, 0.1)
    
        states.append(x)
        
        plt.figure(2)
        plt.plot(x[0], x[1], "ro")
        plt.pause(0.0001)
        
    plt.title("Vehicle Path")
    plt.savefig("Imgs/FO_vehicle_path.svg")
    plt.show()
    

def test_path():
    p = ReferencePath()
    p.plot_path()
    
if __name__ == "__main__":
    # run_simulation()


    test_path()
