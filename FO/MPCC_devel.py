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
        self.path = ReferencePath()
        self.dt = dt
        self.N = N # number of steps to predict
        
        self.nx = 4
        self.nu = 2
        self.T_V = self.nx + self.nu
        self.u_lim = 1.2
        
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
        
    def generate_optimal_path(self, x0_in, x_ref, u_init):
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

        self.Q = ca.MX.zeros(2, 2)
        self.Q[0, 0] = 50  # cross track error
        self.Q[1, 1] = 1000  # lag error

        self.obj = 0  # Objective function
        self.g = []  # constraints vector

        x = ca.SX.sym('x', self.nx, self.N+1)
        u = ca.SX.sym('u', self.nu, self.N)
        
        J = ca.sumsqr(x[:2, :-1] - x_ref[:, :-1])
        
        g = []
        for k in range(self.N):
            x_next = x[:,k] + f(x[:,k], u[:,k])*self.dt
            g.append(x_next - x[:,k+1])

        initial_constraint = x[:,0] - x0_in 
        g.append(initial_constraint)

        x_init = [x0_in]
        for i in range(1, 1+self.N):
            x_init.append(x_init[i-1] + f(x_init[i-1], [u_init[i-1]])*self.dt)
        x_init.append(u_init)
        x_init = ca.vertcat(*x_init)
        
        lbx = [-ca.inf] * self.nx * (self.N+1) + [-self.u_lim] * self.N
        ubx = [ca.inf] * self.nx * (self.N+1) + [self.u_lim] * self.N
        
        x_nlp = ca.vertcat(x.reshape((-1, 1)), u.reshape((-1, 1)))
        g_nlp = ca.vertcat(*g)
        nlp = {'x': x_nlp,
            'f': J,
            'g': g_nlp}

        opts = {'ipopt': {'print_level': 2},
                'print_time': False}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        sol = solver(x0=x_init, lbx=lbx, ubx=ubx, lbg=0, ubg=0)

        x_bar = np.array(sol['x'][:self.nx*(self.N+1)].reshape((self.nx, self.N+1)))
        u_bar = sol['x'][self.nx*(self.N+1):]
        
        if VERBOSE:
            plt.figure(1)
            plt.clf()
            plt.plot(x_ref[0, :], x_ref[1, :], label="x_ref")
            init_states = np.array(x_init[:self.nx*(self.N+1)].reshape((self.nx, self.N+1)))
            plt.plot(init_states[0, :], init_states[1, :], label="Estimated States (x0)")
            plt.plot(x_bar[0, :], x_bar[1, :], label="Solution States (x_bar)") 
            
            plt.title(f"States and Reference Path ")
            plt.legend()
            
            plt.pause(0.01)
        
        return np.array(u_bar)[:, 0]
        

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
