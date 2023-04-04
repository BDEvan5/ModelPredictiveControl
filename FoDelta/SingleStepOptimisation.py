import numpy as np 
import matplotlib.pyplot as plt

import casadi as ca

speed = 2 # m/s
L = 0.33 # vehicle wheelbase
VERBOSE = False
VERBOSE = True

def create_test_path():
    n = 6
    steering = 0.38
    ones, zeros = [steering] * n, [-steering] * n
    controls = []
    for i in range(10):
        controls.append(zeros)
        controls.append(ones)
    controls = np.array(controls).reshape(-1)
    
    states = [[0, 0, np.pi/4]]
    for i in range(len(controls)):
        x_next = update_state(states[-1], controls[i], 0.1)
        states.append(x_next)
        
    return np.array(states)[:, 0:2]


def update_state(x, u, dt):
    """updates the state for simple dynamics

    Args:
        x (ndarray(3)): pos_x, pos_y, theta
        u (float): delta
        dt (floar): timestep

    Returns:
        ndarray(3): new_state
    """
    dx = np.array([np.cos(x[2])*speed, np.sin(x[2])*speed, speed/L * np.tan(u)])
    x_next = x + dx * dt
    return x_next


class SingleStepMPC:
    def __init__(self, path, dt, N):
        self.path = path
        self.dt = dt
        self.N = N # number of steps to predict
        
        self.nx = 3
        self.nu = 1
        self.u_lim = 0.4
        
    def generate_reference_path(self, x0):
        nearest_idx = np.argmin(np.linalg.norm(self.path - x0[:2], axis=1))
        
        reference_path = self.path[nearest_idx:nearest_idx+self.N+2]
        
        reference_theta = np.arctan2(reference_path[1:, 1] - reference_path[:-1, 1], reference_path[1:, 0] - reference_path[:-1, 0])
        th_dot = np.diff(reference_theta) 
        th_dot[0] += (reference_theta[0]- x0[2]) 
        
        u0_estimated = (np.arctan(th_dot) * L / speed) / self.dt
        
        return reference_path, u0_estimated
        
    def plan(self, x0):
        reference_path, u0_estimated = self.generate_reference_path(x0)
        
        self.generate_optimal_path(x0, reference_path[:-1].T, u0_estimated)
        
    def generate_optimal_path(self, x0_in, x_ref, u_init):
        plt.plot(x_ref[0, :], x_ref[1, :], label="x_ref")
        
        #Define the casadi variables for the state and control
        x = ca.SX.sym('x', self.nx, self.N+1)
        u = ca.SX.sym('u', self.nu, self.N)
        
        #Define the cost function to be minimised: minimum distance between each way point
        J = ca.sumsqr(x[:2, :-1] - x_ref[:, :-1])
        
        # Define constraints, first for the dynamics then the initial state
        g = []
        for k in range(self.N):
            x_next = x[:,k] + f(x[:,k], u[:,k])*self.dt
            g.append(x_next - x[:,k+1])

        initial_constraint = x[:,0] - x0_in 
        g.append(initial_constraint)

        # generate an initial guess
        x_init = [x0_in]
        for i in range(1, 1+self.N):
            x_init.append(x_init[i-1] + f(x_init[i-1], [u_init[i-1]])*self.dt)
        x_init.append(u_init)
        x_init = ca.vertcat(*x_init)
        
        # Setup lower (lbx) and upper (ubx) bounds on the state and control variables
        lbx = [-ca.inf] * self.nx * (self.N+1) + [-self.u_lim] * self.N
        ubx = [ca.inf] * self.nx * (self.N+1) + [self.u_lim] * self.N
        
        # Create the NLP problem
        x_nlp = ca.vertcat(x.reshape((-1, 1)), u.reshape((-1, 1)))
        g_nlp = ca.vertcat(*g)
        nlp = {'x': x_nlp,
            'f': J,
            'g': g_nlp}

        # Create the optimizer and solve the problem
        opts = {'ipopt': {'print_level': 2},
                'print_time': False}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        sol = solver(x0=x_init, lbx=lbx, ubx=ubx, lbg=0, ubg=0)

        # plot the results        
        plt.figure(1)
        init_states = np.array(x_init[:self.nx*(self.N+1)].reshape((self.nx, self.N+1)))
        plt.plot(init_states[0, :], init_states[1, :], label="Estimated States (x0)")
        
        x_bar = np.array(sol['x'][:self.nx*(self.N+1)].reshape((self.nx, self.N+1)))
        plt.plot(x_bar[0, :], x_bar[1, :], label="Solution States (x_bar)") 
        
        plt.title(f"States and Reference Path ")
        plt.legend()
        # plt.xlim(-0.5, 4)
        plt.savefig("Imgs/FODelta_single_step.svg")
        
        if VERBOSE: 
            plt.figure(2)
            plt.title(f"Control Actions")
            u_bar = sol['x'][self.nx*(self.N+1):]
            plt.plot(u_init, label="Initial")
            plt.plot(u_bar, label="Control")
            plt.legend()
            
            plt.figure(3)
            plt.title(f"Theta")
            plt.plot(init_states[2, :], label="Initial States")
            plt.plot(x_bar[2, :], label="Solution States")
            plt.legend()
        
        plt.show()
        
def f(x, u):
    # define the dynamics as a casadi array
    xdot = ca.vertcat(
        ca.cos(x[2])*speed,
        ca.sin(x[2])*speed,
        speed/L * ca.tan(u[0])
    )
    return xdot

def optimisate_single_step():
    path = create_test_path()
    planner = SingleStepMPC(path, 0.1, 100)
    plt.figure()
    plt.plot(path[:, 0], path[:, 1], label="Path to follow")
    
    x0 = np.array([0, 0, 0])
    planner.plan(x0)
        
    
if __name__ == "__main__":
    optimisate_single_step()
    