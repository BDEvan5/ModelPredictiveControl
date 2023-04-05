import numpy as np 
import matplotlib.pyplot as plt
from Trajectory import Trajectory

import casadi as ca

L = 0.33
VERBOSE = False
VERBOSE = True


def update_state(x, u, dt):
    """updates the state for simple dynamics

    Args:
        x (ndarray(4)): pos_x, pos_y, theta, speed
        u (ndarray(2)): delta, acceleration
        dt (floar): timestep

    Returns:
        ndarray(3): new_state
    """
    dx = np.array([np.cos(x[2])*x[3], 
                   np.sin(x[2])*x[3], 
                   x[3]/L * np.tan(u[0]),
                    u[1]])
    x_next = x + dx * dt
    return x_next


class PlannerMPC:
    def __init__(self, trajectory, dt, N):
        self.trajectory = trajectory
        self.dt = dt
        self.N = N # number of steps to predict
        
        self.nx = 4
        self.nu = 2
        self.u_min = [-0.4,-8]
        self.u_max = [0.4, 8]
        
    def estimate_u0(self, reference_path, x0):
        reference_theta = np.arctan2(reference_path[1:, 1] - reference_path[:-1, 1], reference_path[1:, 0] - reference_path[:-1, 0])
        th_dot = np.diff(reference_theta) 
        th_dot[0] += (reference_theta[0]- x0[2]) 
        
        speeds = reference_path[:, 2]
        steering_angles = (np.arctan(th_dot) * L / speeds[:-2]) / self.dt
        speeds[0] += (x0[3] - reference_path[0, 2] )
        accelerations = np.diff(speeds) / self.dt
        
        u0_estimated = np.vstack((steering_angles, accelerations[:-1])).T
        
        return u0_estimated
        
    def plan(self, x0):
        reference_path = self.trajectory.get_timed_trajectory_segment(x0, self.dt, self.N+2)
        u0_estimated = self.estimate_u0(reference_path, x0)
        
        u_bar = self.generate_optimal_path(x0, reference_path[:-1].T, u0_estimated)
        
        return u_bar[0] # return the first control action
        
    def generate_optimal_path(self, x0_in, x_ref, u_init):
        """generates a set of optimal control inputs (and resulting states) for an initial position, reference trajectory and estimated control

        Args:
            x0_in (ndarray(3)): the initial pose
            x_ref (ndarray(N+1, 2)): the reference trajectory
            u_init (ndarray(N)): the estimated control inputs

        Returns:
            u_bar (ndarray(N)): optimal control plan
        """
        x = ca.SX.sym('x', self.nx, self.N+1)
        u = ca.SX.sym('u', self.nu, self.N)
        
        speeds = x_ref[2]
        # Add a speed objective cost.
        J = ca.sumsqr(x[:2, :] - x_ref[:2, :])  + ca.sumsqr(x[3, :] - speeds[None, :]) *100
        
        g = []
        for k in range(self.N):
            x_next = x[:,k] + f(x[:,k], u[:,k])*self.dt
            g.append(x_next - x[:,k+1])

        initial_constraint = x[:,0] - x0_in 
        g.append(initial_constraint)

        x_init = [x0_in]
        for i in range(1, self.N+1):
            x_init.append(x_init[i-1] + f(x_init[i-1], u_init[i-1])*self.dt)
        for i in range(len(u_init)):
            x_init.append(u_init[i])
        x_init = ca.vertcat(*x_init)
        
        lbx = [-ca.inf] * self.nx * (self.N+1) + self.u_min * self.N
        ubx = [ca.inf, ca.inf, ca.inf, 8] * (self.N+1) + self.u_max * self.N
        
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
        
        u_return = np.array(u_bar)[:, 0]
        u_return = u_return.reshape((self.N, self.nu))
        
        return u_return
        
        
def f(x, u):
    # define the dynamics as a casadi array
    xdot = ca.vertcat(
        ca.cos(x[2])*x[3],
        ca.sin(x[2])*x[3],
        x[3]/L * ca.tan(u[0]),
        u[1]
    )
    return xdot

        
    
def run_simulation():
    map_name = "esp"
    timestep = 0.2
    t = Trajectory(map_name)
    planner = PlannerMPC(t, timestep, 10)
    plt.figure(2)
    plt.plot(t.wpts[:, 0], t.wpts[:, 1], label="path")
    
    x0 = np.array([0, 0, 0, 0])
    x = x0
    states = []
    for i in range(200):
        u = planner.plan(x)
        x = update_state(x, u, timestep)
    
        states.append(x)
        
        plt.figure(2)
        plt.plot(x[0], x[1], "ro")
        plt.pause(0.0001)
        
    plt.title("Vehicle Path")
    plt.savefig(f"Imgs/FullMPC_vehicle_path_{map_name}.svg")
    plt.show()
    
    
if __name__ == "__main__":
    run_simulation()