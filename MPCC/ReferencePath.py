import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import trajectory_planning_helpers as tph

SPEED = 2 # m/s

def update_fo_state(x, u, dt):
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
        x_next = update_fo_state(states[-1], controls[i], 0.1)
        states.append(x_next)
        
    return np.array(states)[:, 0:2]

class ReferencePath:
    def __init__(self, width=0.8):
        self.width = width
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

        left_path = self.path + self.nvecs * width
        right_path = self.path - self.nvecs * width
        self.left_lut_x, self.left_lut_y = self.get_interpolated_path_casadi('lut_left_x', 'lut_left_y', left_path, self.s_track)
        self.right_lut_x, self.right_lut_y = self.get_interpolated_path_casadi('lut_right_x', 'lut_right_y', right_path, self.s_track)

    def init_path(self):
        self.path = create_test_path()
        track = np.concatenate((self.path, np.ones_like(self.path)*self.width), axis=1)
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
    
    def calculate_s(self, point):
        distances = np.linalg.norm(self.path - point, axis=1)
        idx = np.argmin(distances)
        x, h = self.interp_pts(idx, distances)
        s = (self.s_track[idx] + x) 

        return s

    def interp_pts(self, idx, dists):
        d_ss = self.s_track[idx+1] - self.s_track[idx]
        d1, d2 = dists[idx], dists[idx+1]

        if d1 < 0.01: # at the first point
            x = 0   
            h = 0
        elif d2 < 0.01: # at the second point
            x = dists[idx] # the distance to the previous point
            h = 0 # there is no distance
        else:     # if the point is somewhere along the line
            s = (d_ss + d1 + d2)/2
            Area_square = (s*(s-d1)*(s-d2)*(s-d_ss))
            if Area_square < 0:  # negative due to floating point precision
                h = 0
                x = d_ss + d1
            else:
                Area = Area_square**0.5
                h = Area * 2/d_ss
                x = (d1**2 - h**2)**0.5

        return x, h


    def plot_path(self):
        plt.figure(2)
        plt.clf()

        plt.plot(self.center_lut_x(self.s_track), self.center_lut_y(self.s_track), label="center")
        plt.plot(self.left_lut_x(self.s_track), self.left_lut_y(self.s_track), label="left")
        plt.plot(self.right_lut_x(self.s_track), self.right_lut_y(self.s_track), label="right")

        # plt.show()

if __name__ == "__main__":
    path = ReferencePath(0.8)
    path.plot_path()
    plt.show()
