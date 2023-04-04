import numpy as np 
import csv 
from numba import njit
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

VERBOSE = False
# VERBOSE = True

class Trajectory:
    def __init__(self, map_name):
        self.wpts = None
        self.ss = None
        self.map_name = map_name
        self.total_s = None
        self.vs = None
        
        self.load_raceline()
        
        self.total_s = self.ss[-1]
        self.N = len(self.wpts)
        
        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2

        self.max_distance = 0
        self.distance_allowance = 1
  
    def load_raceline(self):
        track = []
        filename = 'maps/' + self.map_name + "_raceline.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Raceline Loaded: {filename}")

        self.wpts = track[:, 1:3]
        self.vs = track[:, 5] 

        seg_lengths = np.linalg.norm(np.diff(self.wpts, axis=0), axis=1)
        self.ss = np.insert(np.cumsum(seg_lengths), 0, 0)
    
    def plot_wpts(self):
        plt.figure(1)
        
        points = self.wpts.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, 8)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(self.vs)
        lc.set_linewidth(2)
        line = plt.gca().add_collection(lc)
        plt.colorbar(line, fraction=0.046, pad=0.04, shrink=0.99)

        plt.axis('equal')
        # plt.show()
        
    def get_contsant_speed_timed_trajectory_segment(self, position, dt = 0.1, n_pts=10):
        speed = 2
        wpts = np.vstack((self.wpts[:, 0], self.wpts[:, 1])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory_py2(position, wpts, self.l2s, self.diffs)
        
        distance = dt * speed 
        
        interpolated_distances = np.arange(self.ss[i], self.ss[i] + distance*(n_pts+1), distance)
        
        interpolated_xs = np.interp(interpolated_distances, self.ss, self.wpts[:, 0])
        interpolated_ys = np.interp(interpolated_distances, self.ss, self.wpts[:, 1])
        interpolated_waypoints = np.vstack((interpolated_xs, interpolated_ys)).T
        

        if VERBOSE:
            self.plot_wpts()
            plt.plot(interpolated_waypoints[:, 0], interpolated_waypoints[:, 1], 'rx', markersize=10)
            
            print(interpolated_waypoints)
        
            plt.figure(2)
            plt.plot(self.ss, self.wpts[:, 0], label='waypoints')
            plt.plot(interpolated_distances, interpolated_xs, label="Interp")
            plt.xlabel("cumulative distance")
            
            plt.show()    
            
    def get_timed_trajectory_segment(self, position, dt = 0.1, n_pts=10):
        wpts = np.vstack((self.wpts[:, 0], self.wpts[:, 1])).T
        
        trajectory, distances = [], []
        pose = position
        for i in range(n_pts):
            nearest_point, nearest_dist, t, i = nearest_point_on_trajectory_py2(pose[0:2], wpts, self.l2s, self.diffs)
            distance = dt * self.vs[i]
            
            next_distance = self.ss[i] + distance
            distances.append(next_distance)
            
            interpolated_x = np.interp(next_distance, self.ss, self.wpts[:, 0])
            interpolated_y = np.interp(next_distance, self.ss, self.wpts[:, 1])
            interpolated_v = np.interp(next_distance, self.ss, self.vs)
            pose = np.array([interpolated_x, interpolated_y, interpolated_v])
            trajectory.append(pose)
        
        interpolated_waypoints = np.array(trajectory)

        if VERBOSE:
            self.plot_wpts()
            plt.plot(interpolated_waypoints[:, 0], interpolated_waypoints[:, 1], 'rx', markersize=10)
            
            print(interpolated_waypoints)
        
            plt.figure(2)
            plt.plot(self.ss, self.wpts[:, 0], label='waypoints')
            plt.plot(distances, interpolated_waypoints[:, 0], 'rx', label="Interp")
            print(distances)
            plt.xlabel("cumulative distance")
            
            plt.figure(3)
            plt.plot(self.ss, self.vs, label='waypoints')
            plt.plot(distances, interpolated_waypoints[:, 2], 'rx', label="Interp")
            print(distances)
            plt.xlabel("cumulative distance")
            
            plt.figure(4)
            distances = np.array(distances)
            plt.plot(np.diff(distances), 'rx')
            plt.title("distances")
            
            plt.show()
    
    def get_raceline_speed(self, point):
        idx, dists = self.get_trackline_segment(point)
        return self.vs[idx]
    
    def get_lookahead_point(self, position, lookahead_distance):
        wpts = np.vstack((self.wpts[:, 0], self.wpts[:, 1])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory_py2(position, wpts, self.l2s, self.diffs)

        lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
        if i2 == None: 
            return None
        lookahead_point = np.empty((3, ))
        lookahead_point[0:2] = wpts[i2, :]
        lookahead_point[2] = self.vs[i]
        
        return lookahead_point





@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory_py2(point, trajectory, l2s, diffs):
    '''
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    '''
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    ''' starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    '''
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

@njit(fastmath=True, cache=True)
def add_locations(x1, x2, dx=1):
    # dx is a scaling factor
    ret = np.array([0.0, 0.0])
    for i in range(2):
        ret[i] = x1[i] + x2[i] * dx
    return ret

@njit(fastmath=True, cache=True)
def sub_locations(x1=[0, 0], x2=[0, 0], dx=1):
    # dx is a scaling factor
    ret = np.array([0.0, 0.0])
    for i in range(2):
        ret[i] = x1[i] - x2[i] * dx
    return ret


if __name__ == "__main__":
    trajectory = Trajectory("esp")

    # trajectory.get_contsant_speed_timed_trajectory_segment(np.array([5, 0]), 0.1, 20)
    # trajectory.get_timed_trajectory_segment(np.array([10, 0]), 0.5, 20)
    trajectory.get_timed_trajectory_segment(np.array([20, -15]), 0.5, 20)

