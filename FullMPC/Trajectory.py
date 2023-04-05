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
        
    def get_contsant_speed_timed_trajectory_segment(self, position, dt = 0.1, n_pts=10, speed = 2):
        position_distances = np.linalg.norm(self.wpts-position[0:2], axis=1)
        i = np.argmin(position_distances)
        
        distance = dt * speed 
        
        interpolated_distances = np.linspace(self.ss[i], self.ss[i]+distance*(n_pts), n_pts)
        
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
            
        return interpolated_waypoints

    def get_trackline_segment(self, point):
        """Returns the first index representing the line segment that is closest to the point.

        wpt1 = pts[idx]
        wpt2 = pts[idx+1]

        dists: the distance from the point to each of the wpts.
        """
        dists = np.linalg.norm(point - self.wpts, axis=1)

        min_dist_segment = np.argmin(dists)
        if min_dist_segment == 0:
            return 0, dists
        elif min_dist_segment == len(dists)-1:
            return len(dists)-2, dists 

        if dists[min_dist_segment+1] < dists[min_dist_segment-1]:
            return min_dist_segment, dists
        else: 
            return min_dist_segment - 1, dists
        
    
    def interp_pts(self, idx, dists):
        """
        Returns the distance along the trackline and the height above the trackline
        Finds the reflected distance along the line joining wpt1 and wpt2
        Uses Herons formula for the area of a triangle
        
        """
        d_ss = self.ss[idx+1] - self.ss[idx]
        d1, d2 = dists[idx], dists[idx+1]

        if d1 < 0.01: # at the first point
            x = 0   
            h = 0
        elif d2 < 0.01: # at the second point
            x = dists[idx] # the distance to the previous point
            h = 0 # there is no distance
        else: 
            # if the point is somewhere along the line
            s = (d_ss + d1 + d2)/2
            Area_square = (s*(s-d1)*(s-d2)*(s-d_ss))
            if Area_square < 0:
                # negative due to floating point precision
                # if the point is very close to the trackline, then the trianlge area is increadibly small
                h = 0
                x = d_ss + d1
                # print(f"Area square is negative: {Area_square}")
            else:
                Area = Area_square**0.5
                h = Area * 2/d_ss
                x = (d1**2 - h**2)**0.5

        return x, h

    def calculate_progress(self, point):
        idx, dists = self.get_trackline_segment(point)
        x, h = self.interp_pts(idx, dists)
        s = self.ss[idx] + x
        
        return s
            
    def get_timed_trajectory_segment(self, position, dt = 0.1, n_pts=10):
        pose = np.array([position[0], position[1], position[3]])
        trajectory, distances = [pose], [0]
        # trajectory, distances = [], []
        for i in range(n_pts-1):
            distance = dt * pose[2]
            
            current_distance = self.calculate_progress(pose[0:2])
            next_distance = current_distance + distance
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
        
            plt.figure(6)
            plt.plot(self.ss, self.wpts[:, 0], label='waypoints')
            plt.plot(distances, interpolated_waypoints[:, 0], 'rx', label="Interp")
            print(distances)
            plt.xlabel("cumulative distance")
            
            # plt.figure(3)
            # plt.plot(self.ss, self.vs, label='waypoints')
            # plt.plot(distances, interpolated_waypoints[:, 2], 'rx', label="Interp")
            # print(distances)
            # plt.xlabel("cumulative distance")
            
            plt.figure(4)
            distances = np.array(distances)
            plt.plot(np.diff(distances), 'rx')
            plt.title("distance differences")
            
            # plt.show()
    
        return interpolated_waypoints



def measure_time():
    import time
    
    start_time = time.time()
    trajectory = Trajectory("esp")

    # trajectory.get_timed_trajectory_segment(np.array([10, 0]), 0.5, 20)
    print(f"Time elapsed: {time.time() - start_time}")
    trajectory.get_contsant_speed_timed_trajectory_segment(np.array([5, 0]), 0.1, 20)
    # trajectory.get_timed_trajectory_segment(np.array([20, -15]), 0.5, 20)
    print(f"Time elapsed: {time.time() - start_time}")
    trajectory.get_timed_trajectory_segment(np.array([20, -15]), 0.5, 20)
    print(f"Time elapsed: {time.time() - start_time}")

if __name__ == "__main__":
    trajectory = Trajectory("aut")
    trajectory_segment = trajectory.get_timed_trajectory_segment(np.array([7, 0]), 0.3, 10)
    print(trajectory_segment)
    
    trajectory.plot_wpts()
    
    plt.figure(5)
    
    points = trajectory_segment[:, 0:2].reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(0, 8)
    lc = LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(trajectory_segment[:, 2])
    lc.set_linewidth(2)
    line = plt.gca().add_collection(lc)
    plt.colorbar(line, fraction=0.046, pad=0.04, shrink=0.99)

    plt.axis('equal')
    plt.show()

    
