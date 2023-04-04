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
            
    def get_timed_trajectory_segment(self, position, dt = 0.1, n_pts=10):
        trajectory, distances = [], []
        pose = position
        for i in range(n_pts):
            position_distances = np.linalg.norm(self.wpts-pose[0:2], axis=1)
            i = np.argmin(position_distances)
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

    
