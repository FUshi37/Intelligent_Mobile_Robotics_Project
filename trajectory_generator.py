"""
In this file, you should implement your trajectory generation class or function.
Your method must generate a smooth 3-axis trajectory (x(t), y(t), z(t)) that 
passes through all the previously computed path points. A positional deviation 
up to 0.1 m from each path point is allowed.

You should output the generated trajectory and visualize it. The figure must
contain three subplots showing x, y, and z, respectively, with time t (in seconds)
as the horizontal axis. Additionally, you must plot the original discrete path 
points on the same figure for comparison.

You are expected to write the implementation yourself. Do NOT copy or reuse any 
existing trajectory generation code from others. Avoid using external packages 
beyond general scientific libraries such as numpy, math, or scipy. If you decide 
to use additional packages, you must clearly explain the reason in your report.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

class TrajectoryGenerator:
    def __init__(self, path):
        """
        Initialize the Trajectory Generator.
        :param path: N x 3 numpy array of path points
        """
        self.path = np.array(path)
        
        # Calculate cumulative distance along path as time parameter
        distances = np.sqrt(np.sum(np.diff(self.path, axis=0)**2, axis=1))
        # Ensure distances are strictly positive to avoid duplicate time values
        min_distance = 1e-6
        distances = np.maximum(distances, min_distance)
        
        self.t = np.zeros(len(self.path))
        self.t[1:] = np.cumsum(distances)
        
        # Create cubic spline interpolators for x, y, z
        self.cs_x = CubicSpline(self.t, self.path[:, 0])
        self.cs_y = CubicSpline(self.t, self.path[:, 1])
        self.cs_z = CubicSpline(self.t, self.path[:, 2])

    def generate(self, num_points=1000):
        """
        Generate smooth trajectory points.
        :param num_points: Number of points to generate along the trajectory
        :return: t_fine (time array), trajectory (N x 3 array)
        """
        t_fine = np.linspace(self.t[0], self.t[-1], num_points)
        x_fine = self.cs_x(t_fine)
        y_fine = self.cs_y(t_fine)
        z_fine = self.cs_z(t_fine)
        
        trajectory = np.vstack((x_fine, y_fine, z_fine)).T
        return t_fine, trajectory

    def plot_trajectory(self, t_fine, trajectory):
        """
        Plot x, y, z trajectories versus time in three subplots.
        Also plot the original discrete path points for comparison.
        """
        fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        
        # 使用更美观的配色方案
        colors = ['#E74C3C', '#2ECC71', '#3498DB']  # 红色、绿色、蓝色
        labels = ['x (m)', 'y (m)', 'z (m)']
        titles = ['X-axis Position', 'Y-axis Position', 'Z-axis Position']
        
        for i in range(3):
            # 绘制平滑轨迹曲线
            axs[i].plot(t_fine, trajectory[:, i], color=colors[i], 
                       linewidth=2.5, label='Smooth Trajectory', alpha=0.9)
            
            # 绘制原始路径点 - 使用更美观的样式
            axs[i].scatter(self.t, self.path[:, i], 
                          c=colors[i], s=80, marker='o', 
                          edgecolors='white', linewidths=2,
                          label='Path Points', zorder=5, alpha=0.8)
            
            # 美化坐标轴
            axs[i].set_ylabel(labels[i], fontsize=13, fontweight='bold')
            axs[i].set_title(titles[i], fontsize=11, pad=10)
            axs[i].legend(loc='best', framealpha=0.95, fontsize=10)
            axs[i].grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
            axs[i].set_facecolor('#F8F9FA')
            
            # 添加轴的边框样式
            for spine in axs[i].spines.values():
                spine.set_linewidth(1.2)
                spine.set_color('#CCCCCC')

        axs[2].set_xlabel('Time (s)', fontsize=13, fontweight='bold')
        plt.suptitle('Trajectory Generation: Position vs Time', 
                    fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0.01, 1, 0.99])
        plt.show()

