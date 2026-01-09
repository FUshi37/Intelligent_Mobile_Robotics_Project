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

"""
Smooth Trajectory Generator

This module generates smooth trajectories that pass through given path points.
Uses cubic spline interpolation for smooth motion planning.

"""

import numpy as np
import matplotlib.pyplot as plt


class TrajectoryGenerator:
    """轨迹生成器类"""

    def __init__(self, path_points, total_time=10.0, time_step=0.1):
        """
        初始化轨迹生成器

        Args:
            path_points: N×3 numpy数组，路径点
            total_time: 总飞行时间 (秒)
            time_step: 时间步长 (秒)
        """
        self.path_points = np.array(path_points)
        self.total_time = total_time
        self.time_step = time_step
        self.num_points = len(path_points)

        # 计算每个段的时间分配（根据距离）
        self.segment_times = self._calculate_segment_times()
        self.time_points = np.cumsum(self.segment_times)
        self.time_points = np.insert(self.time_points, 0, 0)  # 在开头插入0

        # 生成轨迹
        self.trajectory = self._generate_trajectory()

    def _calculate_segment_times(self):
        """根据路径段距离计算时间分配"""
        distances = []
        for i in range(self.num_points - 1):
            dist = np.linalg.norm(self.path_points[i+1] - self.path_points[i])
            distances.append(dist)

        total_distance = sum(distances)

        # 每段的时间与距离成正比，但设置最小时间
        min_time_per_segment = self.total_time / self.num_points
        segment_times = []

        for dist in distances:
            time_allocation = max(min_time_per_segment,
                                (dist / total_distance) * self.total_time)
            segment_times.append(time_allocation)

        return np.array(segment_times)

    def _cubic_interpolation(self, t, p0, p1, v0=0, v1=0):
        """
        三次多项式插值
        从p0到p1，初始速度v0，终点速度v1
        """
        # 三次多项式: p(t) = a3*t^3 + a2*t^2 + a1*t + a0
        # 边界条件:
        # p(0) = p0, p(1) = p1
        # p'(0) = v0, p'(1) = v1

        a0 = p0
        a1 = v0
        a2 = 3*(p1 - p0) - 2*v0 - v1
        a3 = 2*(p0 - p1) + v0 + v1

        return a3*t**3 + a2*t**2 + a1*t + a0

    def _generate_trajectory(self):
        """生成完整轨迹"""
        trajectory_points = []

        # 时间点
        t_eval = np.arange(0, self.total_time + self.time_step, self.time_step)

        for t in t_eval:
            # 找到当前时间对应的路径段
            segment_idx = 0
            local_t = 0

            for i in range(len(self.segment_times)):
                if t <= self.time_points[i+1]:
                    segment_idx = i
                    local_t = (t - self.time_points[i]) / self.segment_times[i]
                    break
            else:
                # 如果超出总时间，使用最后一段
                segment_idx = len(self.segment_times) - 1
                local_t = 1.0

            # 确保local_t在[0,1]范围内
            local_t = np.clip(local_t, 0, 1)

            # 获取当前段的起点和终点
            p0 = self.path_points[segment_idx]
            p1 = self.path_points[segment_idx + 1]

            # 计算位置（三次插值）
            position = self._cubic_interpolation(local_t, p0, p1)

            trajectory_points.append(position)

        return np.array(trajectory_points)

    def get_trajectory_at_time(self, t):
        """获取指定时间的位置"""
        if t < 0:
            return self.path_points[0]
        if t >= self.total_time:
            return self.path_points[-1]

        # 找到对应的轨迹点
        time_idx = int(t / self.time_step)
        if time_idx >= len(self.trajectory):
            return self.trajectory[-1]

        return self.trajectory[time_idx]

    def plot_trajectory(self):
        """绘制轨迹时间历史"""
        t_eval = np.arange(0, self.total_time + self.time_step, self.time_step)

        # 确保轨迹和时间数组长度匹配
        min_len = min(len(t_eval), len(self.trajectory))
        t_eval = t_eval[:min_len]
        trajectory = self.trajectory[:min_len]

        # 创建子图
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Smooth Trajectory Time History', fontsize=16, fontweight='bold')

        # 定义更美观的颜色方案
        colors = {
            'x_line': '#2E86AB',      # 深蓝色
            'y_line': '#A23B72',      # 深紫色
            'z_line': '#F18F01',      # 橙色
            'points': '#C73E1D',      # 深红色
            'point_edge': '#FFFFFF'   # 白色边框
        }

        # X轴轨迹
        axes[0].plot(t_eval, trajectory[:, 0], color=colors['x_line'], 
                    linewidth=3.5, label='Trajectory', zorder=2, alpha=0.9)
        axes[0].scatter(self.time_points, self.path_points[:, 0], 
                       c=colors['points'], s=35, marker='o', 
                       label='Path Points', zorder=3, edgecolors=colors['point_edge'],
                       linewidths=1.5, alpha=0.8)
        axes[0].set_ylabel('X Position (m)', fontsize=12)
        axes[0].set_title('X Trajectory', fontsize=14, fontweight='bold')
        axes[0].legend(loc='best', fontsize=10)
        axes[0].grid(True, alpha=0.3, linestyle='--')

        # Y轴轨迹
        axes[1].plot(t_eval, trajectory[:, 1], color=colors['y_line'], 
                    linewidth=3.5, label='Trajectory', zorder=2, alpha=0.9)
        axes[1].scatter(self.time_points, self.path_points[:, 1], 
                       c=colors['points'], s=35, marker='o', 
                       label='Path Points', zorder=3, edgecolors=colors['point_edge'],
                       linewidths=1.5, alpha=0.8)
        axes[1].set_ylabel('Y Position (m)', fontsize=12)
        axes[1].set_title('Y Trajectory', fontsize=14, fontweight='bold')
        axes[1].legend(loc='best', fontsize=10)
        axes[1].grid(True, alpha=0.3, linestyle='--')

        # Z轴轨迹
        axes[2].plot(t_eval, trajectory[:, 2], color=colors['z_line'], 
                    linewidth=3.5, label='Trajectory', zorder=2, alpha=0.9)
        axes[2].scatter(self.time_points, self.path_points[:, 2], 
                       c=colors['points'], s=35, marker='o', 
                       label='Path Points', zorder=3, edgecolors=colors['point_edge'],
                       linewidths=1.5, alpha=0.8)
        axes[2].set_xlabel('Time (s)', fontsize=12)
        axes[2].set_ylabel('Z Position (m)', fontsize=12)
        axes[2].set_title('Z Trajectory', fontsize=14, fontweight='bold')
        axes[2].legend(loc='best', fontsize=10)
        axes[2].grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.show()

        # 打印轨迹信息
        print("轨迹生成完成:")
        print(f"  路径点数量: {self.num_points}")
        print(f"  总时间: {self.total_time:.1f} 秒")
        print(f"  时间步长: {self.time_step:.3f} 秒")
        print(f"  轨迹点数量: {len(self.trajectory)}")
        print(".3f")


def generate_smooth_trajectory(path, total_time=10.0):
    """
    生成平滑轨迹的主函数

    Args:
        path: N×3 numpy数组，路径点
        total_time: 总飞行时间 (秒)

    Returns:
        trajectory_generator: TrajectoryGenerator对象
    """
    return TrajectoryGenerator(path, total_time)

