"""
In this file, you should implement your own path planning class or function.
Within your implementation, you may call `env.is_collide()` and `env.is_outside()`
to verify whether candidate path points collide with obstacles or exceed the
environment boundaries.

You are required to write the path planning algorithm by yourself. Copying or calling 
any existing path planning algorithms from others is strictly
prohibited. Please avoid using external packages beyond common Python libraries
such as `numpy`, `math`, or `scipy`. If you must use additional packages, you
must clearly explain the reason in your report.
"""
            
"""
RRT (Rapidly-exploring Random Tree) Path Planning Algorithm

This file implements a RRT algorithm for 3D path planning in environments with cylindrical obstacles.
The algorithm generates a collision-free path from start to goal position.

"""

import numpy as np
import math
import time


class Node:
    """RRT树中的节点类"""
    def __init__(self, position, parent=None):
        self.position = np.array(position)  # (x, y, z)
        self.parent = parent
        self.cost = 0.0  # 从根节点到此节点的代价

    def distance_to(self, other):
        """计算到另一个节点或位置的欧几里得距离"""
        if hasattr(other, 'position'):
            return np.linalg.norm(self.position - other.position)
        else:
            return np.linalg.norm(self.position - np.array(other))


class RRTPlanner:
    """RRT路径规划器"""

    def __init__(self, env, start, goal, max_iter=5000, step_size=0.5, goal_sample_rate=0.05):
        """
        初始化RRT规划器

        Args:
            env: 飞行环境对象，包含碰撞检测和边界检查方法
            start: 起始位置 (x, y, z)
            goal: 目标位置 (x, y, z)
            max_iter: 最大迭代次数
            step_size: 每次扩展的步长
            goal_sample_rate: 采样目标点的概率
        """
        self.env = env
        self.start = Node(start)
        self.goal = Node(goal)
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate

        # RRT树
        self.nodes = [self.start]

        # 环境边界
        self.x_min, self.x_max = 0, env.env_width
        self.y_min, self.y_max = 0, env.env_length
        self.z_min, self.z_max = 0, env.env_height

    def sample_random_point(self):
        """随机采样一个点"""
        if np.random.random() < self.goal_sample_rate:
            return self.goal.position
        else:
            x = np.random.uniform(self.x_min, self.x_max)
            y = np.random.uniform(self.y_min, self.y_max)
            z = np.random.uniform(self.z_min, self.z_max)
            return np.array([x, y, z])

    def find_nearest_node(self, point):
        """找到树中距离给定点最近的节点"""
        distances = [node.distance_to(point) for node in self.nodes]
        nearest_index = np.argmin(distances)
        return self.nodes[nearest_index]

    def steer(self, from_node, to_point):
        """从from_node向to_point方向扩展step_size距离"""
        direction = to_point - from_node.position
        distance = np.linalg.norm(direction)

        if distance < self.step_size:
            return to_point
        else:
            direction = direction / distance
            return from_node.position + direction * self.step_size

    def is_collision_free(self, point1, point2, num_checks=10):
        """检查从point1到point2的直线路径是否无碰撞"""
        direction = point2 - point1
        distance = np.linalg.norm(direction)

        if distance < 1e-6:
            # 如果两点非常接近，只检查point2
            return not (self.env.is_collide(point2) or self.env.is_outside(point2))

        direction = direction / distance

        # 在路径上进行多点检查
        for i in range(num_checks + 1):
            t = i / num_checks
            check_point = point1 + t * direction * distance

            if self.env.is_collide(check_point) or self.env.is_outside(check_point):
                return False

        return True

    def plan(self):
        """
        执行RRT路径规划

        Returns:
            path: N×3 numpy数组，包含路径点，如果规划失败返回None
        """
        print("开始RRT路径规划...")
        print(f"起点: {self.start.position}")
        print(f"终点: {self.goal.position}")
        print(f"最大迭代次数: {self.max_iter}")

        start_time = time.time()

        for i in range(self.max_iter):
            # 1. 随机采样
            random_point = self.sample_random_point()

            # 2. 找到最近节点
            nearest_node = self.find_nearest_node(random_point)

            # 3. 向采样点扩展
            new_point = self.steer(nearest_node, random_point)

            # 4. 碰撞检测
            if not self.is_collision_free(nearest_node.position, new_point):
                continue

            # 5. 创建新节点
            new_node = Node(new_point, parent=nearest_node)
            new_node.cost = nearest_node.cost + nearest_node.distance_to(new_point)
            self.nodes.append(new_node)

            # 6. 检查是否到达目标
            if new_node.distance_to(self.goal) < self.step_size:
                if self.is_collision_free(new_node.position, self.goal.position):
                    # 找到路径！
                    print(f"找到路径！总迭代次数: {i+1}")
                    path = self.extract_path(new_node)
                    end_time = time.time()
                    print(".3f")
                    return path

            # 每100次迭代打印一次进度
            if (i + 1) % 100 == 0:
                print(f"迭代 {i+1}/{self.max_iter}, 当前树大小: {len(self.nodes)}")

        print(f"路径规划失败，未能在 {self.max_iter} 次迭代内找到路径")
        return None

    def extract_path(self, goal_node):
        """从目标节点回溯提取路径"""
        path = []
        current = goal_node

        while current is not None:
            path.append(current.position)
            current = current.parent

        # 反转路径（从起点到终点）
        path.reverse()

        # 添加目标点
        path.append(self.goal.position)

        return np.array(path)


def plan_path(env, start, goal):
    """
    RRT路径规划的主函数

    Args:
        env: 飞行环境对象
        start: 起始位置 (x, y, z)
        goal: 目标位置 (x, y, z)

    Returns:
        path: N×3 numpy数组，包含路径点，如果规划失败返回None
    """
    planner = RRTPlanner(
        env=env,
        start=start,
        goal=goal,
        max_iter=8000,  # 增加迭代次数以提高成功率
        step_size=0.3,  # 减小步长以提高路径质量
        goal_sample_rate=0.1  # 增加采样目标点的概率
    )

    return planner.plan()


















