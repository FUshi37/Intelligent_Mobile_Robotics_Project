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

import numpy as np
import heapq

class AStarPlanner:
    def __init__(self, env, resolution=0.5):
        """
        Initialize the A* planner.
        :param env: FlightEnvironment object
        :param resolution: Grid resolution for discretizing the space
        """
        self.env = env
        self.res = resolution

    def heuristic(self, p1, p2):
        """Euclidean distance heuristic."""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def get_neighbors(self, current):
        """Find non-colliding and within-boundary neighbors."""
        neighbors = []
        
        for dx in [-self.res, 0, self.res]:
            for dy in [-self.res, 0, self.res]:
                for dz in [-self.res, 0, self.res]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    node = (round(current[0] + dx, 2), 
                            round(current[1] + dy, 2), 
                            round(current[2] + dz, 2))
                    
                    # Check boundaries and collisions
                    if not self.env.is_outside(node) and not self.env.is_collide(node):
                        neighbors.append(node)
        return neighbors

    def plan(self, start, goal):
        """
        Plan a path from start to goal using A*.
        :param start: (x, y, z) start coordinate
        :param goal: (x, y, z) goal coordinate
        :return: N x 3 numpy array of the path
        """
        start_node = tuple(round(x, 2) for x in start)
        goal_node = tuple(round(x, 2) for x in goal)

        open_list = []
        heapq.heappush(open_list, (0, start_node))
        
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: self.heuristic(start_node, goal_node)}

        while open_list:
            _, current = heapq.heappop(open_list)

            if self.heuristic(current, goal_node) < self.res:
                path = [goal_node]
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_node)
                return np.array(path[::-1])

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + self.heuristic(current, neighbor)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_node)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

        print("A* failed to find a path.")
        return None











