from flight_environment import FlightEnvironment
from path_planner import plan_path
from trajectory_generator import generate_smooth_trajectory
import time

# 创建飞行环境
env = FlightEnvironment(50)
start = (1, 2, 0)
goal = (18, 18, 3)

print("="*60)
print("    智能移动机器人项目 - RRT路径规划")
print("="*60)
print(f"环境大小: {env.env_width} x {env.env_length} x {env.env_height}")
print(f"障碍物数量: {len(env.cylinders)}")
print(f"起点: {start}")
print(f"终点: {goal}")
print()

# --------------------------------------------------------------------------------------------------- #
# 路径规划阶段
# 使用RRT算法规划从起点到终点的无碰撞路径

print("开始路径规划...")
start_time = time.time()

path = plan_path(env, start, goal)

planning_time = time.time() - start_time

if path is None:
    print("路径规划失败！无法找到从起点到终点的可行路径。")
    print("可能的原因:")
    print("1. 环境中的障碍物过多")
    print("2. 起点或终点在障碍物内")
    print("3. RRT参数需要调整")
    exit(1)

print(f"路径规划成功！用时: {planning_time:.2f}秒")
print(f"路径点数量: {len(path)}")
print()

# --------------------------------------------------------------------------------------------------- #


# 3D路径可视化
print("生成3D路径可视化...")
env.plot_cylinders(path)


# --------------------------------------------------------------------------------------------------- #
# 轨迹生成阶段
# 生成通过所有路径点的平滑轨迹

print("开始轨迹生成...")
trajectory_gen = generate_smooth_trajectory(path, total_time=12.0)

print("生成轨迹可视化...")
trajectory_gen.plot_trajectory()




# --------------------------------------------------------------------------------------------------- #



# You must manage this entire project using Git. 
# When submitting your assignment, upload the project to a code-hosting platform 
# such as GitHub or GitLab. The repository must be accessible and directly cloneable. 
#
# After cloning, running `python3 main.py` in the project root directory 
# should successfully execute your program and display:
#   1) the 3D path visualization, and
#   2) the trajectory plot.
#
# You must also include the link to your GitHub/GitLab repository in your written report.
