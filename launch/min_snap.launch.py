import os
from launch_ros.actions import Node
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    params_file = os.path.join(get_package_share_directory("pcd_to_pgm_demo"), "config", "params.yaml")
    return LaunchDescription([
        Node(
            package="pcd_to_pgm_demo",
            executable="pcd_to_pgm_node",
            name="pcd_to_pgm",
            output="screen",
            parameters=[params_file],
        ),
        Node(
            package="pcd_to_pgm_demo",
            executable="astar_planner_node",
            name="astar_planner",
            output="screen",
            parameters=[params_file],
        ),
        Node(
            package="pcd_to_pgm_demo",
            executable="min_snap_node",
            name="min_snap",
            output="screen",
            parameters=[params_file],
        ),
    ])
