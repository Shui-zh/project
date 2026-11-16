#!/usr/bin/env python3
"""一条命令启动全部三个 demo 节点。

放置位置: src/min_snap_planner_demo/launch/run_all.launch.py
启动方式: ros2 launch min_snap_planner_demo run_all.launch.py
可选参数: rviz:=true  delay:=2.0  log_level:=info
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def include_pkg_launch(package, filename, **kwargs):
    """把某个包 launch/ 下的文件包含进来（走 install 后的 share 目录）。"""
    path = os.path.join(get_package_share_directory(package), 'launch', filename)
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(path), **kwargs)


def generate_launch_description() -> LaunchDescription:
    start_rviz = LaunchConfiguration('rviz')
    delay = LaunchConfiguration('delay')

    declared_arguments = [
        DeclareLaunchArgument('rviz', default_value='false', description='是否启动 RViz2'),
        DeclareLaunchArgument('delay', default_value='2.0',
                              description='pcd 转图后延迟多少秒再起规划节点'),
    ]

    # 1) 点云 → 栅格图：最先起，它产出的地图是后两个节点的输入
    pcd_to_pgm = include_pkg_launch('pcd_to_pgm_demo', 'pcd_to_pgm.launch.py')

    # 2) A* 全局规划 + 3) min-snap 轨迹优化：等地图写完再起，避免读到空文件
    planners = TimerAction(
        period=delay,
        actions=[
            include_pkg_launch('astar_planner_demo', 'astar_planner.launch.py'),
            include_pkg_launch('min_snap_planner_demo', 'min_snap_planner.launch.py'),
        ],
    )

    # 可选可视化；启动时加 rviz:=true 才生效
    rviz_cfg = os.path.join(
        get_package_share_directory('min_snap_planner_demo'), 'config', 'min_snap.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_cfg] if os.path.isfile(rviz_cfg) else [],
        condition=IfCondition(start_rviz),
    )

    return LaunchDescription([
        *declared_arguments,
        pcd_to_pgm,
        planners,
        rviz,
    ])
