#!/usr/bin/env bash
# 在 pcd_to_pgm_demo 目录执行：bash run.sh
set -e
cd "$(dirname "$0")"

source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
colcon build --packages-select pcd_to_pgm_demo
source install/setup.bash

ros2 launch pcd_to_pgm_demo min_snap.launch.py
