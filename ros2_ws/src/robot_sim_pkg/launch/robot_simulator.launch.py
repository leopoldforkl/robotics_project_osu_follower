#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the path to the config file
    config_file = os.path.join(
        get_package_share_directory('robot_sim_pkg'),
        'config',
        'robot_sim_params.yaml'
    )
    
    return LaunchDescription([
        Node(
            package='robot_sim_pkg',
            executable='robot_simulator',
            name='robot_simulator',
            parameters=[config_file],
            output='screen'
        )
    ])