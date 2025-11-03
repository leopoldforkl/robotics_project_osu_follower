#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Check if we want to use source config (for development)
    use_source_config = os.environ.get('USE_SOURCE_CONFIG', 'true').lower() == 'true'
    
    if use_source_config:
        # Use source config file for development (no rebuild needed)
        config_file = '/root/workspace/src/robot_control_pkg/config/p_controller_params.yaml'
    else:
        # Use installed config file (production)
        config_file = os.path.join(
            get_package_share_directory('robot_control_pkg'),
            'config',
            'p_controller_params.yaml'
        )
    
    return LaunchDescription([
        Node(
            package='robot_control_pkg',
            executable='robot_p_control_node',
            name='robot_p_control_node',
            parameters=[config_file],
            output='screen'
        )
    ])