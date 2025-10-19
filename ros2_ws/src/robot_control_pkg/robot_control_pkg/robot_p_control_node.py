#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import numpy as np
import math
from .p_controller import compute_cmd_vel


class RobotPControlNode(Node):
    def __init__(self):
        super().__init__('robot_p_control_node')
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('kp_linear', 3.0),
                ('kp_angular', 3.0),
                ('max_linear_velocity', 2.0),
                ('max_angular_velocity', 1.0),
                ('x_offset', 1.0),
                ('y_offset', 0.0),
                ('angle_offset', 0.7854)
            ]
        )
        
        # Create publisher for cmd_vel
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Frame names
        self.robot_frame = 'robot_sim'
        self.target_frame = 'target'  # Assuming this is the target frame name
        
        # State tracking for controller inputs
        self.prev_target_position = None
        self.prev_cmd_vel = np.zeros(6)
        self.prev_time = None
        
        # Create timer to publish commands at 10Hz
        self.timer = self.create_timer(0.1, self.publish_cmd_vel)
        
        self.get_logger().info('Robot P Control Node started - using P controller with TF frames')
    
    def publish_cmd_vel(self):
        # Initialize command velocities to zero
        cmd_vel_array = np.zeros(6)
        current_time = self.get_clock().now()
        
        try:
            # Get transform from robot_sim frame to target frame
            transform = self.tf_buffer.lookup_transform(
                self.robot_frame,
                self.target_frame,
                rclpy.time.Time()
            )
            
            # Extract target position as 3x1 numpy array
            target_position = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            
            # Calculate dt in milliseconds
            if self.prev_time is not None:
                dt_ns = (current_time - self.prev_time).nanoseconds
                dt_ms = dt_ns / 1e6  # Convert nanoseconds to milliseconds
            else:
                dt_ms = 100.0  # Default 100ms for first iteration
            
            # Set previous target position (use current if not available)
            if self.prev_target_position is None:
                prev_target_pos = target_position.copy()
            else:
                prev_target_pos = self.prev_target_position.copy()
            
            # Get controller parameters
            controller_params = {
                'kp_linear': self.get_parameter('kp_linear').value,
                'kp_angular': self.get_parameter('kp_angular').value,
                'max_linear_velocity': self.get_parameter('max_linear_velocity').value,
                'max_angular_velocity': self.get_parameter('max_angular_velocity').value,
                'x_offset': self.get_parameter('x_offset').value,
                'y_offset': self.get_parameter('y_offset').value,
                'angle_offset': self.get_parameter('angle_offset').value
            }
            
            # Compute command velocities using the controller
            cmd_vel_array = compute_cmd_vel(
                target_position, 
                prev_target_pos, 
                self.prev_cmd_vel, 
                dt_ms,
                controller_params
            )
            
            # Update state for next iteration
            self.prev_target_position = target_position.copy()
            self.prev_cmd_vel = cmd_vel_array.copy()
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            # Frames not available, use zero velocities
            self.get_logger().debug(f'TF lookup failed: {e}')
            cmd_vel_array = np.zeros(6)
            # For now, just return zeros (no control logic implemented yet)
            # Reset state when frames are not available
            self.prev_target_position = None
            self.prev_cmd_vel = np.zeros(6)
        
        # Update time for next iteration
        self.prev_time = current_time
        
        # Create Twist message from numpy array
        cmd_vel = Twist()
        cmd_vel.linear.x = float(cmd_vel_array[0])
        cmd_vel.linear.y = float(cmd_vel_array[1])
        cmd_vel.linear.z = float(cmd_vel_array[2])
        cmd_vel.angular.x = float(cmd_vel_array[3])
        cmd_vel.angular.y = float(cmd_vel_array[4])
        cmd_vel.angular.z = float(cmd_vel_array[5])
        
        # Publish the command
        self.cmd_vel_pub.publish(cmd_vel)


def main(args=None):
    rclpy.init(args=args)
    node = RobotPControlNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()