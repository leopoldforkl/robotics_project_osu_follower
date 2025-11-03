#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
import math

class TargetFramePublisher(Node):
    def __init__(self):
        super().__init__('target_frame_publisher')
        
        # TF2 broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Initial position at t=0: (1.0, 0.5, 0.0)
        self.initial_x = 1.0
        self.initial_y = 0.5
        self.initial_z = 0.0
        
        # Movement parameters
        self.velocity = 0.5  # m/s - same speed for both straight and circular motion
        self.straight_distance = 2.0  # meters
        self.circle_radius = 1.0  # meters
        
        # Calculate timing for each segment
        self.straight_time = self.straight_distance / self.velocity  # time for straight segment
        self.circle_time = (math.pi * self.circle_radius) / self.velocity  # time for half circle
        self.total_cycle_time = 2 * self.straight_time + 2 * self.circle_time  # full cycle time
        
        # Start time
        self.start_time = self.get_clock().now()
        
        # Timer to publish transform at 50Hz
        self.timer = self.create_timer(0.02, self.publish_transform)
        
        self.get_logger().info(f'Target frame publisher started')
        self.get_logger().info(f'Cycle timing - Straight: {self.straight_time:.1f}s, Half-circle: {self.circle_time:.1f}s, Total: {self.total_cycle_time:.1f}s')
    
    def publish_transform(self):
        # Calculate elapsed time
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.start_time).nanoseconds / 1e9
        
        # Get position in current cycle
        cycle_time = elapsed_time % self.total_cycle_time
        
        # Calculate position and orientation based on current segment
        current_x, current_y, current_yaw = self.calculate_position_and_orientation(cycle_time)
        
        # Create transform message
        transform = TransformStamped()
        transform.header.stamp = current_time.to_msg()
        transform.header.frame_id = 'world'
        transform.child_frame_id = 'target'
        
        # Set translation
        transform.transform.translation.x = current_x
        transform.transform.translation.y = current_y
        transform.transform.translation.z = self.initial_z
        
        # Set rotation (yaw rotation only)
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = math.sin(current_yaw / 2.0)
        transform.transform.rotation.w = math.cos(current_yaw / 2.0)
        
        # Send transform
        self.tf_broadcaster.sendTransform(transform)
        
        # Log position every 2 seconds
        if int(elapsed_time) % 2 == 0 and abs(elapsed_time - int(elapsed_time)) < 0.02:
            segment = self.get_current_segment(cycle_time)
            self.get_logger().info(f'Target position: x={current_x:.2f}, y={current_y:.2f}, yaw={math.degrees(current_yaw):.1f}°, segment={segment}')
    
    def calculate_position_and_orientation(self, cycle_time):
        """Calculate position and orientation based on cycle time"""
        
        # Segment 1: First straight line (0 to straight_time)
        if cycle_time <= self.straight_time:
            progress = cycle_time
            x = self.initial_x + progress * self.velocity
            y = self.initial_y
            yaw = 0.0  # facing positive x direction
            
        # Segment 2: First half circle (straight_time to straight_time + circle_time)
        elif cycle_time <= self.straight_time + self.circle_time:
            progress = cycle_time - self.straight_time
            # Center of circle is at (initial_x + straight_distance, initial_y + radius)
            center_x = self.initial_x + self.straight_distance
            center_y = self.initial_y + self.circle_radius
            
            # Angle progress (from -π/2 to π/2 for left turn)
            angle = -math.pi/2 + (progress / self.circle_time) * math.pi
            
            x = center_x + self.circle_radius * math.cos(angle)
            y = center_y + self.circle_radius * math.sin(angle)
            yaw = angle + math.pi/2  # tangent to circle
            
        # Segment 3: Second straight line (straight_time + circle_time to 2*straight_time + circle_time)
        elif cycle_time <= 2 * self.straight_time + self.circle_time:
            progress = cycle_time - self.straight_time - self.circle_time
            start_x = self.initial_x + self.straight_distance
            start_y = self.initial_y + 2 * self.circle_radius
            
            x = start_x - progress * self.velocity  # moving in negative x direction
            y = start_y
            yaw = math.pi  # facing negative x direction
            
        # Segment 4: Second half circle (2*straight_time + circle_time to total_cycle_time)
        else:
            progress = cycle_time - 2 * self.straight_time - self.circle_time
            # Center of second circle is at (initial_x, initial_y + radius)
            center_x = self.initial_x
            center_y = self.initial_y + self.circle_radius
            
            # Angle progress (from π/2 to 3π/2 for left turn)
            angle = math.pi/2 + (progress / self.circle_time) * math.pi
            
            x = center_x + self.circle_radius * math.cos(angle)
            y = center_y + self.circle_radius * math.sin(angle)
            yaw = angle + math.pi/2  # tangent to circle
            
        return x, y, yaw
    
    def get_current_segment(self, cycle_time):
        """Get current segment name for logging"""
        if cycle_time <= self.straight_time:
            return "straight_1"
        elif cycle_time <= self.straight_time + self.circle_time:
            return "circle_1"
        elif cycle_time <= 2 * self.straight_time + self.circle_time:
            return "straight_2"
        else:
            return "circle_2"

def main(args=None):
    rclpy.init(args=args)
    node = TargetFramePublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()