#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
import tf2_ros
import tf_transformations
import math

class RobotSimulator(Node):
    def __init__(self):
        super().__init__('robot_simulator')

        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('initial_x', 0.0),
                ('initial_y', 0.0),
                ('initial_yaw', 0.0),
                ('publish_rate', 50.0),
                ('cmd_timeout', 0.5)
            ]
        )

        # Load them
        self.x = self.get_parameter('initial_x').value
        self.y = self.get_parameter('initial_y').value
        self.yaw = self.get_parameter('initial_yaw').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.cmd_timeout = self.get_parameter('cmd_timeout').value

        # Initialize velocities
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0

        self.last_cmd_time = self.get_clock().now()
        self.last_update_time = self.last_cmd_time

        # ROS interfaces
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom_sim', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.timer = self.create_timer(1.0 / self.publish_rate, self.update)

    def cmd_vel_callback(self, msg):
        self.vx = msg.linear.x
        self.vy = msg.linear.y
        self.wz = msg.angular.z
        self.last_cmd_time = self.get_clock().now()

    def update(self):
        now = self.get_clock().now()
        dt = (now - self.last_update_time).nanoseconds / 1e9
        self.last_update_time = now

        # Stop robot if cmd_vel timed out
        if (now - self.last_cmd_time).nanoseconds / 1e9 > self.cmd_timeout:
            self.vx = 0.0
            self.vy = 0.0
            self.wz = 0.0

        # Integrate motion
        self.x += (self.vx * math.cos(self.yaw) - self.vy * math.sin(self.yaw)) * dt
        self.y += (self.vx * math.sin(self.yaw) + self.vy * math.cos(self.yaw)) * dt
        self.yaw += self.wz * dt

        # Normalize yaw
        self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi

        # Publish transform
        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'robot_sim'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        q = tf_transformations.quaternion_from_euler(0, 0, self.yaw)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

        # Publish odometry (optional)
        odom = Odometry()
        odom.header = t.header
        odom.child_frame_id = t.child_frame_id
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation = t.transform.rotation
        self.odom_pub.publish(odom)

def main(args=None):
    rclpy.init(args=args)
    node = RobotSimulator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
