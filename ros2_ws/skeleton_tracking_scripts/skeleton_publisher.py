import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import numpy as np

class SkeletonPublisher(Node):
    def __init__(self):
        super().__init__('skeleton_pub')

        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, 'skeleton_markers', 10)
        self.static_broadcaster = StaticTransformBroadcaster(self)

        # Your real edges
        self.edges = [
            (0,1), (0,4), (1,3), (4,6), (3,7), (6,8),
            (9,10), (11,12), (11,13), (11,23), (12,14), (12,24), (24,23),
            (13,15), (14,16), (15,17), (15,19), (15,21),
            (17,19), (16,18), (16,20), (16,22), (18,20),
            (24,26), (26,28), (23,25), (25,27),
            (27,31), (27,29), (29,31), (28,32), (28,30), (30,32)
        ]

        # Publish static transform once
        self.publish_static_tf()

        # Timer to publish skeleton continuously at 2 Hz
        self.timer = self.create_timer(0.5, self.publish_skeleton)

    def publish_static_tf(self):
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = 'world'
        tf.child_frame_id = 'camera'

        # Zero translation and rotation
        tf.transform.translation.x = 0.0
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = 0.0

        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = 0.0
        tf.transform.rotation.w = 1.0

        self.static_broadcaster.sendTransform(tf)

    def publish_skeleton(self):
        # Generate new random skeleton data each update
        self.joints = np.random.rand(33, 3)
        
        marker_array = MarkerArray()

        # ==========================================================
        # JOINTS MARKER (SPHERE_LIST)
        # ==========================================================
        joints_marker = Marker()
        joints_marker.header.frame_id = 'camera'
        joints_marker.ns = "joints"
        joints_marker.id = 0
        joints_marker.type = Marker.SPHERE_LIST
        joints_marker.action = Marker.ADD

        joints_marker.scale.x = 0.05
        joints_marker.scale.y = 0.05
        joints_marker.scale.z = 0.05

        joints_marker.color.a = 1.0
        joints_marker.color.r = 1.0
        joints_marker.color.g = 0.0
        joints_marker.color.b = 0.0

        for (x, y, z) in self.joints:
            joints_marker.points.append(Point(x=x, y=y, z=z))

        marker_array.markers.append(joints_marker)

        # ==========================================================
        # BONES (LINE_LIST)
        # ==========================================================
        bones_marker = Marker()
        bones_marker.header.frame_id = 'camera'
        bones_marker.ns = "bones"
        bones_marker.id = 1
        bones_marker.type = Marker.LINE_LIST
        bones_marker.action = Marker.ADD

        bones_marker.scale.x = 0.01
        bones_marker.color.a = 1.0
        bones_marker.color.r = 1.0
        bones_marker.color.g = 1.0
        bones_marker.color.b = 1.0

        for i, j in self.edges:
            bones_marker.points.append(Point(x=float(self.joints[i][0]), y=float(self.joints[i][1]), z=float(self.joints[i][2])))
            bones_marker.points.append(Point(x=float(self.joints[j][0]), y=float(self.joints[j][1]), z=float(self.joints[j][2])))

        marker_array.markers.append(bones_marker)

        # ==========================================================
        # JOINT ID LABELS (TEXT MARKERS)
        # ==========================================================
        for idx, (x, y, z) in enumerate(self.joints):
            text_marker = Marker()
            text_marker.header.frame_id = "camera"
            text_marker.ns = "joint_ids"
            text_marker.id = 1000 + idx  # unique ID for each joint

            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.scale.z = 0.08   # text height

            text_marker.color.a = 1.0
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0

            text_marker.pose.position.x = float(x)
            text_marker.pose.position.y = float(y)
            text_marker.pose.position.z = float(z) + 0.03  # slightly above joint

            text_marker.text = str(idx)

            marker_array.markers.append(text_marker)

        # Publish everything
        self.marker_pub.publish(marker_array)


def main():
    rclpy.init()
    node = SkeletonPublisher()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
