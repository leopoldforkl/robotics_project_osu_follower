import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from PosEstimationModule import poseDetector


class PoseSkeletonPublisher(Node):
    def __init__(self):
        super().__init__('pose_skeleton_publisher')
        self.bridge = CvBridge()
        
        # Initialize pose detector
        self.detector = poseDetector()
        
        # Subscribe to camera topics
        self.imageSubscriber = self.create_subscription(
            Image, 
            '/camera/color/image_raw', 
            self.image_callback, 
            10
        )
        self.depthSubscriber = self.create_subscription(
            Image, 
            '/camera/depth/image_raw', 
            self.depth_callback, 
            10
        )
        
        # Publisher for skeleton markers
        self.marker_pub = self.create_publisher(MarkerArray, 'skeleton_markers', 10)
        
        # Static transform broadcaster
        self.static_broadcaster = StaticTransformBroadcaster(self)
        
        self.imageMatrix = None
        self.depthMatrix = None
        self.joints = None  # Will store the current pose landmarks
        
        # MediaPipe pose connections (edges between landmarks)
        # Based on MediaPipe's 33-landmark pose model
        self.edges = [
            (0, 1), (0, 4), (1, 3), (4, 6), (3, 7), (6, 8),
            (9, 10), (11, 12), (11, 13), (11, 23), (12, 14), (12, 24), (24, 23),
            (13, 15), (14, 16), (15, 17), (15, 19), (15, 21),
            (17, 19), (16, 18), (16, 20), (16, 22), (18, 20),
            (24, 26), (26, 28), (23, 25), (25, 27),
            (27, 31), (27, 29), (29, 31), (28, 32), (28, 30), (30, 32)
        ]
        
        # Publish static transform once
        self.publish_static_tf()
        
        # Create a timer to process images at regular intervals
        self.timer = self.create_timer(0.03, self.process_image)  # ~30 FPS
        
        self.get_logger().info('Pose Skeleton Publisher Node initialized')
        
    def publish_static_tf(self):
        """Publish static transform from world to camera frame"""
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
        
    def image_callback(self, msg):
        """Callback for color image"""
        self.imageMatrix = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
    def depth_callback(self, msg):
        """Callback for depth image"""
        self.depthMatrix = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        
    def process_image(self):
        """Process the image with pose detection and publish skeleton markers"""
        if self.imageMatrix is not None:
            # Detect pose and draw landmarks
            img = self.detector.findPose(self.imageMatrix.copy())
            
            # Get pose landmarks
            lmList, rawList = self.detector.getPosition(img)
            
            # Convert rawList to dictionary for easier access
            # rawList format: [[id, x, y, z], ...]
            if len(rawList) > 0:
                # Create a dictionary mapping landmark ID to 3D coordinates
                joints_dict = {}
                for landmark in rawList:
                    landmark_id = landmark[0]
                    x, y, z = landmark[1], landmark[2], landmark[3]
                    joints_dict[landmark_id] = (x, y, z)
                
                self.joints = joints_dict
                
                # Publish skeleton markers
                self.publish_skeleton()
                
                # Display landmark info
                self.get_logger().info(f'Detected {len(self.joints)} pose landmarks', throttle_duration_sec=2.0)
                
                # Example: Calculate angle for left arm if landmarks exist
                if len(lmList) > 15:
                    angle = self.detector.findAngle(img, 11, 13, 15, draw=True)
                    self.get_logger().info(f'Left elbow angle: {int(angle)}°', throttle_duration_sec=2.0)
            
            # Show FPS on image
            self.detector.showFps(img)
            
            # Display the processed image
            cv2.imshow("Pose Estimation - ROS", img)
            cv2.waitKey(1)
    
    def publish_skeleton(self):
        """Publish skeleton markers based on detected pose landmarks"""
        if self.joints is None or len(self.joints) == 0:
            return
        
        marker_array = MarkerArray()
        
        # ==========================================================
        # JOINTS MARKER (SPHERE_LIST)
        # ==========================================================
        joints_marker = Marker()
        joints_marker.header.frame_id = 'camera'
        joints_marker.header.stamp = self.get_clock().now().to_msg()
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

        # Add all detected joints
        for joint_id in sorted(self.joints.keys()):
            x, y, z = self.joints[joint_id]
            joints_marker.points.append(Point(x=float(x), y=float(y), z=float(z)))

        marker_array.markers.append(joints_marker)

        # ==========================================================
        # BONES (LINE_LIST)
        # ==========================================================
        bones_marker = Marker()
        bones_marker.header.frame_id = 'camera'
        bones_marker.header.stamp = self.get_clock().now().to_msg()
        bones_marker.ns = "bones"
        bones_marker.id = 1
        bones_marker.type = Marker.LINE_LIST
        bones_marker.action = Marker.ADD

        bones_marker.scale.x = 0.01
        bones_marker.color.a = 1.0
        bones_marker.color.r = 1.0
        bones_marker.color.g = 1.0
        bones_marker.color.b = 1.0

        # Add bones only if both endpoint joints exist
        for i, j in self.edges:
            if i in self.joints and j in self.joints:
                x1, y1, z1 = self.joints[i]
                x2, y2, z2 = self.joints[j]
                bones_marker.points.append(Point(x=float(x1), y=float(y1), z=float(z1)))
                bones_marker.points.append(Point(x=float(x2), y=float(y2), z=float(z2)))

        marker_array.markers.append(bones_marker)

        # ==========================================================
        # JOINT ID LABELS (TEXT MARKERS)
        # ==========================================================
        for joint_id in sorted(self.joints.keys()):
            x, y, z = self.joints[joint_id]
            
            text_marker = Marker()
            text_marker.header.frame_id = "camera"
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "joint_ids"
            text_marker.id = 1000 + joint_id  # unique ID for each joint

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

            text_marker.text = str(joint_id)

            marker_array.markers.append(text_marker)

        # Publish all markers
        self.marker_pub.publish(marker_array)


def main():
    rclpy.init()
    pose_skeleton_node = PoseSkeletonPublisher()
    
    try:
        rclpy.spin(pose_skeleton_node)
    except KeyboardInterrupt:
        pass
    finally:
        pose_skeleton_node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
