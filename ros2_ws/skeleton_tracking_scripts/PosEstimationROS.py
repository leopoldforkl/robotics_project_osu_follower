import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PosEstimationModule import poseDetector


class PoseEstimationROS(Node):
    def __init__(self):
        super().__init__('pose_estimation_node')
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
        
        self.imageMatrix = None
        self.depthMatrix = None
        
        # Create a timer to process images at regular intervals
        self.timer = self.create_timer(0.03, self.process_image)  # ~30 FPS
        
        self.get_logger().info('Pose Estimation ROS Node initialized')
        
    def image_callback(self, msg):
        """Callback for color image"""
        self.imageMatrix = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
    def depth_callback(self, msg):
        """Callback for depth image"""
        self.depthMatrix = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        
    def process_image(self):
        """Process the image with pose detection"""
        if self.imageMatrix is not None:
            # Detect pose and draw landmarks
            img = self.detector.findPose(self.imageMatrix.copy())
            
            # Get pose landmarks
            lmList, rawList = self.detector.getPosition(img)
            
            # Display landmark info if pose detected
            if len(lmList) > 0:
                self.get_logger().info(f'Detected {len(lmList)} pose landmarks')
                
                # Example: Calculate angle for left arm (landmarks 11, 13, 15)
                # 11: left shoulder, 13: left elbow, 15: left wrist
                if len(lmList) > 15:
                    angle = self.detector.findAngle(img, 11, 13, 15, draw=True)
                    self.get_logger().info(f'Left elbow angle: {int(angle)}°')
            
            # Show FPS on image
            self.detector.showFps(img)
            
            # Display the processed image
            cv2.imshow("Pose Estimation - ROS", img)
            cv2.waitKey(1)


def main():
    rclpy.init()
    pose_estimation_node = PoseEstimationROS()
    
    try:
        rclpy.spin(pose_estimation_node)
    except KeyboardInterrupt:
        pass
    finally:
        pose_estimation_node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
