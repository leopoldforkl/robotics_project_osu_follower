import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from HandTrackingModule import handDedector


class HandTrackingROS(Node):
    def __init__(self):
        super().__init__('hand_tracking_node')
        self.bridge = CvBridge()
        
        # Initialize hand detector
        self.detector = handDedector()
        
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
        
        self.get_logger().info('Hand Tracking ROS Node initialized')
        
    def image_callback(self, msg):
        """Callback for color image"""
        self.imageMatrix = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
    def depth_callback(self, msg):
        """Callback for depth image"""
        self.depthMatrix = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        
    def process_image(self):
        """Process the image with hand detection"""
        if self.imageMatrix is not None:
            # Detect hands and draw landmarks
            img = self.detector.findHands(self.imageMatrix.copy())
            
            # Get hand landmarks
            lmList = self.detector.findPositon(img)
            
            # Display landmark info if hands detected
            if len(lmList) > 0:
                self.get_logger().info(f'Detected {len(lmList)} landmarks')
                # Example: print tip of index finger (landmark 8)
                if len(lmList) > 8:
                    self.get_logger().info(f'Index finger tip: {lmList[8]}')
            
            # Display the processed image
            cv2.imshow("Hand Tracking - ROS", img)
            cv2.waitKey(1)


def main():
    rclpy.init()
    hand_tracking_node = HandTrackingROS()
    
    try:
        rclpy.spin(hand_tracking_node)
    except KeyboardInterrupt:
        pass
    finally:
        hand_tracking_node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
