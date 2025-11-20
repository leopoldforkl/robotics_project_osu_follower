import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.timer = self.create_timer(0.033, self.timer_callback) # 30 FPS
        self.cap = cv2.VideoCapture(0)
        self.bridge = CvBridge()
        self.get_logger().info('Camera Publisher Node has been started.')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Publish the image
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.publisher_.publish(msg)
            
            # Display the image
            cv2.imshow('Camera Stream', frame)
            cv2.waitKey(1)
        else:
            self.get_logger().warn('Failed to capture image')

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisher()
    try:
        rclpy.spin(camera_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        camera_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
