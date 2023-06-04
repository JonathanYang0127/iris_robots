from iris_robots.sawyer.robot import SawyerRobot
from iris_robots.camera_utils.multi_camera_wrapper import MultiCameraWrapper
from iris_robots.server.robot_server import start_server

if __name__ == '__main__':
    robot = SawyerRobot()
    cameras = MultiCameraWrapper(camera_types=['cv2'], reverse=True)
    start_server(robot, cameras)
