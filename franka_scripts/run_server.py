from iris_robots.franka.robot import FrankaRobot
from iris_robots.camera_utils.multi_camera_wrapper import MultiCameraWrapper
from iris_robots.server.robot_server import start_server

if __name__ == '__main__':
    robot = FrankaRobot(blocking=True)
    cameras = MultiCameraWrapper(camera_types=['cv2'])
    start_server(robot, cameras)
