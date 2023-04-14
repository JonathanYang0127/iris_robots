import rospy, actionlib
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from fetch_driver_msgs.msg import GripperState
from threading import Lock

class GripperController:
    def __init__(self, create_node=False):
        self._joint_lock = Lock()
        self.gripperClient = actionlib.SimpleActionClient("gripper_controller/gripper_action", GripperCommandAction)
        self.gripperClient.wait_for_server()
        rospy.Subscriber(f"/gripper_state", GripperState, self._gripper_callback)
        self.gripper_state = None

    def _gripper_callback(self, msg):
        with self._joint_lock:
            self.gripper_state = msg.joints[0].position

    def set_position(self, target):
        target = self.denormalize(target)
        goal = GripperCommandGoal()
        goal.command.max_effort = 10.0
        goal.command.position = target
        gripperClient.send_goal(target)

    def get_continuous_position(self):
        return self.normalize(self.gripper_state)

    def normalize(self, x):
        return 1 - x

    def denormalize(self, x):
        return 1 - x
