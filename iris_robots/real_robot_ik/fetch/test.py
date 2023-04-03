import pybullet as p
p.connect(p.DIRECT)

robot_id = p.loadURDF("fetch_description/robots/fetch.urdf")
print(p.getNumJoints(robot_id))
joint_names = ["shoulder_pan_joint",
            "shoulder_lift_joint", "upperarm_roll_joint",
            "elbow_flex_joint", "forearm_roll_joint",
            "wrist_flex_joint", "wrist_roll_joint"]
joint_map = [0] * len(joint_names)
for i in range(24):
    info = p.getJointInfo(robot_id, i)
    print(info)



