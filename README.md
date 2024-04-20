# iris_robots
## Environment Setup
We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and setting up an environment:  \
    ```
    conda create --name iris_robots python=3.10
    ```

First, install the required packages:   \
    ```
    pip install -r requirements.txt
    ```

Finally, install the iris_robots module:   \
    ```
    pip install -e .
    ```

## WidowX250S Setup
First, move to the WidowX control directory: \\
```
cd iris_robots/widowx
```

Then, follow the instructions here to set up interbotix ws and ROS:
https://www.trossenrobotics.com/docs/interbotix_xsarms/ros_interface/software_setup.html
Copy the contents of the interbotix_ws to this directory: \\

```
mv interbotix_ws/src/* src
catkin_make 
```

## Start the WidowX205S Robot
Again, move to the WidowX control directory. Then, run the following commands in a separate terminal: \
```
cd iris_robots/widowx
source devel/setup.bash
sh launch_robot wx250s
```
This should open an RViz window with the WidowX robot


## WidowX Default Joints
Set the WidowX default joints in [params.py](https://github.com/JonathanYang0127/iris_robots/blob/release/params.py):
This can be found by using ROS: \
```
rostopic echo /wx250s/joint_states
```
Move the robot to the desired position and copy the joint states to the 'wx250s' robot.

## Teleoperation
Follow the instructions to set up an Oculus in [oculus_reader](https://github.com/rail-berkeley/oculus_reader). You can now control the robot with VR teleoperation using the following command:
```
python scripts/collect_trajectories_teleop.py
```


