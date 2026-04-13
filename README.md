# General info
- names icps1 and icps2
- ip: 190.160.1.121 and 190.160.1.122
- May need: export ROS_DOMAIN_ID=

# Aliases
SLAM
```
alias slam="cd $HOME/sim_ws/; source install/setup.bash; ros2 launch slam_toolbox online_async_launch.py slam_params_file:=/home/icps2/f1tenth_ws/src/f1tenth_system/f1tenth_stack/config/f1tenth_online_async.yaml"
```
Particle filter
```
alias particle_filter="cd $HOME/sim_ws/; source install/setup.bash; ros2 launch particle_filter localize_launch.py"
```
F1tenth simulator
```
alias sim='cd ~/sim_ws/; source install/local_setup.bash; ros2 launch f1tenth_gym_ros gym_bridge_launch.py'
```
F1Tenth on car
```
alias f1tenth="cd $HOME/f1tenth_ws/; source install/setup.bash; ros2 launch f1tenth_stack bringup_launch.py"
```

# Setup venv
```bash
sudo apt install python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade 'pip<24.1' setuptools wheel
pip install -r requirements.txt
```
> colcon build must be done in venv

# To activate venv
```bash
source .venv/bin/activate
source install/setup.bash
```

# To deactivate
```bash
deactivate
```

# Commands (ros2 run stephen *)
- pure_pursuit - Arc based waypoint following
- stanley - Error based waypoint following with heading
- path_follow - Disparity following algorithm with extension
- gap_follow - Gap follow algorithm with naive disparity extension
- test - Path follow template that can be modified for testing
- scan_visual - Vispy app that displays scans and additional information