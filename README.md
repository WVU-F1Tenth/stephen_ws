# Setup venv
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
> colcon build must be done in venv

# To activate venv
```bash
source ./venv/bin/activate
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