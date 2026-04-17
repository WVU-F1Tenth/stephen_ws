#!/usr/bin/env bash

# Ensure script is sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced:"
    echo "  source set_map.sh <map-name>"
    exit 1
fi

read -p "Are you sure you want to install f1tenth? y/n: " answer
if [[ "$answer" != 'y' ]]; then
echo 'Cancelling...'
return 1
fi

set -o pipefail

echo 'Installing f1tenth_gym'

cd $HOME
source /opt/ros/foxy/setup.bash
git clone https://github.com/f1tenth/f1tenth_gym
cd f1tenth_gym && pip3 install -e .
pip install transforms3d

echo 'Installing f1tenth_gym_ros'

cd $HOME && mkdir -p sim_ws/src

cd $HOME/sim_ws/src
git clone https://github.com/f1tenth/f1tenth_gym_ros

echo 'Intalling f1tenth_gym_ros rosdeps'

source /opt/ros/foxy/setup.bash
cd ..
rosdep install --from-path src --rosdistro foxy -y

# Map path fix
sed -Ei "s|map_path: '.*'|map_path: '$HOME/sim_ws/src/f1tenth_gym_ros/maps/levine'|" ~/sim_ws/src/f1tenth_gym_ros/config/sim.yaml
colcon build

# Bashrc
echo 'Writing Aliases'

echo "alias sim='cd ~/sim_ws/; \
source install/setup.bash; \
ros2 launch f1tenth_gym_ros gym_bridge_launch.py'" >> ~/.bashrc

echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc

echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> ~/.bashrc

source ~/.bashrc