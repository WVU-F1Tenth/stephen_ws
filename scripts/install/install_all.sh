#!/usr/bin/env bash

# Ensure script is sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced:"
    echo "  source set_map.sh <map-name>"
    exit 1
fi

read -p "Are you sure you want to install f1tenth and ros2 foxy? y/n: " answer
if [[ "$answer" != 'y' ]]; then
echo 'Cancelling...'
return 1
fi

set -o pipefail

# Ros2 Foxy

sudo apt update && sudo apt upgrade -y && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

sudo apt install -y software-properties-common
sudo add-apt-repository -y universe

sudo apt update && sudo apt install -y curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt -y upgrade

sudo apt install -y ros-foxy-desktop python3-argcomplete python3-pip
sudo apt install -y ros-dev-tools

# Colcon

sudo apt install -y python3-colcon-common-extensions

# Dependencies

sudo apt install -y python3-bloom python3-rosdep fakeroot debhelper dh-python

sudo rm -f /etc/ros/rosdep/sources.list.d/20-default.list
sudo rosdep init
rosdep update --include-eol-distros

# F1Tenth
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
