#!/usr/bin/env bash

# Map png/pgm and yaml file must be in src/stephen/data/maps/<map-name>
# 1. Cleans map image
# 2. Sets MAP_PATH variable the the current map and save in .bashrc
# 3. Copies map files to particle filter and raceline optimization 

# Ensure script is sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced:"
    echo "Usage: source set_map.sh <map-name>"
    exit 1
fi

if [[ -z "$1" ]]; then
    echo "Usage: source set_map.sh <map-name>"
    return 1
fi

export MAP_PATH="$HOME/stephen_ws/src/stephen/data/maps/$1/$1"
export MAP_NAME="$1"

if [[ ! -f "$MAP_PATH.png" && ! -f "$MAP_PATH.pgm" ]]; then
    echo "ERROR: No png or pgm file"
    return 1
fi

if [[ ! -f "$MAP_PATH.yaml" ]]; then
    echo "ERROR: No map yaml file"
    return 1
fi

# Write MAP_PATH export to .bashrc
sed -i \
    -e "/^export MAP_PATH=.*/d" \
    -e "\$a export MAP_PATH=\"$MAP_PATH\"" \
    "$HOME/.bashrc" ||
    echo "Error: Failed to edit .bashrc"

# Edit sim_ws yaml
if [[ -f "$HOME/sim_ws/src/f1tenth_gym_ros/config/sim.yaml" ]]; then
    sed -Ei "s|(^[[:space:]]*)map_path:.*|\1map_path: '$MAP_PATH'|" \
        "$HOME/sim_ws/src/f1tenth_gym_ros/config/sim.yaml" ||
        echo "Error: Failed to update sim yaml"
    echo "Sim set"
fi

# Clean map
if [[ -f ${MAP_PATH}.png ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    python "$SCRIPT_DIR/verify_map.py" || {
        echo "Error: Failed to clean map"
        return 1
    }
fi

# Add to particle filter maps
PF_MAPS="$HOME/sim_ws/src/particle_filter/maps"
if [[ -d "$PF_MAPS" && -f "$MAP_PATH.pgm" ]]; then
    # Copy map to pf maps
    cp "$MAP_PATH.pgm" "$PF_MAPS/" ||
        echo "Error: Failed to copy pgm map to pf maps"
    cp "$MAP_PATH.yaml" "$PF_MAPS/" ||
        echo "Error: Failed to copy map yaml to pf maps"
    # Edit map yaml map image name to assure pgm
    sed -Ei "s|(^[[:space:]]*)image:.*|\1image: '$MAP_NAME.pgm'|" \
        "$PF_MAPS/$MAP_NAME.yaml" ||
        echo "Error: Failed to update pf map yaml"
    # Edit localize.yaml map name
    sed -Ei "s|(^[[:space:]]*)map:.*|\1map: '$MAP_NAME'|" \
    "$HOME/sim_ws/src/particle_filter/config/localize.yaml" ||
        echo "Error: Failed to update pf localize.yaml"
    echo 'Particle filter set'
fi

# Add to raceline maps
RL_MAPS="$HOME/Raceline-Optimization/maps"
if [[ -d "$RL_MAPS" ]]; then
    cp "$MAP_PATH.png" "$RL_MAPS/" 2>/dev/null ||
        cp "$MAP_PATH.pgm" "$RL_MAPS/" ||
        echo "Error: Failed to copy map image to RL maps"
    cp "$MAP_PATH.yaml" "$RL_MAPS/" ||
        echo "Error: Failed to copy map yaml to RL maps"
    echo "Raceline optimization set"
fi

echo 'Map set'