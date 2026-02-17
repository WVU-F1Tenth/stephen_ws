#!/usr/bin/env bash

# Ensure script is sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced:"
    echo "  source set_map.sh <map-name>"
    exit 1
fi

if [[ -z "$1" ]]; then
    echo "Usage: source set_map.sh <map-name>"
    return 1
fi

MAP_PATH="$HOME/stephen_ws/src/stephen/data/maps/$1/$1"

if [[ ! -f "${MAP_PATH}_map.png" && ! -f "${MAP_PATH}_map.pgm" ]]; then
    echo "ERROR: No png or pgm file"
    echo "Must have form <name>_map.png or <name>_map.pgm"
    return 1
fi

if [[ ! -f "${MAP_PATH}_map.yaml" ]]; then
    echo "ERROR: No yaml file"
    echo "Must have form <name>_map.yaml"
    return 1
fi

# Write MAP_PATH export to .bashrc
sed -i \
-e "/^export MAP_PATH=.*/d" \
-e "\$a export MAP_PATH=\"$MAP_PATH\"" \
"$HOME/.bashrc"

# Edit yaml file to accomodate _map naming
sed -Ei "s#(^[[:space:]]*)image:.*(\.png|\.pgm)#\1image: '${1}_map\2'#" \
    "${MAP_PATH}_map.yaml"

# Edit sim_ws yaml
sed -Ei "s|(^[[:space:]]*)map_path:.*|\1map_path: '${MAP_PATH}_map'|" \
    "$HOME/sim_ws/src/f1tenth_gym_ros/config/sim.yaml"

# Add to particle filter maps
PF_MAPS="$HOME/sim_ws/src/particle_filter/maps/"
cp "${MAP_PATH}_map.png" "$PF_MAPS" 2>/dev/null || cp "${MAP_PATH}_map.pgm" "$PF_MAPS"
cp "${MAP_PATH}_map.yaml" "$PF_MAPS"

# Add to raceline maps
RL_MAPS="$HOME/Raceline-Optimization/maps"
cp "${MAP_PATH}_map.png" "$RL_MAPS" 2>/dev/null || cp "${MAP_PATH}_map.pgm" "$RL_MAPS"
cp "${MAP_PATH}_map.yaml" "$RL_MAPS"
