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
echo "Writing to .bashrc"
sed -i \
-e "/^export MAP_PATH=.*/d" \
-e "\$a export MAP_PATH=\"$MAP_PATH\"" \
"$HOME/.bashrc"

# Edit map yaml file to accomodate _map naming
sed -Ei "s#(^[[:space:]]*)image:.*(\.png|\.pgm)'?#\1image: '${1}_map\2'#" \
    "${MAP_PATH}_map.yaml"

# Edit sim_ws yaml
sed -Ei "s|(^[[:space:]]*)map_path:.*|\1map_path: '${MAP_PATH}_map'|" \
    "$HOME/sim_ws/src/f1tenth_gym_ros/config/sim.yaml"

# Clean map
if [[ -f ${MAP_PATH}_map.png ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    python "$SCRIPT_DIR/verify_map.py" || return 1
fi

# Add to particle filter maps
PF_MAPS="$HOME/sim_ws/src/particle_filter/maps/"
if [ -d "$PF_MAPS" ]; then
    # Copy map to pf maps
    cp "${MAP_PATH}_map.pgm" "$PF_MAPS/$1.pgm"
    cp "${MAP_PATH}_map.yaml" "$PF_MAPS/$1.yaml"
    echo "Copied pgm map to particle filter"
    # Edit localize.yaml map name
    sed -Ei "s|(^[[:space:]]*)map:.*|\1map: '${1}'|" \
    "$HOME/sim_ws/src/particle_filter/config/localize.yaml"
    echo "Updated pf localize yaml"
    # Edit map yaml map image name
    sed -Ei "s|(^[[:space:]]*)map_path:.*|\1map_path: '${1}.pgm'|" \
    "$HOME/sim_ws/src/particle_filter/maps/${1}.yaml"
    echo "Updated pf map yaml"
fi

# Add to raceline maps
RL_MAPS="$HOME/Raceline-Optimization/maps"
if [ -d "$RL_MAPS" ]; then
    cp "${MAP_PATH}_map.png" "$RL_MAPS/$1.png" 2>/dev/null || cp "${MAP_PATH}_map.pgm" "$RL_MAPS/$1.pgm"
    cp "${MAP_PATH}_map.yaml" "$RL_MAPS/$1.yaml"
    echo "Copied map to raceline"
fi

echo 'Map set'