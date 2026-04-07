#!/usr/bin/env bash

# Converts map to png and pushes to windows desktop.

# Assert: 
# DESKTOP=<desktop path>

# Ensure script is sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced:"
    echo "Usage: source push_desktop_map.sh"
    exit 1
fi

if [[ -z "$1" ]]; then
    echo "Usage: source push_desktop_map.sh <map-name>"
    return 1
fi

if [[ -z "$DESKTOP" ]]; then
    echo 'DESKTOP not set'
    echo "Usage: source push_desktop_map.sh"
    return 1
fi

MAP_PATH="$HOME/stephen_ws/src/stephen/data/maps/$1/$1"

if [[ ! -d "${MAP_PATH%/*}" ]]; then
    echo "Error: ${MAP_PATH%/*} doesn't exist"
    return 1
fi

if [[ -f "$MAP_PATH.png" ]]; then
    cp "$MAP_PATH.png" "$DESKTOP/"
elif [[ -f "$MAP_PATH.pgm" ]]; then
    convert "$MAP_PATH.pgm" "$MAP_PATH.png"
    cp "$MAP_PATH.png" "$DESKTOP/"
else
    echo 'No map image found'
    return 1
fi

echo 'Map pushed to desktop'
