#!/usr/bin/env bash

# Fetches png map from DESKTOP

# Assert: 
# DESKTOP=<desktop path>
# MAP_PATH

# Ensure script is sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced:"
    echo 'Usage: source fetch_desktop_map.sh'
    exit 1
fi

if [[ -z "$1" ]]; then
    echo "Usage: source fetch_desktop_map.sh <map-name>"
    return 1
fi

if [[ -z "$DESKTOP" ]]; then
    echo 'DESKTOP not set'
    echo 'Usage: source fetch_desktop_map.sh'
    return 1
fi

MAP_PATH="$HOME/stephen_ws/src/stephen/data/maps/$1/$1"

if [[ ! -d "${MAP_PATH%/*}" ]]; then
    echo "Error: ${MAP_PATH%/*} doesn't exist"
    return 1
fi

cp "$DESKTOP/$1.png" "$MAP_PATH.png" || {
    echo "Failed to fetch desktop png map"
    return 1
}

echo 'Desktop png fetched'
