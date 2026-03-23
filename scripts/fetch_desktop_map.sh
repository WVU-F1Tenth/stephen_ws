#!/usr/bin/env bash

# Fetches map from car. Car must have active ssh server.

# Assert: 
# DESKTOP=<desktop path>
# $1=MAP_NAME

# Ensure script is sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced:"
    echo "Usage: source set_map.sh <map-name>"
    exit 1
fi

if [[ -z "$DESKTOP" ]]; then
    echo 'DESKTOP not set'
    return 1
fi

if [[ -z "$1" ]]; then
    echo 'Usage: source fetch_race_map.sh <map name>'
    return 1
fi

if [[ ! -d "$HOME/stephen_ws/src/stephen/data/maps/${1}/" ]]; then
    echo "Error: $HOME/stephen_ws/src/stephen/data/maps/${1} doesn't exist"
    return 1
fi

if ! cp "$DESKTOP/${1}.png" "$HOME/stephen_ws/src/stephen/data/maps/${1}/${1}_map.png"; then
    echo 'Failed to copy map'
fi

echo 'png map fetched'
