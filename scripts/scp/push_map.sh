#!/usr/bin/env bash

# Pushes map to car. Car must have active ssh server.

# Assert: 
# $CAR_IP=<car ip addr>
# $CAR_USER=<car username>
# $1=MAP_NAME

# Ensure script is sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced:"
    echo "Usage: source set_map.sh <map-name>"
    exit 1
fi

if [[ -z "$CAR_IP" ]]; then
    echo 'CAR_IP not set'
    return 1
fi
if [[ -z "$CAR_USER" ]]; then
    echo 'CAR_USER not set'
    return 1
fi
if [[ -z "$1" ]]; then
    echo 'Usage: source set_map.sh <map-name>'
    return 1
fi

if [[ ! -d "$HOME/stephen_ws/src/stephen/data/maps/${1}/" ]]; then
    echo "${1} directory not found in maps"
    return 1
fi

# Try to fetch raceline if it's not in data/maps
if [[ ! -f "$HOME/stephen_ws/src/stephen/data/maps/${1}/${1}_raceline.csv" ]]; then
    source "$HOME/stephen_ws/scripts/fetch_raceline.sh" "${1}" || return 1
fi

echo 'Pushing map to car...'

# Push map to car
if ! scp -r \
    "$HOME/stephen_ws/src/data/maps/${1}" \
    "${CAR_USER}@${CAR_IP}:/home/${CAR_USER}/stephen_ws/src/stephen/data/maps/"; then
    echo 'scp failed'
    return 1
fi

echo "$1 pushed to maps"