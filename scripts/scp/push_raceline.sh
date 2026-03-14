#!/usr/bin/env bash

# Ensure script is sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced:"
    echo "Usage: source set_map.sh <map-name>"
    exit 1
fi

# Assert: 
# $CAR_IP=<car ip addr>
# $CAR_USER=<car username>
# $1=MAP_NAME

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

if [[ ! -d "$HOME/stephen_ws/src/data/maps/${1}/" ]]; then
    echo "${1} directory not found in maps"
    return 1
fi

if [[ ! -f "$HOME/stephen_ws/src/data/maps/${1}/${1}_raceline.csv" ]]; then
    source "$HOME/stephen_ws/scripts/fetch_raceline.sh" "${1}" || return 1
fi

echo 'Pushing map to car...'

# Push map to car
scp -r "$HOME/stephen_ws/src/data/maps/${1}" \
    "${CAR_USER}@${CAR_IP}:/home/${CAR_USER}/stephen_ws/src/data/maps/"
