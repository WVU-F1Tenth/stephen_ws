#!/usr/bin/env bash

# Fetches map from car. Car must have active ssh server.

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

if [[ ! -d "$HOME/stephen_ws/src/stephen/data/maps/${1}" ]]; then
    mkdir "$HOME/stephen_ws/src/stephen/data/maps/${1}"
fi

echo 'Fetching map from car...'

# Copy map from car
if ! scp \
    "${CAR_USER}@${CAR_IP}:/home/${CAR_USER}/sim_ws/maps/${1}.pgm" \
    "${CAR_USER}@${CAR_IP}:/home/${CAR_USER}/sim_ws/maps/${1}.yaml" \
    "$HOME/stephen_ws/src/stephen/data/maps/${1}/"; then
    echo 'scp failed'
    return 1
fi

echo "$1 map fetched"
