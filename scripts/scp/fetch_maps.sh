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

if [[ ! -d "$HOME/stephen_ws/src/data/maps/${1}" ]]; then
    mkdir "$HOME/stephen_ws/src/data/maps/${1}"
fi

echo 'Copying map from car...'

# Copy map from car
scp "${CAR_USER}@${CAR_IP}:/home/${CAR_USER}/sim_ws/maps/${1}_map.pgm" \
    "${CAR_USER}@${CAR_IP}:/home/${CAR_USER}/sim_ws/maps/${1}_map.yaml" \
    "$HOME/stephen_ws/src/data/maps/${1}/"
