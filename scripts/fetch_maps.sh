#!/usr/bin/env bash

# Ensure script is sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced:"
    echo "Usage: source set_map.sh <map-name>"
    exit 1
fi

# Assert: 
# CAR_IP=<car ip addr>
# CAR_USER=<car username>
# 1=MAP_NAME

if [[ -z ]]


scp "${CAR_USER}@${CAR_IP}:/home/${CAR_USER}/sim_ws/maps' 