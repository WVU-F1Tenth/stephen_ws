#!/usr/bin/env bash

# Recursively pushes to $HOME directory

# Assert: 
# $CAR_IP=<car ip addr>
# $CAR_USER=<car username>
# $1=<file to push>

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
    echo 'Usage: source push_to_home.sh <file-name>'
    return 1
fi

read -p "Are you sure you want to push $1 y/n: " response
if [[ $response != 'y' ]]; then
    echo 'Cancelling...'
    return 1
fi

echo 'Pushing to car...'

# Push map to car
if ! scp -r \
    "$1" \
    "${CAR_USER}@${CAR_IP}:/home/${CAR_USER}/"; then
    echo 'scp failed'
    return 1
fi

echo "$1 pushed to \$HOME"