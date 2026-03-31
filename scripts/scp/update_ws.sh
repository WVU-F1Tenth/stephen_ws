#!/usr/bin/env bash

# Recursively pushes to $HOME directory

# Assert: 
# $CAR_IP=<car ip addr>
# $CAR_USER=<car username>
# $1=<file to push>

# Ensure script is sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced:"
    echo "Usage: source update_ws"
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

# Update scripts and src
if ! scp -r \
    "$HOME/stephen_ws/src" \
    "${CAR_USER}@${CAR_IP}:/home/${CAR_USER}/stephen_ws/src"; then
    echo 'scp failed'
    return 1
fi

if ! scp -r \
    "$HOME/stephen_ws/scripts" \
    "${CAR_USER}@${CAR_IP}:/home/${CAR_USER}/stephen_ws/scripts"; then
    echo 'scp failed'
    return 1
fi

echo "ws updated"