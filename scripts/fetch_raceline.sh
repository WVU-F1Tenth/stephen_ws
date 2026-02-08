#!/usr/bin/env bash

# Ensure script is sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced:"
    echo "  source set_map.sh <map-name>"
    exit 1
fi

if [[ -z "$1" ]]; then
    echo "Usage: source fetch_raceline.sh <map-name>"
    return 1
fi

if [[ ! -n MAP_PATH ]]; then
    echo 'MAP_PATH not set'
    return 1
fi

RACELINE_DEST="${MAP_PATH}_raceline.csv"

RACELINE_SRC="$HOME/Raceline-Optimization/outputs/$1_map/*"
RACELINE_SRC=( $RACELINE_SRC )
if (( ${#RACELINE_SRC[@]} == 0 )); then
    echo 'ERROR: No raceline files found'
    return 1
elif (( ${#RACELINE_SRC[@]} > 1 )); then
    echo 'ERROR: Multiple racelines found'
    return 1
fi
RACELINE_SRC=${RACELINE_SRC[0]}

cp "$RACELINE_SRC" "$RACELINE_DEST"
