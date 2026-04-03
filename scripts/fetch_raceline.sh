#!/usr/bin/env bash

# Ensure script is sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced:"
    echo "Usage: source fetch_raceline.sh"
    exit 1
fi

if [[ -z $MAP_PATH ]]; then
    echo 'MAP_PATH not set'
    return 1
fi

RACELINE_DEST="${MAP_PATH}_raceline.csv"

RACELINE_SRC="$HOME/Raceline-Optimization/outputs/${MAP_PATH##*/}/*"
RACELINE_SRC=( $RACELINE_SRC )
if (( ${#RACELINE_SRC[@]} == 0 )); then
    echo 'ERROR: No raceline files found'
    return 1
elif (( ${#RACELINE_SRC[@]} > 1 )); then
    echo 'ERROR: Multiple racelines found'
    return 1
fi
RACELINE_SRC=${RACELINE_SRC[0]}

cp "$RACELINE_SRC" "$RACELINE_DEST" || {
    echo "Error: failed to fetch raceline"
    return 1
    }

echo "Raceline fetched"
