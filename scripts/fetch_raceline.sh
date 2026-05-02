#!/usr/bin/env bash

# Ensure script is sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced:"
    echo "Usage: source fetch_raceline.sh"
    exit 1
fi

if [[ -z "$1" ]]; then
    echo "Usage: source fetch_raceline.sh <map-name>"
    return 1
fi

if [[ -z "$HOME/stephen_ws/src/stephen/data/maps/$1/$1" ]]; then
    echo "$1 map doesn't exist"
    return 1
fi

CENTERLINE_DEST="$HOME/stephen_ws/src/stephen/data/maps/$1/${1}_centerline.csv"
CENTERLINE_SRC="$HOME/raceline-optimization/inputs/tracks/$1.csv"

{
    cp "$CENTERLINE_SRC" "$CENTERLINE_DEST"
    echo 'Centerline fetched'
} || {
    echo "Error: Failed to fetch centerline"
    return 1
}

RACELINE_DEST="$HOME/stephen_ws/src/stephen/data/maps/$1/${1}_raceline.csv"
RACELINE_SRC="$HOME/raceline-optimization/outputs/$1/${1}_raceline.csv"

if [[ ! -f "$RACELINE_SRC" ]]; then
    echo "$1 raceline not found"
    return 1
fi

cp "$RACELINE_SRC" "$RACELINE_DEST" || {
    echo "Error: Failed to fetch raceline"
    return 1
    }

echo "Raceline fetched"
