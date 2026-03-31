#!/usr/bin/env bash

# Ensure script is sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced:"
    echo "  source rename_ws.sh <name>"
    exit 1
fi

if [[ -z $1 ]]; then
    echo "  source rename_ws.sh <name>"
    return 1
fi

read "Are you sure you want to change script name usage to $1? y/n: " response
if [[ $response != 'y' ]]; then
    echo 'Cancelling...'
    return 1
fi

find scripts/ -type f -name '*.sh' -exec sed "s/stephen/$1/g" {} + || {
    echo "find failed"
    return 1
}

echo "Name usage changed to $1"
