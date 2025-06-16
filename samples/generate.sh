#!/usr/bin/env bash

path=$(realpath $(dirname "$0"))

for file in "$path"/*.b; do
    [ -f "$file" ] || continue

    midi_path="$path"/$(basename "$file" .b).mid
    cat "$file" | cargo run from-bf > "$midi_path" && timidity "$midi_path"
done
