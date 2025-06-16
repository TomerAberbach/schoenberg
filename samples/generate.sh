#!/usr/bin/env bash

path=$(realpath $(dirname "$0"))

for file in "$path"/*.b; do
    [ -f "$file" ] || continue

    midi_path="$path"/$(basename "$file" .b).mid
    [ -f "$midi_path" ] || cat "$file" | cargo run from-bf > "$midi_path"

    ogg_path="$path"/$(basename "$file" .b).ogg
    [ -f "$ogg_path" ] || timidity "$midi_path"
done
