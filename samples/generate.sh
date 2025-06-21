#!/usr/bin/env bash

path=$(realpath $(dirname "$0"))

for file in "$path"/*.b; do
    [ -f "$file" ] || continue

    midi_path="$path"/$(basename "$file" .b).mid
    [ -f "$midi_path" ] || cat "$file" | cargo run from-bf > "$midi_path"
done

for file in "$path"/*.mid; do
    [ -f "$file" ] || continue

    ogg_path="$path"/$(basename "$file" .mid).ogg
    [ -f "$ogg_path" ] || timidity "$file"
done
