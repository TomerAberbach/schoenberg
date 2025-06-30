#!/usr/bin/env bash

path=$(realpath $(dirname "$0"))

for file in "$path"/*.b; do
    [ -f "$file" ] || continue

    midi_path="$path"/$(basename "$file" .b).mid
    [ -f "$midi_path" ] || cat "$file" | cargo run from-bf > "$midi_path"
done

for file in "$path"/*.mid; do
    [ -f "$file" ] || continue

    mp3_path="$path"/$(basename "$file" .mid).mp3
    [ -f "$mp3_path" ] || timidity -Ow -o - "$file" | ffmpeg -i - -vn -acodec libmp3lame -ab 192k "$mp3_path"
done
