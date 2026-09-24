#!/usr/bin/env bash
# Build looping GIF previews from media/videos/*.mp4 into media/gifs/.
# GitHub plays GIFs inline in a README; it does not play relative-path mp4 files.
# Two-pass palettegen/paletteuse: these are line plots on white, so a per-clip palette
# keeps them both sharp and small (~100-200 KB each).
set -euo pipefail
cd "$(dirname "$0")/.."
FPS="${FPS:-12}"
mkdir -p media/gifs
for f in media/videos/*.mp4; do
    n=$(basename "$f" .mp4)
    # two-panel clips are twice as wide, so give them more pixels
    w=520; [ "$(ffprobe -v error -select_streams v:0 -show_entries stream=width \
               -of csv=p=0 "$f" 2>/dev/null || echo 756)" -gt 1000 ] && w=760
    pal=$(mktemp --suffix=.png)
    ffmpeg -loglevel error -i "$f" \
        -vf "fps=$FPS,scale=$w:-1:flags=lanczos,palettegen=stats_mode=diff" -y "$pal"
    ffmpeg -loglevel error -i "$f" -i "$pal" \
        -lavfi "fps=$FPS,scale=$w:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5" \
        -y "media/gifs/$n.gif"
    rm -f "$pal"
    printf "%-34s %s\n" "media/gifs/$n.gif" "$(du -h "media/gifs/$n.gif" | cut -f1)"
done
