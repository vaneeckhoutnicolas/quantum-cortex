#!/bin/bash
# Join the scene files in order with the concat demuxer, no re-encoding.
set -e
cd "$1"
python3 -c "
import json; sc=json.load(open('scenes.json'))
open('concat.txt','w').write(''.join(\"file '%s.mp4'\n\" % s['id'] for s in sc))"
ffmpeg -y -v error -f concat -safe 0 -i concat.txt -c copy "$2"
ffprobe -v error -show_entries format=duration,size -of default=nw=1 "$2"
