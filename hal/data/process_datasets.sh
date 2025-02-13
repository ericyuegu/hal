#!/bin/bash
set -e

# Create output base directory
mkdir -p /tmp/hal/top_players

# Process each player directory
for player_dir in /opt/slippi/data/top_players/*/; do
    # Extract player name from directory path
    player_name=$(basename "$player_dir")

    # Create output directory for this player
    output_dir="/tmp/hal/top_players/$player_name"
    mkdir -p "$output_dir"

    echo "Processing replays for $player_name..."

    # Run process_replays.py for this player
    python hal/data/process_replays.py \
        --replay_dir "$player_dir" \
        --output_dir "$output_dir"
done

echo "Processing complete!"
