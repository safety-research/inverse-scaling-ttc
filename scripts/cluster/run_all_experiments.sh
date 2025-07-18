#!/bin/bash

# Create logs directory for Hydra if it doesn't exist
mkdir -p .logs

DEFAULT_SESSION_NAME="experiment_session"
SESSION_NAME="$DEFAULT_SESSION_NAME"

# Check for session name argument
if [[ "$1" == "-s" || "$1" == "--session" ]]; then
    if [ -n "$2" ]; then # Check if a session name is actually provided after the flag
        SESSION_NAME="$2"
        shift 2 # Remove -s and its value from arguments
    else
        echo "Error: Session name flag $1 used but no session name provided."
        echo "Usage: $0 [-s|--session <session_name>] <experiment_name_1> ..."
        exit 1
    fi
fi

# Check if experiment names are provided as arguments
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 [-s|--session <session_name>] <experiment_name_1> <experiment_name_2> ... <experiment_name_n>"
    echo "Example: $0 -s my_runs claude4o_cautioned claude4o_natural_overthinking"
    echo "         $0 claude4o claude4s"
    exit 1
fi

COMMANDS=()
BASE_COMMAND="python run_experiment.py experiment="

# Build the full commands from the arguments
for EXP_NAME in "$@"; do
    COMMANDS+=("${BASE_COMMAND}${EXP_NAME}")
done

# Check if byobu is installed
if ! command -v byobu &> /dev/null
then
    echo "byobu could not be found. Please install byobu first."
    exit 1
fi

# Kill existing session with the same name, if any, to prevent errors
byobu kill-session -t "$SESSION_NAME" 2>/dev/null || true

echo "Starting byobu session '$SESSION_NAME'..."

# Start new byobu session with the first command
WINDOW_NAME="exp1"
echo "Launching: ${COMMANDS[0]} in window $WINDOW_NAME"
byobu new-session -d -s "$SESSION_NAME" -n "$WINDOW_NAME" "${COMMANDS[0]}"

# Add other commands in new windows
# Ensure there are more commands to run before starting the loop
if [ "${#COMMANDS[@]}" -gt 1 ]; then
    for i in $(seq 1 $((${#COMMANDS[@]} - 1))); do
        WINDOW_INDEX=$((i + 1))
        WINDOW_NAME="exp${WINDOW_INDEX}"
        echo "Launching: ${COMMANDS[$i]} in window $WINDOW_NAME"
        byobu new-window -t "$SESSION_NAME" -n "$WINDOW_NAME" "${COMMANDS[$i]}"
    done
fi

echo ""
echo "Byobu session '$SESSION_NAME' created with ${#COMMANDS[@]} windows."
echo "Each window is running an experiment script."
echo "To attach to the session, run: byobu attach-session -t $SESSION_NAME"