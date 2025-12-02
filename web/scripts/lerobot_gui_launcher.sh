#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="/home/jannick/PycharmProjects/lerobot-hil-serl/web/scripts"
REPO_ROOT="/home/jannick/PycharmProjects/lerobot-hil-serl"
LOG_DIR="$HOME/.local/share/lerobot"
LOG_FILE="$LOG_DIR/gui-launch.log"

mkdir -p "$LOG_DIR"

log() {
    printf '%s %s\n' "[$(date '+%Y-%m-%d %H:%M:%S')]" "$*" | tee -a "$LOG_FILE"
}

keep_terminal_open() {
    echo "" | tee -a "$LOG_FILE"
    echo "Press Enter to close this window..." | tee -a "$LOG_FILE"
    # shellcheck disable=SC2162
    read _
}

cd "$SCRIPT_DIR"

log "[LeRobot] Desktop launcher started"
log "Script dir: $SCRIPT_DIR"
log "Repo root:  $REPO_ROOT"
log "Log file:   $LOG_FILE"

# Helpers
open_url() {
    local url="$1"
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$url" >/dev/null 2>&1 || true
    elif [[ -n "$PYTHON_EXE" ]]; then
        "$PYTHON_EXE" - <<'PY'
import webbrowser
webbrowser.open('"$1"')
PY
    fi
}

notify() {
    if command -v notify-send >/dev/null 2>&1; then
        notify-send "LeRobot GUI" "$1"
    fi
}

is_port_in_use() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltn 2>/dev/null | awk '{print $4}' | grep -E "[:.]${port}$" -q
    elif command -v lsof >/dev/null 2>&1; then
        lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
    elif command -v netstat >/dev/null 2>&1; then
        netstat -tln 2>/dev/null | awk '{print $4}' | grep -E "[:.]${port}$" -q
    else
        return 1
    fi
}

# If backend already running, just open the browser to it
if is_port_in_use 8000; then
    log "Detected existing service on port 8000. Opening browser..."
    notify "LeRobot GUI is already running. Opening in your browser."
    open_url "http://localhost:8000"
    exit 0
fi

# Try to find a Python executable that has lerobot installed
PYTHON_EXE=""

# 1) Prefer conda 'lerobotHilSerl' env by reading environments.txt (works without interactive shells)
if [[ -z "${PYTHON_EXE}" ]] && [[ -f "$HOME/.conda/environments.txt" ]]; then
    if ENV_PATH=$(grep -E "/envs/lerobotHilSerl$" "$HOME/.conda/environments.txt" | tail -n1); then
        if [[ -n "$ENV_PATH" && -x "$ENV_PATH/bin/python" ]]; then
            PYTHON_EXE="$ENV_PATH/bin/python"
            log "Found conda env via environments.txt: $ENV_PATH"
        fi
    fi
fi

# 2) Common default locations for conda/miniconda - check lerobotHilSerl first
for BASE in "$HOME/miniconda3" "$HOME/mambaforge" "$HOME/anaconda3"; do
    if [[ -z "${PYTHON_EXE}" && -x "$BASE/envs/lerobotHilSerl/bin/python" ]]; then
        PYTHON_EXE="$BASE/envs/lerobotHilSerl/bin/python"
        log "Found conda env at: $BASE/envs/lerobotHilSerl"
        break
    fi
done

# 3) If conda command is available, try to resolve base and env path
if [[ -z "${PYTHON_EXE}" ]] && command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    if conda env list | grep -q "^[^#].*\blerobotHilSerl\b"; then
        # Try to get the absolute path of the env
        ENV_DIR=$(conda env list | awk '/\blerobotHilSerl\b/ {print $NF}' | tail -n1)
        if [[ -n "$ENV_DIR" && -x "$ENV_DIR/bin/python" ]]; then
            PYTHON_EXE="$ENV_DIR/bin/python"
            log "Found conda env via conda env list: $ENV_DIR"
        else
            # Fallback: activate and rely on PATH
            conda activate lerobot || true
        fi
    fi
fi

# 4) If still not set, try the current PATH's python
if [[ -z "${PYTHON_EXE}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_EXE="$(command -v python)"
        log "Using python from PATH: $PYTHON_EXE"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_EXE="$(command -v python3)"
        log "Using python3 from PATH: $PYTHON_EXE"
    fi
fi

log "Launching LeRobot GUI..."
{
    # Prefer production launcher (works without Node if dist exists)
    if [[ -n "$PYTHON_EXE" && -f "$SCRIPT_DIR/start_gui.py" ]]; then
        log "Running: $PYTHON_EXE start_gui.py"
        "$PYTHON_EXE" "$SCRIPT_DIR/start_gui.py" 2>&1 | tee -a "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
    elif command -v lerobot-gui >/dev/null 2>&1; then
        log "Running fallback: lerobot-gui"
        lerobot-gui 2>&1 | tee -a "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
    elif [[ -n "$PYTHON_EXE" ]]; then
        log "Running fallback: $PYTHON_EXE -m web.scripts.cli gui"
        "$PYTHON_EXE" -m web.scripts.cli gui 2>&1 | tee -a "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
    else
        log "ERROR: No suitable Python interpreter found."
        EXIT_CODE=127
    fi

    if [[ $EXIT_CODE -ne 0 ]]; then
        log "Launcher exited with code $EXIT_CODE"
        notify "LeRobot GUI failed to start (code $EXIT_CODE). See $LOG_FILE"
        keep_terminal_open
        exit $EXIT_CODE
    fi
} || {
    CODE=$?
    log "Unexpected launcher failure with code $CODE"
    notify "LeRobot GUI crashed (code $CODE). See $LOG_FILE"
    keep_terminal_open
    exit $CODE
}

exit 0
