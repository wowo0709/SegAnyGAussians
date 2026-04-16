#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_path> [extra saga_gui args...]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="$1"
shift || true

DISPLAY_NUM="${DISPLAY_NUM:-:99}"
SCREEN_GEOM="${SCREEN_GEOM:-1920x1080x24}"
VNC_PORT="${VNC_PORT:-5901}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
RUNTIME_DIR="${RUNTIME_DIR:-/tmp/saga_gui_headless}"
mkdir -p "$RUNTIME_DIR"

XVFB_PID=""
FLUXBOX_PID=""
X11VNC_PID=""
WEBSOCKIFY_PID=""

cleanup() {
  local pids=("$WEBSOCKIFY_PID" "$X11VNC_PID" "$FLUXBOX_PID" "$XVFB_PID")
  for pid in "${pids[@]}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

start_or_reuse() {
  local __pid_var="$1"
  local service_name="$2"
  local match_pattern="$3"
  local log_path="$4"
  shift 4

  if pgrep -f "$match_pattern" >/dev/null 2>&1; then
    echo "Reusing existing $service_name"
    printf -v "$__pid_var" '%s' ""
    return 0
  fi

  "$@" >"$log_path" 2>&1 &
  local pid=$!
  printf -v "$__pid_var" '%s' "$pid"
  sleep 1

  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Failed to start $service_name. See $log_path" >&2
    if [[ -s "$log_path" ]]; then
      tail -n 40 "$log_path" >&2 || true
    fi
    return 1
  fi
}

export DISPLAY="$DISPLAY_NUM"

if ! xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
  Xvfb "$DISPLAY" -screen 0 "$SCREEN_GEOM" -ac +extension GLX +render -noreset >"$RUNTIME_DIR/xvfb.log" 2>&1 &
  XVFB_PID=$!
  sleep 2
  if ! xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
    echo "Failed to start Xvfb on $DISPLAY. See $RUNTIME_DIR/xvfb.log" >&2
    tail -n 40 "$RUNTIME_DIR/xvfb.log" >&2 || true
    exit 1
  fi
fi

start_or_reuse FLUXBOX_PID   "fluxbox"   "fluxbox -display $DISPLAY"   "$RUNTIME_DIR/fluxbox.log"   fluxbox -display "$DISPLAY"

start_or_reuse X11VNC_PID   "x11vnc"   "x11vnc -display $DISPLAY -rfbport $VNC_PORT"   "$RUNTIME_DIR/x11vnc.log"   x11vnc -display "$DISPLAY" -rfbport "$VNC_PORT" -localhost -forever -shared -nopw

start_or_reuse WEBSOCKIFY_PID   "websockify"   "websockify --web=/usr/share/novnc 127.0.0.1:$NOVNC_PORT localhost:$VNC_PORT"   "$RUNTIME_DIR/websockify.log"   websockify --web=/usr/share/novnc 127.0.0.1:"$NOVNC_PORT" localhost:"$VNC_PORT"

cat <<EOF
noVNC URL: http://127.0.0.1:${NOVNC_PORT}/vnc.html?host=127.0.0.1&port=${NOVNC_PORT}
If you are connected remotely, port-forward ${NOVNC_PORT} to your local machine.
Logs: ${RUNTIME_DIR}
EOF

source /opt/conda/etc/profile.d/conda.sh
conda activate saga
cd "$ROOT_DIR"
python saga_gui.py --model_path "$MODEL_PATH" "$@"
