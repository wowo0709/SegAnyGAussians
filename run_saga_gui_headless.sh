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

export DISPLAY="$DISPLAY_NUM"

if ! xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
  Xvfb "$DISPLAY" -screen 0 "$SCREEN_GEOM" -ac +extension GLX +render -noreset >"$RUNTIME_DIR/xvfb.log" 2>&1 &
  XVFB_PID=$!
  sleep 2
fi

fluxbox -display "$DISPLAY" >"$RUNTIME_DIR/fluxbox.log" 2>&1 &
FLUXBOX_PID=$!
sleep 1

x11vnc -display "$DISPLAY" -rfbport "$VNC_PORT" -localhost -forever -shared -nopw >"$RUNTIME_DIR/x11vnc.log" 2>&1 &
X11VNC_PID=$!
sleep 1

websockify --web=/usr/share/novnc 127.0.0.1:"$NOVNC_PORT" localhost:"$VNC_PORT" >"$RUNTIME_DIR/websockify.log" 2>&1 &
WEBSOCKIFY_PID=$!
sleep 1

cat <<EOF
noVNC URL: http://127.0.0.1:${NOVNC_PORT}/vnc.html?host=127.0.0.1&port=${NOVNC_PORT}
If you are connected remotely, port-forward ${NOVNC_PORT} to your local machine.
Logs: ${RUNTIME_DIR}
EOF

source /opt/conda/etc/profile.d/conda.sh
conda activate saga
cd "$ROOT_DIR"
python saga_gui.py --model_path "$MODEL_PATH" "$@"
