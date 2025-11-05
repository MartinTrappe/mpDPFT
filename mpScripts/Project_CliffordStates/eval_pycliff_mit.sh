# #!/usr/bin/env bash
# #
# # eval_pycliff_mit.sh
# #
# # Execution script for pycliff_mit.py int1 int2 int3 ... intD
# #                                     [with integers 1...D]
# #
# #
#!/usr/bin/env bash
#
# eval_pycliff_mit.sh
#
# Execution script for pycliff_mit.py int1 int2 int3 ... intD
#                                     [with integers 1...D]
#
#
# Author: Martin-Isbjörn Trappe
# Email: martin.trappe@quantumlah.org
# Date: 2025-09-16
#
# Usage:
#   bash eval_pycliff_mit.sh int1 int2 int3 ... intD
#                           [with integers 1...D]
#
# This script:
#   - Assumes that the environment is set up properly via run_pycliff_mit.sh
#   - Executes the Clifford Circuit Optimization script "pycliff_mit.py"
#   - Returns a single float, the Lanczos-minimized energy for D stabilizer states
#     that are identified by int1 int2 int3 ... intD
#




set -Eeuo pipefail

# Resolve paths
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
py="${script_dir}/pycliff_mit.py"

# Pick python: prefer local qc-env if present
python_bin="python3"
for d in "$script_dir" "$script_dir/.." "$script_dir/../.." "$script_dir/../../.."; do
    if [[ -x "$d/qc-env/bin/python3" ]]; then
        python_bin="$d/qc-env/bin/python3"
        # shellcheck disable=SC1091
        source "$d/qc-env/bin/activate"
        break
    fi
done

# Run once, capture all output, then emit ONLY the last numeric token
out="$("$python_bin" -u "$py" "$@" 2>&1 || true)"
num="$(printf '%s\n' "$out" | grep -Eo '[-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?' | tail -n 1)"

if [[ -n "${num:-}" ]]; then
    printf '%s\n' "$num"
else
    printf 'nan\n'
fi





# set -euo pipefail
#
# # Find the absolute path to the directory containing this script
# SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
#
# # Define paths relative to the script's directory
# VENV_PYTHON="$SCRIPT_DIR/qc-env/bin/python"
# PY_SCRIPT="$SCRIPT_DIR/pycliff_mit.py"
# STDERR_LOG="$SCRIPT_DIR/eval_pycliff_mit.stderr"
#
# # Check that the necessary files exist
# if [[ ! -x "$VENV_PYTHON" ]]; then
#   echo "Error: Python executable not found at $VENV_PYTHON. Please run 'bash run_pycliff_mit.sh' first." >&2
#   exit 1
# fi
#
# if [[ ! -f "$PY_SCRIPT" ]]; then
#   echo "Error: Python script not found at $PY_SCRIPT." >&2
#   exit 1
# fi
#
# # Suppress Qibo's informational banner
# export QIBO_LOG_LEVEL=4
#
# # Execute the python script using the virtual environment's python
# # Forward all command-line arguments ("$@")
# # Redirect any errors to a log file to keep stdout clean for the C++ program
# exec "$VENV_PYTHON" "$PY_SCRIPT" "$@" 2>>"$STDERR_LOG"


# # Author: Martin-Isbjörn Trappe
# # Email: martin.trappe@quantumlah.org
# # Date: 2025-09-16
# #
# # Usage:
# #   bash eval_pycliff_mit.sh int1 int2 int3 ... intD
# #                           [with integers 1...D]
# #
# # This script:
# #   - Assumes that the environment is set up properly via run_pycliff_mit.sh
# #   - Executes the Clifford Circuit Optimization script "pycliff_mit.py"
# #   - Returns a single float, the Lanczos-minimized energy for D stabilizer states
# #     that are identified by int1 int2 int3 ... intD
# #
#
# set -euo pipefail
# export LC_ALL=C
#
# export QIBO_LOG_LEVEL=4   # suppress INFO banner
#
# SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# PY="$SCRIPT_DIR/qc-env/bin/python"          # venv python next to this script
# PY_SCRIPT="$SCRIPT_DIR/pycliff_mit.py"      # script next to this script
#
# [[ -x "$PY" ]] || { echo "missing $PY" >&2; exit 1; }
# [[ -f "$PY_SCRIPT" ]] || { echo "missing $PY_SCRIPT" >&2; exit 1; }
#
# exec "$PY" "$PY_SCRIPT" "$@" 2>>"$SCRIPT_DIR/eval_pycliff_mit.stderr"
#
# # set -euo pipefail
# #
# # VENV=qc-env
# # PYTHON=python3
# #
# # # 1) Activate venv
# # source "$VENV/bin/activate"
# #
# # # 2) Misc
# # export QIBO_LOG_LEVEL=4   # suppress INFO banner
# #
# # # 3) Run GAO interface: forward all CLI integers; print only the last line
# # $PYTHON pycliff_mit.py "$@" | tail -n 1
# #
# # # 4) Deactivate
# # deactivate
#
#
# #To completely remove the virtual environment:
# #rm -rf qc-env
