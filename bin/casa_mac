#!/bin/zsh

# "casa"
# A wrapper around casa to separate resource directories and versions for each session.
# 
# Configuration:
# 1. macos change it to /bin/zsh, linux to zsh or bash(>4.0)
# 2. modify the "known CASA versions"
#
# History:
#   Original: RHEL 8 version, D. Petry (ESO) 
#   2022-01-06: Change the default configuration folder as the working directory, v4.2, Jianhang Chen
#   2022-06-08: Add general default configuration directory, v4.3, Jianhang Chen
# 
#   2025-12-27: support version as optionparameter and support macos, v5.0, Jianhang Chen
#

# -----------------------------
# Help message
# -----------------------------

echo '(casa.sh version 5.0 Dec 2025, jhchen)'

print_help() {
    cat <<EOF
Usage: casa [options] [-- <CASA arguments>]

Wrapper for CASA versions with optional pipeline mode.

Options:
  --list                 List available CASA versions
  -h, --help             Print this help message
  -s, --session ID       Specify session ID (optional)
  -v, --version VERSION  Specify CASA version (default: CASA_DEFAULT_VERSION or 6.6.1)
  --pipeline             Run CASA in pipeline mode (default: off)

All other arguments after '--' are passed directly to CASA.
EOF
}

set -e

DEFAULTCONFIG="$HOME/.casa"
CASADIR="$HOME/.local/casa/alma/configs"
CASA_DEFAULT_VERSION=6.6.1
PIPELINE=0  # Default: pipeline off

# -----------------------------
# Known CASA versions
# -----------------------------

CASADIR="/Users/jhchen/.local/casa/alma"
declare -A CASA_VERSIONS=(
    ["6.1.1"]="${CASADIR}/casa-6.1.1.15-pipeline-2020.1.0.40-10.15"
    ["6.2.1"]="${CASADIR}/casa-6.2.1.7-pipeline-2021.2.0.128-10.15-py36"
    ["6.4.1"]="${CASADIR}/casa-6.4.1.12-pipeline-2022.2.0.68-11.0-py36"
    ["6.5.4"]="${CASADIR}/casa-6.5.4.9-pipeline-2023.1.0.124-11.0-py38"
    ["6.6.1"]="${CASADIR}/casa-6.6.1.17-pipeline-2024.1.0.8-12.0-py38"
)

# for linux
#declare -A CASA_VERSIONS=(
#    ["5.6.1"]="${CASADIR}/casa-pipeline-release-5.6.1-8"
#    ["5.6.2"]="${CASADIR}/casa-pipeline-release-5.6.2-3"
#    ["5.7.0"]="${CASADIR}/casa-release-5.7.0-134"
#    ["6.2.1"]="${CASADIR}/casa-6.2.1-7-pipeline-2021.2.0.128"
#  )


# -----------------------------
# OS / architecture detection
# -----------------------------
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Darwin) PLATFORM="macos" ;;
    Linux)  PLATFORM="linux" ;;
    *)
        echo "ERROR: Unsupported OS '$OS'"
        exit 1
        ;;
esac

case "$ARCH" in
    x86_64|amd64)  ARCH="x86_64" ;;
    arm64|aarch64) ARCH="arm64" ;;
    *)
        echo "ERROR: Unsupported architecture '$ARCH'"
        exit 1
        ;;
esac


# -----------------------------
# --help
# -----------------------------
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    print_help
    exit 0
fi

# -----------------------------
# --list-versions
# -----------------------------
#if [[ "$1" == "--list" ]]; then
#    echo "Available CASA versions:"
#    for v in "${!CASA_VERSIONS[@]}"; do
#        echo "  $v"
#    done | sort
#    exit 0
#fi

if [[ "$1" == "--list" ]]; then
    echo "Available CASA versions:"
    if [[ -n "${ZSH_VERSION:-}" ]]; then
        for v in ${(ok)CASA_VERSIONS}; do
            echo "  $v"
        done
    else
        for v in "${!CASA_VERSIONS[@]}"; do
            echo "  $v"
        done
    fi | sort
    exit 0
fi


# -----------------------------
# Wrapper-only options
# -----------------------------
SESSION_OVERRIDE=""
CASA_VERSION=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--session)
            SESSION_OVERRIDE="$2"
            shift 2
            ;;
        -v|--version)
            CASA_VERSION="$2"
            shift 2
            ;;
        --pipeline)
            PIPELINE=1
            shift
            ;;
        *)
            break
            ;;
    esac
done

# -----------------------------
# CASA binary resolver
# -----------------------------
resolve_casa_bin() {
    local base="$1"
    if [[ "$PLATFORM" == "macos" ]]; then
        echo "${base}.app/Contents/MacOS/casa"
    else
        echo "${base}/bin/casa"
    fi
}

CASA_BASE="${CASA_VERSIONS[$CASA_VERSION]:-${CASA_VERSIONS[$CASA_DEFAULT_VERSION]}}"
if [[ -z "$CASA_BASE" ]]; then
    echo "ERROR: Unsupported CASA version '$EFFECTIVE_VERSION'"
    exit 1
fi

CASA_BIN="$(resolve_casa_bin "$CASA_BASE")"

if [[ ! -x "$CASA_BIN" ]]; then
    echo "ERROR: CASA binary not found or not executable:"
    echo "  $CASA_BIN"
    exit 1
fi

# -----------------------------
# Determine session ID
# -----------------------------
if [[ -n "$SESSION_OVERRIDE" ]]; then
    SESSION_ID="$SESSION_OVERRIDE"
elif [[ -n "$CASA_SESSION_ID" ]]; then
    SESSION_ID="$CASA_SESSION_ID"
else
    SESSION_ID="$(whoami)"
fi

RCFILENAME="${PWD}/.${SESSION_ID}.casa"
if [[ ! -e "$RCFILENAME" ]]; then
    echo "Creating CASA resource directory: $RCFILENAME"
    mkdir -p "$RCFILENAME"
    if [[ -d "$DEFAULTCONFIG" ]]; then
        cp -r "${DEFAULTCONFIG}/"* "$RCFILENAME/"
    fi
else
    echo "Using CASA resource directory: $RCFILENAME"
fi

# -----------------------------
# Execute CASA
# -----------------------------
if [[ -n "$PIPELINE" ]]; then
    exec "$CASA_BIN" --pipeline --rcdir "$RCFILENAME" "$@"
else
    exec "$CASA_BIN" --rcdir "$RCFILENAME" "$@"
fi

