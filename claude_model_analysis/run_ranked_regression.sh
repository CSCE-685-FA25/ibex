#!/usr/bin/env bash
# Run only the ranked/pruned tests by iterating through test-id,seed pairs.

set -euo pipefail
trap 'echo "[run_ranked_regression] Error on line $LINENO" >&2' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CORE_IBEX_DIR="${REPO_ROOT}/ibex/dv/uvm/core_ibex"

LIST_FILE="${SCRIPT_DIR}/tuned_results/pruned_tests.txt"
LIMIT=0
DRY_RUN=0
OUT_DIR=""
PRESERVE_METADATA=1
RESUME_FILE=""
RESUME_ENABLED=1
RESET_RESUME=0

resolve_out_path() {
  local raw="$1"
  if [[ -z "$raw" ]]; then
    echo "${CORE_IBEX_DIR}/out"
  elif [[ "$raw" = /* ]]; then
    echo "$raw"
  else
    echo "${CORE_IBEX_DIR}/${raw}"
  fi
}

usage() {
  cat <<'EOF'
Usage: run_ranked_regression.sh [--list FILE] [--limit N] [--dry-run] [--out-dir DIR]

Options:
  --list FILE   Path to pruned test list (default: tuned_results/pruned_tests.txt)
  --limit N     Only run the first N entries from the list
  --dry-run     Print the make commands without executing them
  --out-dir DIR Use a dedicated ibex/dv out directory (metadata + logs)
  --discard-metadata  Remove metadata between runs instead of archiving it
  --preserve-metadata Archive metadata between runs (default)
  --resume-log FILE   Path to the resume log (default: OUT/ranked_completed.log)
  --no-resume         Disable resume/skip behavior for this invocation
  --resume            Re-enable resume behavior if previously disabled
  --reset-resume      Delete the resume log before running

Environment knobs:
  SIMULATOR     RTL simulator passed to make (default: xlm)
  COV           Set to 1 to enable coverage (default: 1)
  WAVES         Set to 1 to turn on waveform dumps (default: 0)
  MAKE_EXTRA_ARGS Additional arguments appended to the make invocation
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)
      LIST_FILE="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --discard-metadata)
      PRESERVE_METADATA=0
      shift
      ;;
    --preserve-metadata)
      PRESERVE_METADATA=1
      shift
      ;;
    --resume-log)
      RESUME_FILE="$2"
      shift 2
      ;;
    --no-resume)
      RESUME_ENABLED=0
      shift
      ;;
    --resume)
      RESUME_ENABLED=1
      shift
      ;;
    --reset-resume)
      RESET_RESUME=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$LIST_FILE" ]]; then
  echo "List file not found: $LIST_FILE" >&2
  exit 1
fi

source "${REPO_ROOT}/env_setup.bash"
SIMULATOR="${SIMULATOR:-xlm}"
COV="${COV:-1}"
WAVES="${WAVES:-0}"
EXTRA_ARGS=()
if [[ -n "${MAKE_EXTRA_ARGS:-}" ]]; then
  IFS=' ' read -r -a EXTRA_ARGS <<< "${MAKE_EXTRA_ARGS}"
fi

if [[ -z "$OUT_DIR" && -n "${OUT:-}" ]]; then
  OUT_DIR="$OUT"
fi

OUT_PATH="$(resolve_out_path "$OUT_DIR")"
declare -A COMPLETED_TESTS=()

if [[ -z "$RESUME_FILE" ]]; then
  RESUME_FILE="${OUT_PATH%/}/ranked_completed.log"
elif [[ "$RESUME_FILE" != /* ]]; then
  RESUME_FILE="${SCRIPT_DIR}/${RESUME_FILE}"
fi

SKIPPED=0

if [[ $RESUME_ENABLED -eq 1 ]]; then
  RESUME_DIR="$(dirname "$RESUME_FILE")"
  mkdir -p "$RESUME_DIR"
  if [[ $RESET_RESUME -eq 1 && -f "$RESUME_FILE" ]]; then
    rm -f "$RESUME_FILE"
  fi
  if [[ -f "$RESUME_FILE" ]]; then
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      COMPLETED_TESTS["$line"]=1
    done < "$RESUME_FILE"
  fi
  touch "$RESUME_FILE"
  echo "Resume log: $RESUME_FILE (loaded ${#COMPLETED_TESTS[@]} entries)"
else
  RESUME_FILE=""
fi

TOTAL=$(awk 'NF>0{c++} END{print c+0}' "$LIST_FILE")
echo "Using test list: $LIST_FILE ($TOTAL entries)"
COUNT=0
RUNS=0

while IFS=, read -r TEST_ID PROB; do
  [[ -z "$TEST_ID" ]] && continue
  ((++COUNT))
  if [[ $LIMIT -gt 0 && $COUNT -gt $LIMIT ]]; then
    break
  fi
  BASE_TEST="${TEST_ID%.*}"
  SEED="${TEST_ID##*.}"
  TEST_KEY="${BASE_TEST}.${SEED}"
  ENTRY_DESC="[$COUNT/$TOTAL] TEST=${BASE_TEST} SEED=${SEED} (score=${PROB})"

  if [[ $RESUME_ENABLED -eq 1 && -n "${COMPLETED_TESTS[$TEST_KEY]:-}" ]]; then
    echo "${ENTRY_DESC} - skipped (already completed)"
    ((SKIPPED++))
    continue
  fi

  CMD=(make -C "${CORE_IBEX_DIR}" run
       SIMULATOR="$SIMULATOR"
       COV="$COV"
       WAVES="$WAVES"
       TEST="$BASE_TEST"
       ITERATIONS=1
       SEED="$SEED"
       OUT="$OUT_PATH")
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi
  echo "$ENTRY_DESC"
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '  %q' "${CMD[@]}"
    printf '\n'
  else
    METADATA_DIR="${OUT_PATH%/}/metadata"
    if [[ -d "$METADATA_DIR" ]]; then
      if [[ $PRESERVE_METADATA -eq 1 ]]; then
        ARCHIVE_ROOT="${OUT_PATH%/}/metadata_archive"
        mkdir -p "$ARCHIVE_ROOT"
        STAMP=$(date +%Y%m%d_%H%M%S)
        ARCHIVE_PATH="${ARCHIVE_ROOT}/${BASE_TEST}.${SEED}_${STAMP}"
        mv "$METADATA_DIR" "$ARCHIVE_PATH"
        echo "Archived previous metadata to $ARCHIVE_PATH"
      else
        rm -rf -- "$METADATA_DIR"
      fi
    fi
    "${CMD[@]}"
    if [[ $RESUME_ENABLED -eq 1 ]]; then
      echo "$TEST_KEY" >> "$RESUME_FILE"
      COMPLETED_TESTS["$TEST_KEY"]=1
    fi
  fi
  ((RUNS++))
  echo
done < "$LIST_FILE" || true

SUMMARY="Scheduled $RUNS test runs from $LIST_FILE"
if [[ $RESUME_ENABLED -eq 1 ]]; then
  echo "$SUMMARY (${SKIPPED} skipped via $RESUME_FILE)"
else
  echo "$SUMMARY"
fi
