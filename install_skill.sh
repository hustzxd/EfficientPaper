#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/hustzxd/EfficientPaper.git"
REPO_REF="main"
SKILL_NAME="efficientpaper-paper-research"
CODEX_ROOT="${CODEX_HOME:-$HOME/.codex}"
INSTALL_ROOT="${CODEX_ROOT}/skills"
INSTALL_DIR="${INSTALL_ROOT}/${SKILL_NAME}"
UPGRADE=0

usage() {
  cat <<'EOF'
Install the EfficientPaper Research Codex skill.

Options:
  --target DIR       Install into DIR instead of ~/.codex/skills
  --ref REF          Install from a branch or tag (default: main)
  --repo-url URL     Git repository URL
  --upgrade          Replace an existing installation with the downloaded version
  -h, --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      [[ $# -ge 2 ]] || { echo "--target requires a directory" >&2; exit 2; }
      INSTALL_ROOT="$2"
      INSTALL_DIR="${INSTALL_ROOT}/${SKILL_NAME}"
      shift 2
      ;;
    --ref)
      [[ $# -ge 2 ]] || { echo "--ref requires a branch or tag" >&2; exit 2; }
      REPO_REF="$2"
      shift 2
      ;;
    --repo-url)
      [[ $# -ge 2 ]] || { echo "--repo-url requires a URL" >&2; exit 2; }
      REPO_URL="$2"
      shift 2
      ;;
    --upgrade)
      UPGRADE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v git >/dev/null 2>&1; then
  echo "git is required to install this skill" >&2
  exit 1
fi

if [[ -e "$INSTALL_DIR" && "$UPGRADE" -ne 1 ]]; then
  echo "Skill already exists: $INSTALL_DIR" >&2
  echo "Remove that directory first if you want to reinstall it." >&2
  exit 1
fi

TEMP_ROOT="$(mktemp -d)"
cleanup() {
  rm -rf "$TEMP_ROOT"
}
trap cleanup EXIT

CHECKOUT_DIR="${TEMP_ROOT}/repo"
echo "Downloading ${SKILL_NAME} from ${REPO_URL} (${REPO_REF})..."
git clone --quiet --depth 1 --filter=blob:none --sparse --branch "$REPO_REF" "$REPO_URL" "$CHECKOUT_DIR"
git -C "$CHECKOUT_DIR" sparse-checkout set "skills/${SKILL_NAME}"

SOURCE_DIR="${CHECKOUT_DIR}/skills/${SKILL_NAME}"
if [[ ! -f "${SOURCE_DIR}/SKILL.md" ]]; then
  echo "Downloaded repository does not contain ${SOURCE_DIR}/SKILL.md" >&2
  exit 1
fi

mkdir -p "$INSTALL_ROOT"
if [[ "$UPGRADE" -eq 1 && -e "$INSTALL_DIR" ]]; then
  BACKUP_DIR="${TEMP_ROOT}/previous-skill"
  mv "$INSTALL_DIR" "$BACKUP_DIR"
  if ! cp -R "$SOURCE_DIR" "$INSTALL_DIR"; then
    rm -rf "$INSTALL_DIR"
    mv "$BACKUP_DIR" "$INSTALL_DIR"
    exit 1
  fi
  echo "Updated ${SKILL_NAME} at ${INSTALL_DIR}"
else
  cp -R "$SOURCE_DIR" "$INSTALL_DIR"
  echo "Installed ${SKILL_NAME} to ${INSTALL_DIR}"
fi
echo "Start a new Codex task to use the skill."
