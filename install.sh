#!/usr/bin/env bash

set -euo pipefail

# ─────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────

# --- Required NVIDIA driver ---
# Isaac Sim 5.1.0 requires an NVIDIA driver between 550.x and 580.x inclusive.
# Driver >= 590.x (e.g. 595) causes a crash in librtx.scenedb.plugin.so during
# Hydra/RTX engine initialisation (known incompatibility).
if nvidia-smi &>/dev/null; then
    driver_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    if [[ -n "$driver_ver" ]]; then
        major="${driver_ver%%.*}"
        if [[ "$major" -ge 590 ]]; then
            cat >&2 <<INCOMPAT

========================================================================
WARNING: NVIDIA driver ${driver_ver} detected.
Isaac Sim 5.1.0 is incompatible with driver series >= 590.

Working versions: 550, 560, 570, 580 series
        (tested: 550.163.01, 580.159.03)

To downgrade on Ubuntu 24.04:
  sudo apt-get install --reinstall nvidia-driver-550-open
  # Reboot after install.

On newer kernels (6.17+) the 550/570 meta-packages pull in the 580 kernel
module automatically – that is fine; only the userspace libraries matter.
========================================================================

INCOMPAT
        fi
    fi
else
    cat >&2 <<NODRIVER

========================================================================
ERROR: NVIDIA driver is not running.
Isaac Sim requires an NVIDIA GPU with a proprietary driver (550-580 series).

  Check:  nvidia-smi
  Install on Ubuntu 24.04:
    sudo apt-get install nvidia-driver-570-open
    # Reboot and (if Secure Boot is on) enrol the MOK key at the blue
    # EFI prompt that appears after reboot.
========================================================================

NODRIVER
    exit 1
fi

# --- Python 3.11 ---
if ! uv python find 3.11 &>/dev/null; then
    echo "[install.sh] Installing Python 3.11 via uv …"
    uv python install 3.11
fi

# ─────────────────────────────────────────────────────────────
# Isaac Lab
# ─────────────────────────────────────────────────────────────
# Isaac Lab v2.3.0 is committed directly into this repo at IsaacLab/ — it is
# not a submodule (it used to be one, under Training/).  Isaac Sim 5.1 resolves
# cleanly against v2.3.0; newer Isaac Lab refs pull in a Starlette pin that
# conflicts with Isaac Sim's FastAPI stack, so the ref is pinned by shipping
# the tree.  pyproject.toml installs IsaacLab/source/* as editable path deps,
# so a plain `git clone` is all that is needed.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
isaaclab_dir="${script_dir}/IsaacLab"

for pkg in isaaclab isaaclab_assets isaaclab_rl isaaclab_tasks; do
    if [[ ! -d "${isaaclab_dir}/source/${pkg}" ]]; then
        cat >&2 <<MISSING

========================================================================
ERROR: ${isaaclab_dir}/source/${pkg} is missing.

Isaac Lab ships inside this repository, so a plain clone is enough:
  git clone git@github.com:MCiuzelis/LituanicaX_IsaacSimChallenge.git

If IsaacLab/ is missing or empty, the clone did not complete.  Re-clone
into a new, empty directory — cloning on top of an existing directory
fails with "already exists and is not an empty directory" and leaves a
half-populated tree behind.
========================================================================

MISSING
        exit 1
    fi
done

# ─────────────────────────────────────────────────────────────
# Python dependencies
# ─────────────────────────────────────────────────────────────
cd "${script_dir}"
uv venv --python 3.11
uv sync

# Isaac Sim pulls in opencv-python-headless (no imshow support).
# Force-replace it with the GUI-capable build; GTK3 is already present on
# Ubuntu 24.04.  We pin numpy to 1.x because Isaac Sim's compiled extensions
# ship with 1.26 and the ABI isn't guaranteed across the 1→2 boundary.
uv pip install --force-reinstall "opencv-python" "numpy<2.0.0"

# setuptools 82 removed pkg_resources which TensorBoard 2.x still imports.
# Pin to the last version that ships it (matches the build-system constraint).
uv pip install "setuptools<82.0.0"

# ─────────────────────────────────────────────────────────────
# Helper scripts in ~/.local/bin
# ─────────────────────────────────────────────────────────────
# On dual-GPU systems where displays are driven by the iGPU, Vulkan defaults
# to the Intel device.  Setting __NV_PRIME_RENDER_OFFLOAD=1 forces Vulkan to
# the NVIDIA GPU for Omniverse Kit.
local_bin="${HOME}/.local/bin"
mkdir -p "${local_bin}"
PRIME_WRAPPER='if nvidia-smi -L 2>/dev/null | grep -qi intel; then export __NV_PRIME_RENDER_OFFLOAD=1; fi'

# train and play are part of the team's half of the repository, so they live in
# teamcode/ next to the environment and the PPO config they drive.
for name in train play; do
    cat > "${local_bin}/${name}" <<SCRIPT
#!/bin/bash
PROJECT_DIR="${script_dir}"
${PRIME_WRAPPER}
cd "\$PROJECT_DIR"
exec uv run python -m teamcode.${name} "\$@"
SCRIPT
    chmod +x "${local_bin}/${name}"
done

# The official benchmark lives in the SDK, not at the project root — it is the
# scoring rules, not a project script.
cat > "${local_bin}/benchmark" <<SCRIPT
#!/bin/bash
PROJECT_DIR="${script_dir}"
${PRIME_WRAPPER}
cd "\$PROJECT_DIR"
exec uv run python -m lituanicax_sdk.benchmark "\$@"
SCRIPT
chmod +x "${local_bin}/benchmark"

# Before the leaderboard existed the scorer was called `evaluate`. A stale copy
# would still be first on someone's PATH and would run a module that is gone.
rm -f "${local_bin}/evaluate"

echo ""
echo "=============================="
echo "  Setup complete!"
echo "  - Virtual env:  .venv/"
echo "  - Isaac Lab:    IsaacLab/ @ v2.3.0"
echo "  - Isaac Sim:    5.1.0 (pip: nvidia index)"
echo ""
echo "  Commands:"
echo "    train --num_envs 200 --headless   # train a policy"
echo "    play  --num_envs 1                # watch the newest checkpoint"
echo "    benchmark                         # official score: best of ten laps"
echo ""
echo "  Edit teamcode/ ; lituanicax_sdk/ is locked (see README.md)."
echo "=============================="
