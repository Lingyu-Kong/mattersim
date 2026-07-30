#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATTERSIM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LAMMPS_REPO="${LAMMPS_REPO:-https://github.com/lammps/lammps.git}"
LAMMPS_REF="${LAMMPS_REF:-stable_22Jul2025}"
PATCH_PATH="${PATCH_PATH:-${MATTERSIM_ROOT}/patches/lammps/0001-mliap-python-expose-types-tags-box-lengths-stable_22Jul2025.patch}"

WORK_ROOT="${WORK_ROOT:-${PWD}/lammps-mliap-build}"
SOURCE_DIR="${SOURCE_DIR:-}"
BUILD_DIR="${BUILD_DIR:-}"
INSTALL_PREFIX="${INSTALL_PREFIX:-${CONDA_PREFIX:-}}"
WRAPPER_DIR="${WRAPPER_DIR:-${CONDA_PREFIX:+${CONDA_PREFIX}/bin}}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-$(command -v python || true)}"

CUDA_ROOT="${CUDA_ROOT:-}"
KOKKOS_ARCH="${KOKKOS_ARCH:-}"
LMP_NAME="${LMP_NAME:-lmp}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
BUILD_SHARED_LIBS="${BUILD_SHARED_LIBS:-yes}"
BUILD_MPI="${BUILD_MPI:-yes}"
EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS:-}"
SKIP_CLONE="${SKIP_CLONE:-0}"
SKIP_PATCH="${SKIP_PATCH:-0}"
INSTALL_PYTHON="${INSTALL_PYTHON:-1}"
CREATE_WRAPPER="${CREATE_WRAPPER:-1}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Build patched LAMMPS ML-IAP + Kokkos CUDA for MatterSim ghost-target FEP-TI.

Usage:
  bash scripts/build_lammps_mliap_kokkos.sh [options]

Options:
  --work-root PATH          Parent directory for clone/build.
  --source-dir PATH         LAMMPS source checkout.
  --build-dir PATH          CMake build directory.
  --install-prefix PATH     Architecture-specific installation directory.
  --wrapper-dir PATH        Directory where the launcher is created.
  --lmp-name NAME           Final executable name, such as lmp_L40S.
  --repo URL                LAMMPS git repository.
  --ref REF                 LAMMPS commit, tag, or branch.
  --patch PATH              MatterSim ML-IAP patch.
  --python-executable PATH  Python used by the ML-IAP bridge.
  --cuda-root PATH          CUDA root containing bin/nvcc.
  --kokkos-arch NAME        Kokkos GPU architecture.
  --jobs N                  Parallel build jobs.
  --skip-clone              Use existing source directory.
  --skip-patch              Do not apply the MatterSim patch.
  --skip-install-python     Do not install the LAMMPS Python wheel.
  --skip-wrapper            Do not create a launcher in WRAPPER_DIR.
  --dry-run                 Print commands only.
  -h, --help                Show this help.

Examples:
  --kokkos-arch VOLTA70 --lmp-name lmp_V100
  --kokkos-arch AMPERE80 --lmp-name lmp_A100
  --kokkos-arch ADA89 --lmp-name lmp_L40S
  --kokkos-arch HOPPER90 --lmp-name lmp_H100

Environment overrides:
  LAMMPS_REPO LAMMPS_REF PATCH_PATH WORK_ROOT SOURCE_DIR BUILD_DIR
  INSTALL_PREFIX WRAPPER_DIR PYTHON_EXECUTABLE CUDA_ROOT KOKKOS_ARCH
  LMP_NAME JOBS BUILD_SHARED_LIBS BUILD_MPI EXTRA_CMAKE_ARGS
  SKIP_CLONE SKIP_PATCH INSTALL_PYTHON CREATE_WRAPPER DRY_RUN
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --source-dir) SOURCE_DIR="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --install-prefix) INSTALL_PREFIX="$2"; shift 2 ;;
    --wrapper-dir) WRAPPER_DIR="$2"; shift 2 ;;
    --lmp-name) LMP_NAME="$2"; shift 2 ;;
    --repo) LAMMPS_REPO="$2"; shift 2 ;;
    --ref) LAMMPS_REF="$2"; shift 2 ;;
    --patch) PATCH_PATH="$2"; shift 2 ;;
    --python-executable) PYTHON_EXECUTABLE="$2"; shift 2 ;;
    --cuda-root) CUDA_ROOT="$2"; shift 2 ;;
    --kokkos-arch) KOKKOS_ARCH="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --skip-clone) SKIP_CLONE=1; shift ;;
    --skip-patch) SKIP_PATCH=1; shift ;;
    --skip-install-python) INSTALL_PYTHON=0; shift ;;
    --skip-wrapper) CREATE_WRAPPER=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${SOURCE_DIR}" ]]; then
  SOURCE_DIR="${WORK_ROOT}/lammps"
fi

if [[ -z "${BUILD_DIR}" ]]; then
  BUILD_DIR="${SOURCE_DIR}/build-mattersim-mliap-kokkos"
fi

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

if [[ -z "${INSTALL_PREFIX}" ]]; then
  echo "INSTALL_PREFIX is empty. Activate a Conda environment or pass --install-prefix." >&2
  exit 2
fi

if [[ -z "${PYTHON_EXECUTABLE}" || ! -x "${PYTHON_EXECUTABLE}" ]]; then
  echo "Could not find a Python executable." >&2
  exit 2
fi

if [[ "${SKIP_PATCH}" != "1" && ! -f "${PATCH_PATH}" ]]; then
  echo "Patch file not found: ${PATCH_PATH}" >&2
  exit 2
fi

if [[ ! "${LMP_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid executable name: ${LMP_NAME}" >&2
  exit 2
fi

if [[ ! "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid number of build jobs: ${JOBS}" >&2
  exit 2
fi

detect_cuda_root() {
  if [[ -n "${CUDA_ROOT}" ]]; then
    return
  fi

  local nvcc_path
  nvcc_path="$(command -v nvcc || true)"

  if [[ -z "${nvcc_path}" ]]; then
    nvcc_path="$(find /usr/local /usr -path '*/bin/nvcc' -type f 2>/dev/null | sort -V | tail -n 1 || true)"
  fi

  if [[ -z "${nvcc_path}" ]]; then
    echo "Could not find nvcc. Pass --cuda-root /path/to/cuda." >&2
    exit 2
  fi

  CUDA_ROOT="$(cd "$(dirname "${nvcc_path}")/.." && pwd)"
}

detect_kokkos_arch() {
  if [[ -n "${KOKKOS_ARCH}" ]]; then
    return
  fi

  local capability
  capability="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true)"

  case "${capability}" in
    7.0) KOKKOS_ARCH="VOLTA70" ;;
    7.5) KOKKOS_ARCH="TURING75" ;;
    8.0) KOKKOS_ARCH="AMPERE80" ;;
    8.6) KOKKOS_ARCH="AMPERE86" ;;
    8.7) KOKKOS_ARCH="AMPERE87" ;;
    8.9) KOKKOS_ARCH="ADA89" ;;
    9.0) KOKKOS_ARCH="HOPPER90" ;;
    10.0) KOKKOS_ARCH="BLACKWELL100" ;;
    10.3) KOKKOS_ARCH="BLACKWELL103" ;;
    12.0) KOKKOS_ARCH="BLACKWELL120" ;;
    12.1) KOKKOS_ARCH="BLACKWELL121" ;;
    *)
      KOKKOS_ARCH="AMPERE80"
      echo "Could not auto-detect Kokkos architecture. Defaulting to ${KOKKOS_ARCH}." >&2
      echo "Pass --kokkos-arch explicitly for the target GPU." >&2
      ;;
  esac
}

detect_cuda_root
detect_kokkos_arch

if [[ ! -x "${CUDA_ROOT}/bin/nvcc" ]]; then
  echo "nvcc not found or not executable: ${CUDA_ROOT}/bin/nvcc" >&2
  exit 2
fi

export PATH="${CUDA_ROOT}/bin:${PATH}"

echo "LAMMPS_REPO=${LAMMPS_REPO}"
echo "LAMMPS_REF=${LAMMPS_REF}"
echo "SOURCE_DIR=${SOURCE_DIR}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "INSTALL_PREFIX=${INSTALL_PREFIX}"
echo "WRAPPER_DIR=${WRAPPER_DIR}"
echo "PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}"
echo "CUDA_ROOT=${CUDA_ROOT}"
echo "KOKKOS_ARCH=${KOKKOS_ARCH}"
echo "LMP_NAME=${LMP_NAME}"
echo "JOBS=${JOBS}"
echo "PATCH_PATH=${PATCH_PATH}"
echo "INSTALL_PYTHON=${INSTALL_PYTHON}"
echo "CREATE_WRAPPER=${CREATE_WRAPPER}"

if [[ "${SKIP_CLONE}" != "1" ]]; then
  if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
    run_cmd mkdir -p "$(dirname "${SOURCE_DIR}")"
    run_cmd git clone "${LAMMPS_REPO}" "${SOURCE_DIR}"
  fi

  run_cmd git -C "${SOURCE_DIR}" fetch --tags origin
  run_cmd git -C "${SOURCE_DIR}" checkout "${LAMMPS_REF}"
fi

if [[ "${SKIP_PATCH}" != "1" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] would check and apply patch: ${PATCH_PATH}"
  elif git -C "${SOURCE_DIR}" apply --reverse --check "${PATCH_PATH}" >/dev/null 2>&1; then
    echo "Patch already applied: ${PATCH_PATH}"
  else
    run_cmd git -C "${SOURCE_DIR}" apply --check "${PATCH_PATH}"
    run_cmd git -C "${SOURCE_DIR}" apply "${PATCH_PATH}"
  fi
fi

run_cmd mkdir -p "${BUILD_DIR}"
run_cmd mkdir -p "${INSTALL_PREFIX}"

EXTRA_ARGS=()
if [[ -n "${EXTRA_CMAKE_ARGS}" ]]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_CMAKE_ARGS}"
fi

run_cmd cmake -S "${SOURCE_DIR}/cmake" -B "${BUILD_DIR}" \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -D CMAKE_CUDA_COMPILER="${CUDA_ROOT}/bin/nvcc" \
  -D BUILD_SHARED_LIBS="${BUILD_SHARED_LIBS}" \
  -D BUILD_MPI="${BUILD_MPI}" \
  -D PKG_ML-IAP=on \
  -D PKG_MANYBODY=yes \
  -D PKG_MOLECULE=yes \
  -D PKG_EXTRA-PAIR=yes \
  -D PKG_EXTRA-DUMP=yes \
  -D PKG_KSPACE=yes \
  -D PKG_EXTRA-FIX=yes \
  -D MLIAP_ENABLE_PYTHON=on \
  -D PKG_PYTHON=on \
  -D PKG_KOKKOS=on \
  -D Kokkos_ENABLE_CUDA=on \
  -D Kokkos_ARCH_"${KOKKOS_ARCH}"=on \
  -D Python_EXECUTABLE="${PYTHON_EXECUTABLE}" \
  -D CMAKE_INSTALL_RPATH='$ORIGIN/../lib' \
  "${EXTRA_ARGS[@]}"

run_cmd cmake --build "${BUILD_DIR}" -j "${JOBS}"
run_cmd cmake --install "${BUILD_DIR}"

if [[ "${INSTALL_PYTHON}" == "1" ]]; then
  run_cmd cmake --build "${BUILD_DIR}" --target install-python
fi

INSTALLED_LMP="${INSTALL_PREFIX}/bin/lmp"
NAMED_LMP="${INSTALL_PREFIX}/bin/${LMP_NAME}"

if [[ "${DRY_RUN}" != "1" && ! -x "${INSTALLED_LMP}" ]]; then
  echo "Installed LAMMPS executable not found: ${INSTALLED_LMP}" >&2
  exit 2
fi

if [[ "${LMP_NAME}" != "lmp" ]]; then
  run_cmd mv -f "${INSTALLED_LMP}" "${NAMED_LMP}"
else
  NAMED_LMP="${INSTALLED_LMP}"
fi

if [[ "${CREATE_WRAPPER}" == "1" ]]; then
  if [[ -z "${WRAPPER_DIR}" ]]; then
    echo "WRAPPER_DIR is empty. Pass --wrapper-dir or activate a Conda environment." >&2
    exit 2
  fi

  run_cmd mkdir -p "${WRAPPER_DIR}"

  WRAPPER_PATH="${WRAPPER_DIR}/${LMP_NAME}"

  if [[ "${WRAPPER_PATH}" != "${NAMED_LMP}" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[dry-run] would create wrapper: ${WRAPPER_PATH}"
    else
      cat > "${WRAPPER_PATH}" <<EOF
#!/usr/bin/env bash
set -e
prefix="${INSTALL_PREFIX}"
export LD_LIBRARY_PATH="\${prefix}/lib:\${prefix}/lib64:\${LD_LIBRARY_PATH:-}"
exec "\${prefix}/bin/${LMP_NAME}" "\$@"
EOF
      chmod +x "${WRAPPER_PATH}"
    fi
  fi
fi

cat <<EOF

Installed patched LAMMPS:

  Architecture:
    ${KOKKOS_ARCH}

  Installation prefix:
    ${INSTALL_PREFIX}

  Real executable:
    ${NAMED_LMP}

  Launcher:
    ${WRAPPER_DIR}/${LMP_NAME}

Recommended runtime command:

  ${LMP_NAME} -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.lammps

Verification:

  which ${LMP_NAME}
  ${LMP_NAME} -h | head -40
  ldd ${NAMED_LMP} | grep -E 'lammps|cuda|kokkos|python'
EOF