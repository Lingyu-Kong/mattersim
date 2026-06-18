LAMMPS Ghost-Target FEP-TI Setup
================================

MatterTune ghost-target FEP-TI in LAMMPS requires a small ML-IAP Python bridge
patch. The patch exposes three pieces of LAMMPS runtime metadata to Python
models:

- ``data.types``: original LAMMPS atom type for each local/ghost atom.
- ``data.tags``: original LAMMPS atom ID for periodic-image de-duplication.
- ``data.box_lengths``: current periodic box lengths for LJ cutoff capping.

The Python model code lives in MatterSim/MatterTune. The LAMMPS patch is only a
bridge extension and should be applied to official LAMMPS source.

Tested LAMMPS Base
------------------

The bundled patch was generated against:

.. code-block:: text

   LAMMPS tag: stable_22Jul2025
   LAMMPS commit: c7ae612a9497437412cb787b78769570f48653dd
   Upstream branches containing the tag: origin/stable, origin/maintenance

The patch file is:

.. code-block:: text

   patches/lammps/0001-mliap-python-expose-types-tags-box-lengths-stable_22Jul2025.patch

Quick Environment Setup
-----------------------

Assume MatterSim and MatterTune are cloned into the same parent directory:

.. code-block:: bash

   mkdir -p ~/workspace/electrolyte-fep
   cd ~/workspace/electrolyte-fep

   git clone <your-mattersim-repo-url> mattersim
   git clone <your-mattertune-repo-url> MatterTune

Create and activate a Python environment:

.. code-block:: bash

   conda create -n mattersim-elec python=3.10 -y
   conda activate mattersim-elec

   python -m pip install -U pip setuptools wheel

Install PyTorch matching your CUDA/driver stack. For CUDA 12.6 wheels:

.. code-block:: bash

   python -m pip install torch --index-url https://download.pytorch.org/whl/cu126

Install the local packages:

.. code-block:: bash

   python -m pip install -e ./mattersim
   python -m pip install -e ./MatterTune

Build Patched LAMMPS
--------------------

Use the bundled build helper from the MatterSim repo:

.. code-block:: bash

   cd ~/workspace/electrolyte-fep/mattersim

   bash scripts/build_lammps_mliap_kokkos.sh \
     --work-root ~/workspace/electrolyte-fep/_lammps \
     --kokkos-arch AMPERE86

If ``nvcc`` is not on ``PATH``, pass CUDA explicitly:

.. code-block:: bash

   bash scripts/build_lammps_mliap_kokkos.sh \
     --work-root ~/workspace/electrolyte-fep/_lammps \
     --cuda-root /usr/local/cuda-12.6 \
     --kokkos-arch AMPERE86

Common Kokkos architecture names:

===========  ============================
Arch         GPUs
===========  ============================
VOLTA70      V100
AMPERE80     A100
AMPERE86     RTX A6000, A5000, A4000
ADA89        RTX 4090, L40, L40S
HOPPER90     H100, H200
===========  ============================

The script installs LAMMPS into the active conda environment by default:

.. code-block:: bash

   ${CONDA_PREFIX}/bin/lmp
   ${CONDA_PREFIX}/lib/liblammps.so

Use the full ``${CONDA_PREFIX}/bin/lmp`` path if another ``lmp`` appears earlier
in ``PATH``.

Build Script Notes
------------------

The helper performs:

.. code-block:: bash

   git clone https://github.com/lammps/lammps.git
   git checkout stable_22Jul2025
   git apply mattersim/patches/lammps/0001-mliap-python-expose-types-tags-box-lengths-stable_22Jul2025.patch
   cmake -D PKG_ML-IAP=on -D MLIAP_ENABLE_PYTHON=on -D PKG_KOKKOS=on ...
   cmake --build ...
   cmake --install ...

To use an existing LAMMPS checkout:

.. code-block:: bash

   bash scripts/build_lammps_mliap_kokkos.sh \
     --source-dir /path/to/lammps \
     --build-dir /path/to/lammps/build-mattersim-mliap-kokkos \
     --skip-clone \
     --kokkos-arch AMPERE86

To apply the patch manually:

.. code-block:: bash

   cd /path/to/lammps
   git checkout stable_22Jul2025
   git apply /path/to/mattersim/patches/lammps/0001-mliap-python-expose-types-tags-box-lengths-stable_22Jul2025.patch

Verification
------------

Check that the installed executable is the conda-env LAMMPS:

.. code-block:: bash

   ${CONDA_PREFIX}/bin/lmp -h | head

Check Python imports:

.. code-block:: bash

   PYTHONNOUSERSITE=1 python - <<'PY'
   import lammps
   from mattersim.lammps.ghost_target_mliap_wrapper import GhostTargetMatterSimMLIAP
   print("lammps:", lammps.__file__)
   print("wrapper:", GhostTargetMatterSimMLIAP.__name__)
   PY

Runtime Command
---------------

For single-GPU Kokkos runs:

.. code-block:: bash

   PYTHONNOUSERSITE=1 ${CONDA_PREFIX}/bin/lmp \
     -k on g 1 \
     -sf kk \
     -pk kokkos newton on neigh half \
     -in in.lammps

``pair_mliap`` requires ``newton on``. With Kokkos, also request half neighbor
lists via ``-pk kokkos newton on neigh half``.

MatterTune Mix FEP-TI Example
-----------------------------

After installing patched LAMMPS, run the packaged MatterTune example:

.. code-block:: bash

   cd ~/workspace/electrolyte-fep/MatterTune

   bash examples/elec-Li-new/mix_further_ft/run_lammps_fep_ti.sh \
     --lambda-value 0.5 \
     --cuda-visible-devices 0

Useful overrides:

.. code-block:: bash

   bash examples/elec-Li-new/mix_further_ft/run_lammps_fep_ti.sh \
     --lambda-value 0.5 \
     --checkpoint /path/to/mattersim-best.ckpt \
     --structure /path/to/top.pdb \
     --run-dir /path/to/run_lambda_0p5 \
     --steps 100000 \
     --warmup-steps 20 \
     --target-indices 0 \
     --lj-cutoff 10.0 \
     --cuda-visible-devices 0

The example writes:

.. code-block:: text

   ghost-target-lambdaXXX-type8-rc10.0.pt
   top_target_type8.data
   in.lambdaXXX.lammps
   prepare_metadata.json
   log.lambdaXXX.lammps
   traj_lambdaXXX.lammpstrj
   final_lambdaXXX.data

The target atom index is zero-based in the PDB/ASE convention. The converter
writes that target Li as a separate LAMMPS atom type, while ``pair_coeff`` maps
both ordinary Li and target Li back to element ``Li``.

Limitations
-----------

- The current ghost-target wrapper is validated for single MPI rank and one
  Kokkos GPU.
- Orthorhombic periodic cells are assumed by the provided PDB-to-data helper.
- The soft-core LJ correction uses one MIC interaction per target-environment
  atom pair and an effective cutoff ``min(lj_cutoff, Lmin/2)``.
- Multi-rank/domain-decomposed FEP-TI needs additional validation.
