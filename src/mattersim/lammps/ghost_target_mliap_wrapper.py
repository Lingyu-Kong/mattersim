"""Ghost-target alchemical MatterSim wrapper for LAMMPS ML-IAP."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import torch

from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
from mattersim.lammps.graph_builder import build_m3gnet_input_from_lammps
from mattersim.lammps.m3gnet_lammps import M3GnetLammps
from mattersim.lammps.mattertune_mliap_wrapper import (
    MatterTuneEnergyDenormalizer,
    MatterTuneMatterSimMLIAP,
)
from mattersim.lammps.mliap_wrapper import MatterSimMLIAP, _freeze

LOG = logging.getLogger(__name__)

ENERGY_LOG_FIELDS = (
    "step",
    "time_fs",
    "time_ps",
    "temperature_K",
    "mixed_energy_eV",
    "E_I_eV",
    "E_F_with_LJ_eV",
    "E_F_without_LJ_eV",
    "E_LJ_eV",
    "deltaE_with_LJ_eV",
    "deltaE_without_LJ_eV",
)


def _as_float_tensor(value: float, *, device: torch.device) -> torch.Tensor:
    return torch.tensor([float(value)], dtype=torch.float32, device=device)


def _scalar_float(value: torch.Tensor | float | None) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].cpu())
    return float(value)


def _load_mattertune_mattersim_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    strict: bool = False,
) -> tuple[M3Gnet, MatterTuneEnergyDenormalizer | None, float, float]:
    from mattertune.backbones.mattersim.model import MatterSimM3GNetBackboneModule
    from mattertune.main import load_finetuned_checkpoint

    module = load_finetuned_checkpoint(
        str(checkpoint_path),
        map_location=device,
        strict=strict,
    )
    if not isinstance(module, MatterSimM3GNetBackboneModule):
        raise TypeError(
            "Expected a MatterTune MatterSim checkpoint, got "
            f"{type(module).__module__}.{type(module).__qualname__}."
        )

    module.eval()
    model: M3Gnet = module.backbone.model  # type: ignore[assignment]
    cutoff = float(model.model_args["cutoff"])
    threebody_cutoff = float(model.model_args["threebody_cutoff"])
    energy_denormalizer = MatterTuneEnergyDenormalizer.from_mattertune_module(module)
    return model, energy_denormalizer, cutoff, threebody_cutoff


class GhostTargetMatterSimMLIAP(MatterTuneMatterSimMLIAP):
    """MatterTune-MatterSim delete-endpoint interpolation for LAMMPS."""

    def __init__(
        self,
        model: M3Gnet,
        *,
        lambda_value: float,
        target_types: tuple[int, ...],
        epsilon: float,
        sigma: float,
        lj_cutoff: float,
        ghost_model: M3Gnet | None = None,
        energy_denormalizer: MatterTuneEnergyDenormalizer | None = None,
        ghost_energy_denormalizer: MatterTuneEnergyDenormalizer | None = None,
        cutoff: float = 5.0,
        threebody_cutoff: float = 4.0,
        ghost_cutoff: float | None = None,
        ghost_threebody_cutoff: float | None = None,
        energy_log_path: str | Path | None = None,
        energy_log_interval: int = 1,
        energy_log_timestep_fs: float = 1.0,
        compile: bool = True,
    ) -> None:
        if not 0.0 <= float(lambda_value) <= 1.0:
            raise ValueError("lambda_value must be in [0, 1].")
        if not target_types:
            raise ValueError("target_types must contain at least one LAMMPS atom type.")
        if any(int(atom_type) <= 0 for atom_type in target_types):
            raise ValueError("LAMMPS atom types are 1-based and must be positive.")
        if sigma <= 0.0:
            raise ValueError("sigma must be positive.")
        if lj_cutoff <= 0.0:
            raise ValueError("lj_cutoff must be positive.")
        if int(energy_log_interval) <= 0:
            raise ValueError("energy_log_interval must be positive.")
        if energy_log_timestep_fs <= 0.0:
            raise ValueError("energy_log_timestep_fs must be positive.")

        super().__init__(
            model,
            energy_denormalizer=energy_denormalizer,
            cutoff=cutoff,
            threebody_cutoff=threebody_cutoff,
            compile=compile,
        )

        self.lambda_value = float(lambda_value)
        self.target_types = tuple(sorted({int(atom_type) for atom_type in target_types}))
        self.epsilon = float(epsilon)
        self.sigma = float(sigma)
        self.lj_cutoff = float(lj_cutoff)

        self.ghost_m3gnet_lammps = (
            None if ghost_model is None else M3GnetLammps(_freeze(ghost_model))
        )
        self.ghost_energy_denormalizer = (
            energy_denormalizer
            if ghost_energy_denormalizer is None
            else ghost_energy_denormalizer
        )
        self.ghost_cutoff = float(cutoff if ghost_cutoff is None else ghost_cutoff)
        self.ghost_threebody_cutoff = float(
            threebody_cutoff
            if ghost_threebody_cutoff is None
            else ghost_threebody_cutoff
        )

        # LAMMPS doubles rcutfac internally. The neighbor list must cover both
        # MatterSim graph edges and the explicit ghost-endpoint LJ correction.
        self.rcutfac = 0.5 * max(self.cutoff, self.ghost_cutoff, self.lj_cutoff)

        self.last_real_endpoint_energy: float | None = None
        self.last_ghost_base_energy: float | None = None
        self.last_ghost_lj_energy: float | None = None
        self.last_ghost_endpoint_energy: float | None = None
        self.last_effective_lj_cutoff: float | None = None

        self.energy_log_path = (
            None if energy_log_path is None else str(Path(energy_log_path))
        )
        self.energy_log_interval = int(energy_log_interval)
        self.energy_log_timestep_fs = float(energy_log_timestep_fs)
        self._energy_log_handle = None
        self._energy_log_eval_count = 0

    @classmethod
    def from_mattertune_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        lambda_value: float,
        target_types: tuple[int, ...],
        epsilon: float = 0.00694,
        sigma: float = 2.337,
        lj_cutoff: float = 10.0,
        ghost_checkpoint: str | Path | None = None,
        energy_log_path: str | Path | None = None,
        energy_log_interval: int = 1,
        energy_log_timestep_fs: float = 1.0,
        device: str = "cpu",
        strict: bool = False,
        **kwargs: Any,
    ) -> "GhostTargetMatterSimMLIAP":
        model, energy_denormalizer, cutoff, threebody_cutoff = (
            _load_mattertune_mattersim_checkpoint(
                checkpoint_path,
                device=device,
                strict=strict,
            )
        )

        ghost_model = None
        ghost_energy_denormalizer = None
        ghost_cutoff = None
        ghost_threebody_cutoff = None
        if ghost_checkpoint is not None:
            (
                ghost_model,
                ghost_energy_denormalizer,
                ghost_cutoff,
                ghost_threebody_cutoff,
            ) = _load_mattertune_mattersim_checkpoint(
                ghost_checkpoint,
                device=device,
                strict=strict,
            )

        return cls(
            model,
            lambda_value=lambda_value,
            target_types=target_types,
            epsilon=epsilon,
            sigma=sigma,
            lj_cutoff=lj_cutoff,
            ghost_model=ghost_model,
            energy_denormalizer=energy_denormalizer,
            ghost_energy_denormalizer=ghost_energy_denormalizer,
            cutoff=cutoff,
            threebody_cutoff=threebody_cutoff,
            ghost_cutoff=ghost_cutoff,
            ghost_threebody_cutoff=ghost_threebody_cutoff,
            energy_log_path=energy_log_path,
            energy_log_interval=energy_log_interval,
            energy_log_timestep_fs=energy_log_timestep_fs,
            **kwargs,
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_energy_log_handle"] = None
        return state

    def _initialize_device(self, data) -> None:  # type: ignore[no-untyped-def]
        super()._initialize_device(data)

        if self.ghost_m3gnet_lammps is not None:
            self.ghost_m3gnet_lammps = self.ghost_m3gnet_lammps.to(self.device)
            if self._compile and self.device.type == "cuda":
                LOG.info("Compiling ghost-endpoint M3GNetLammps ...")
                self.ghost_m3gnet_lammps = torch.compile(self.ghost_m3gnet_lammps)

        if (
            self.ghost_energy_denormalizer is not None
            and self.ghost_energy_denormalizer is not self.energy_denormalizer
        ):
            self.ghost_energy_denormalizer = self.ghost_energy_denormalizer.to(
                self.device
            )

    def compute_forces(self, data) -> None:  # type: ignore[no-untyped-def]
        if not self.initialized:
            self._initialize_device(data)

        if data.nlocal == 0:
            return
        if not hasattr(data, "types") or data.types is None:
            raise RuntimeError(
                "GhostTargetMatterSimMLIAP requires LAMMPS ML-IAP data.types. "
                "Rebuild LAMMPS with the patched ML-IAP Python bridge."
            )
        if not hasattr(data, "tags") or data.tags is None:
            raise RuntimeError(
                "GhostTargetMatterSimMLIAP requires LAMMPS ML-IAP data.tags "
                "to de-duplicate periodic images in the LJ correction. "
                "Rebuild LAMMPS with the patched ML-IAP Python bridge."
            )
        if not hasattr(data, "box_lengths") or data.box_lengths is None:
            raise RuntimeError(
                "GhostTargetMatterSimMLIAP requires LAMMPS ML-IAP data.box_lengths "
                "to cap the LJ correction at half the minimum box length. "
                "Rebuild LAMMPS with the patched ML-IAP Python bridge."
            )

        elems = torch.as_tensor(data.elems, dtype=torch.int64, device=self.device) + 1
        types = torch.as_tensor(data.types, dtype=torch.int64, device=self.device)
        tags = torch.as_tensor(data.tags, dtype=torch.int64, device=self.device)
        box_lengths = torch.as_tensor(
            data.box_lengths,
            dtype=torch.float64,
            device=self.device,
        )
        pair_i = torch.as_tensor(data.pair_i, dtype=torch.int64, device=self.device)
        pair_j = torch.as_tensor(data.pair_j, dtype=torch.int64, device=self.device)
        rij = torch.as_tensor(data.rij, dtype=torch.float32, device=self.device)

        target_types = torch.tensor(
            self.target_types,
            dtype=torch.int64,
            device=self.device,
        )
        target_mask = (types[:, None] == target_types[None, :]).any(dim=1)
        keep_mask = ~target_mask

        all_pair_mask = torch.ones(pair_i.shape[0], dtype=torch.bool, device=self.device)
        real_energy, real_pair_forces = self._compute_model_endpoint(
            data,
            elems=elems,
            pair_i=pair_i,
            pair_j=pair_j,
            rij=rij,
            pair_mask=all_pair_mask,
            local_mask=None,
            m3gnet_lammps=self.m3gnet_lammps,
            energy_denormalizer=self.energy_denormalizer,
            cutoff=self.cutoff,
            threebody_cutoff=self.threebody_cutoff,
        )

        ghost_pair_mask = keep_mask[pair_i] & keep_mask[pair_j]
        ghost_energy, ghost_pair_forces = self._compute_model_endpoint(
            data,
            elems=elems,
            pair_i=pair_i,
            pair_j=pair_j,
            rij=rij,
            pair_mask=ghost_pair_mask,
            local_mask=keep_mask[: data.nlocal],
            m3gnet_lammps=(
                self.m3gnet_lammps
                if self.ghost_m3gnet_lammps is None
                else self.ghost_m3gnet_lammps
            ),
            energy_denormalizer=self.ghost_energy_denormalizer,
            cutoff=self.ghost_cutoff,
            threebody_cutoff=self.ghost_threebody_cutoff,
        )

        lj_energy, lj_pair_forces = self._compute_lj_correction(
            target_mask=target_mask,
            tags=tags,
            box_lengths=box_lengths,
            pair_i=pair_i,
            pair_j=pair_j,
            rij=rij,
        )
        ghost_total_energy = ghost_energy + lj_energy.to(ghost_energy.dtype)
        ghost_total_forces = ghost_pair_forces + lj_pair_forces.to(
            ghost_pair_forces.dtype
        )

        lam = self.lambda_value
        energy = (1.0 - lam) * real_energy + lam * ghost_total_energy
        pair_forces = (1.0 - lam) * real_pair_forces + lam * ghost_total_forces

        self.last_real_endpoint_energy = float(real_energy.detach().cpu())
        self.last_ghost_base_energy = float(ghost_energy.detach().cpu())
        self.last_ghost_lj_energy = float(lj_energy.detach().cpu())
        self.last_ghost_endpoint_energy = float(ghost_total_energy.detach().cpu())

        self._write_energy_log_row(energy)

        MatterSimMLIAP._update_lammps_data(
            self,
            data,
            pair_forces.detach(),
            energy.detach(),
        )

    def _compute_model_endpoint(
        self,
        data,  # type: ignore[no-untyped-def]
        *,
        elems: torch.Tensor,
        pair_i: torch.Tensor,
        pair_j: torch.Tensor,
        rij: torch.Tensor,
        pair_mask: torch.Tensor,
        local_mask: torch.Tensor | None,
        m3gnet_lammps: torch.nn.Module,
        energy_denormalizer: MatterTuneEnergyDenormalizer | None,
        cutoff: float,
        threebody_cutoff: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_pair_indices = torch.nonzero(pair_mask, as_tuple=False).flatten()
        pair_i_selected = pair_i[selected_pair_indices]
        pair_j_selected = pair_j[selected_pair_indices]
        rij_selected = rij[selected_pair_indices]

        input_dict = build_m3gnet_input_from_lammps(
            elems=elems,
            pair_i=pair_i_selected,
            pair_j=pair_j_selected,
            rij=rij_selected,
            ntotal=data.ntotal,
            cutoff=cutoff,
            threebody_cutoff=threebody_cutoff,
            device=self.device,
        )

        sort_idx = input_dict.pop("_sort_idx")
        edge_mask = input_dict.pop("_edge_mask")
        npairs_lammps = pair_i.shape[0]

        input_dict["pbc_offsets"].requires_grad_(True)
        raw_energy = m3gnet_lammps(
            input_dict,
            nlocal=data.nlocal,
            forward_exchange_fn=data.forward_exchange,
            reverse_exchange_fn=data.reverse_exchange,
            local_mask=local_mask,
        )

        pair_forces = torch.zeros(
            (npairs_lammps, 3),
            dtype=rij.dtype,
            device=self.device,
        )
        if input_dict["pbc_offsets"].numel() > 0:
            (pair_forces_sorted,) = torch.autograd.grad(
                outputs=raw_energy,
                inputs=input_dict["pbc_offsets"],
                grad_outputs=torch.ones_like(raw_energy),
                allow_unused=True,
            )
            if pair_forces_sorted is None:
                pair_forces_sorted = torch.zeros_like(input_dict["pbc_offsets"])
            inverse_sort = torch.argsort(sort_idx)
            pair_forces_filtered = pair_forces_sorted[inverse_sort]
            pair_forces[selected_pair_indices[edge_mask]] = pair_forces_filtered

        local_atomic_numbers = elems[: data.nlocal]
        if local_mask is not None:
            local_atomic_numbers = local_atomic_numbers[local_mask]
        energy = raw_energy
        if energy_denormalizer is not None:
            energy = energy_denormalizer(raw_energy, local_atomic_numbers)

        return energy, pair_forces

    def _compute_lj_correction(
        self,
        *,
        target_mask: torch.Tensor,
        tags: torch.Tensor,
        box_lengths: torch.Tensor,
        pair_i: torch.Tensor,
        pair_j: torch.Tensor,
        rij: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair_forces = torch.zeros_like(rij)
        if box_lengths.numel() != 3 or torch.any(box_lengths <= 0.0):
            raise ValueError(
                "LJ correction requires three positive periodic box lengths."
            )
        effective_lj_cutoff = torch.minimum(
            torch.as_tensor(self.lj_cutoff, dtype=torch.float64, device=self.device),
            0.5 * torch.min(box_lengths),
        )
        self.last_effective_lj_cutoff = float(effective_lj_cutoff.detach().cpu())

        distances_sq = torch.sum(rij.to(torch.float64) ** 2, dim=1)
        pair_mask = (
            target_mask[pair_i]
            ^ target_mask[pair_j]
        ) & (distances_sq <= self.lj_cutoff**2)

        if not torch.any(pair_mask):
            return _as_float_tensor(0.0, device=self.device), pair_forces

        pair_indices = torch.nonzero(pair_mask, as_tuple=False).flatten()
        r2_all = distances_sq[pair_indices]
        target_on_i = target_mask[pair_i[pair_indices]]
        target_tags = torch.where(
            target_on_i,
            tags[pair_i[pair_indices]],
            tags[pair_j[pair_indices]],
        )
        env_tags = torch.where(
            target_on_i,
            tags[pair_j[pair_indices]],
            tags[pair_i[pair_indices]],
        )
        key_stride = tags.max() + 1
        pair_keys = target_tags * key_stride + env_tags
        unique_keys, inverse = torch.unique(pair_keys, return_inverse=True)
        min_r2 = torch.full(
            (unique_keys.shape[0],),
            torch.inf,
            dtype=r2_all.dtype,
            device=self.device,
        )
        min_r2.scatter_reduce_(0, inverse, r2_all, reduce="amin", include_self=True)
        nearest_mask = torch.isclose(
            r2_all,
            min_r2[inverse],
            rtol=1e-7,
            atol=1e-10,
        )
        nearest_mask &= min_r2[inverse] <= effective_lj_cutoff**2
        if not torch.any(nearest_mask):
            return _as_float_tensor(0.0, device=self.device), pair_forces

        selected_pair_indices = pair_indices[nearest_mask]
        selected_inverse = inverse[nearest_mask]

        selected_counts = torch.zeros(
            (unique_keys.shape[0],),
            dtype=r2_all.dtype,
            device=self.device,
        )
        selected_counts.scatter_add_(
            0,
            selected_inverse,
            torch.ones_like(selected_inverse, dtype=r2_all.dtype),
        )
        weights = 1.0 / selected_counts[selected_inverse]

        r2 = distances_sq[selected_pair_indices]
        if torch.any(r2 <= 0.0):
            raise ValueError(
                "LJ correction encountered a zero-distance target-environment pair."
            )

        reduced_r6 = (r2 / self.sigma**2) ** 3
        pair_energies = 4.0 * self.epsilon * (
            1.0 / reduced_r6**2 - 1.0 / reduced_r6
        )
        force_scalar = (
            24.0
            * self.epsilon
            * (-2.0 / reduced_r6**2 + 1.0 / reduced_r6)
            / r2
        )

        # Match the target ASE-style MIC behavior: for each target-environment
        # tag pair, keep only the shortest available image and then apply the
        # effective cutoff min(lj_cutoff, Lmin/2).
        pair_forces[selected_pair_indices] = (
            weights[:, None].to(rij.dtype)
            * force_scalar[:, None].to(rij.dtype)
            * rij[selected_pair_indices]
        )
        energy = (weights * pair_energies).sum()
        return energy.reshape(1).to(rij.dtype), pair_forces

    def _open_energy_log(self):
        energy_log_path = getattr(self, "energy_log_path", None)
        if energy_log_path is None:
            return None
        energy_log_handle = getattr(self, "_energy_log_handle", None)
        if energy_log_handle is not None:
            return energy_log_handle

        path = Path(energy_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("w", encoding="utf-8", newline="")
        handle.write(",".join(ENERGY_LOG_FIELDS) + "\n")
        handle.flush()
        self._energy_log_handle = handle
        return handle

    def _write_energy_log_row(self, mixed_energy: torch.Tensor) -> None:
        if getattr(self, "energy_log_path", None) is None:
            return

        eval_index = int(getattr(self, "_energy_log_eval_count", 0))
        self._energy_log_eval_count = eval_index + 1
        energy_log_interval = int(getattr(self, "energy_log_interval", 1))
        if eval_index % energy_log_interval != 0:
            return

        handle = self._open_energy_log()
        if handle is None:
            return

        timestep_fs = float(getattr(self, "energy_log_timestep_fs", 1.0))
        time_fs = eval_index * timestep_fs
        real_energy = _scalar_float(self.last_real_endpoint_energy)
        ghost_total_energy = _scalar_float(self.last_ghost_endpoint_energy)
        ghost_base_energy = _scalar_float(self.last_ghost_base_energy)
        ghost_lj_energy = _scalar_float(self.last_ghost_lj_energy)
        mixed_energy_float = _scalar_float(mixed_energy)
        delta_with_lj = ghost_total_energy - real_energy
        delta_without_lj = ghost_base_energy - real_energy

        row = (
            eval_index,
            time_fs,
            time_fs / 1000.0,
            float("nan"),
            mixed_energy_float,
            real_energy,
            ghost_total_energy,
            ghost_base_energy,
            ghost_lj_energy,
            delta_with_lj,
            delta_without_lj,
        )
        handle.write(
            ",".join(
                str(value) if isinstance(value, int) else f"{value:.16g}"
                for value in row
            )
            + "\n"
        )
        handle.flush()

    def save(self, path: str) -> None:
        saved = copy.deepcopy(self)
        saved._energy_log_handle = None
        saved._energy_log_eval_count = 0
        saved.m3gnet_lammps = M3GnetLammps(saved.m3gnet_lammps.m3gnet.cpu())
        if saved.ghost_m3gnet_lammps is not None:
            saved.ghost_m3gnet_lammps = M3GnetLammps(
                saved.ghost_m3gnet_lammps.m3gnet.cpu()
            )
        torch.save(saved, path)
