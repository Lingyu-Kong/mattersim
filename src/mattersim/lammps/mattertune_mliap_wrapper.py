"""LAMMPS MLIAP export path for MatterTune-finetuned MatterSim models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
from mattersim.lammps.mliap_wrapper import MatterSimMLIAP

LOG = logging.getLogger(__name__)

# MatterTune's MatterSim adapter builds composition vectors with
# F.one_hot(atomic_numbers, num_classes=120)[:, 1:].
_MATTERTUNE_NUM_ATOM_CLASSES = 120


class _EnergyDenormalizerOp(nn.Module):
    """One MatterTune normalizer converted into a runtime-only operation."""

    def __init__(
        self,
        kind: str,
        *,
        only_for_target: bool,
        references: torch.Tensor | None = None,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        rms: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.only_for_target = bool(only_for_target)

        if references is not None:
            self.register_buffer("references", references.detach().clone().float())
        if mean is not None:
            self.register_buffer("mean", mean.detach().clone().float())
        if std is not None:
            self.register_buffer("std", std.detach().clone().float())
        if rms is not None:
            self.register_buffer("rms", rms.detach().clone().float())

    @classmethod
    def from_mattertune_normalizer(
        cls,
        normalizer: nn.Module,
    ) -> "_EnergyDenormalizerOp":
        kind = type(normalizer).__name__
        only_for_target = bool(getattr(normalizer, "only_for_target"))

        if kind == "PerAtomReferencingNormalizerModule":
            return cls(
                "per_atom_reference",
                only_for_target=only_for_target,
                references=getattr(normalizer, "references").detach().cpu(),
            )
        if kind == "PerAtomNormalizerModule":
            return cls("per_atom", only_for_target=only_for_target)
        if kind == "MeanStdNormalizerModule":
            return cls(
                "mean_std",
                only_for_target=only_for_target,
                mean=getattr(normalizer, "mean").detach().cpu(),
                std=getattr(normalizer, "std").detach().cpu(),
            )
        if kind == "RMSNormalizerModule":
            return cls(
                "rms",
                only_for_target=only_for_target,
                rms=getattr(normalizer, "rms").detach().cpu(),
            )

        raise NotImplementedError(f"Unsupported MatterTune normalizer: {kind}")

    def denormalize(
        self,
        energy: torch.Tensor,
        *,
        num_atoms: torch.Tensor,
        compositions: torch.Tensor,
    ) -> torch.Tensor:
        if self.kind == "per_atom_reference":
            references = self.references
            compositions = compositions[:, : references.numel()].to(references.dtype)
            shift = torch.einsum("ij,j->i", compositions, references).reshape(
                energy.shape
            )
            return energy + shift

        if self.kind == "per_atom":
            if len(energy.shape) == 1:
                return energy * num_atoms
            return energy * num_atoms[:, None]

        if self.kind == "mean_std":
            return energy * self.std + self.mean

        if self.kind == "rms":
            return energy * self.rms

        raise RuntimeError(f"Unknown energy denormalizer op: {self.kind}")


class MatterTuneEnergyDenormalizer(nn.Module):
    """Apply MatterTune prediction-time energy denormalization in LAMMPS."""

    def __init__(self, ops: Iterable[_EnergyDenormalizerOp]) -> None:
        super().__init__()
        self.ops = nn.ModuleList(ops)

    @classmethod
    def from_mattertune_module(
        cls,
        module: nn.Module,
    ) -> "MatterTuneEnergyDenormalizer | None":
        normalizers = getattr(module, "normalizers", None)
        if not normalizers:
            return None

        energy_prop_name = getattr(module, "energy_prop_name", "energy")
        if energy_prop_name not in normalizers:
            return None

        energy_normalizer = normalizers[energy_prop_name]
        ops = [
            _EnergyDenormalizerOp.from_mattertune_normalizer(normalizer)
            for normalizer in getattr(energy_normalizer, "normalizers")
        ]
        return cls(ops)

    def forward(
        self,
        energy: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        atomic_numbers = atomic_numbers.long()
        compositions = F.one_hot(
            atomic_numbers,
            num_classes=_MATTERTUNE_NUM_ATOM_CLASSES,
        ).sum(dim=0, keepdim=True)
        compositions = compositions[:, 1:]
        num_atoms = torch.tensor(
            [atomic_numbers.numel()],
            device=atomic_numbers.device,
            dtype=energy.dtype,
        )

        for op in reversed(self.ops):
            if op.only_for_target:
                energy = op.denormalize(
                    energy,
                    num_atoms=num_atoms,
                    compositions=compositions,
                )
        return energy


class MatterTuneMatterSimMLIAP(MatterSimMLIAP):
    """MatterSim MLIAP wrapper that preserves MatterTune energy normalizers."""

    def __init__(
        self,
        model: M3Gnet,
        *,
        energy_denormalizer: MatterTuneEnergyDenormalizer | None = None,
        cutoff: float = 5.0,
        threebody_cutoff: float = 4.0,
        compile: bool = True,
    ) -> None:
        super().__init__(
            model,
            cutoff=cutoff,
            threebody_cutoff=threebody_cutoff,
            compile=compile,
        )
        self.energy_denormalizer = energy_denormalizer

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        strict: bool = False,
        **kwargs: Any,
    ) -> "MatterTuneMatterSimMLIAP":
        """Load a MatterTune-finetuned MatterSim checkpoint for LAMMPS.

        ``strict=False`` is the default because MatterSim has added derived
        buffers such as ``sbf.coef`` across versions. These are reconstructed
        when the pretrained backbone is instantiated and are not learned
        checkpoint parameters.
        """
        from mattertune.backbones.mattersim.model import (
            MatterSimM3GNetBackboneModule,
        )
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
        energy_denormalizer = MatterTuneEnergyDenormalizer.from_mattertune_module(
            module
        )

        return cls(
            model,
            cutoff=cutoff,
            threebody_cutoff=threebody_cutoff,
            energy_denormalizer=energy_denormalizer,
            **kwargs,
        )

    def _initialize_device(self, data) -> None:  # type: ignore[no-untyped-def]
        super()._initialize_device(data)
        if self.energy_denormalizer is not None:
            self.energy_denormalizer = self.energy_denormalizer.to(self.device)

    def _update_lammps_data(
        self,
        data,  # type: ignore[no-untyped-def]
        pair_forces: torch.Tensor,
        energy: torch.Tensor,
    ) -> None:
        if self.energy_denormalizer is not None:
            atomic_numbers = (
                torch.as_tensor(data.elems, dtype=torch.int64, device=self.device)[
                    : data.nlocal
                ]
                + 1
            )
            energy = self.energy_denormalizer(energy, atomic_numbers)

        super()._update_lammps_data(data, pair_forces, energy)

    def save(self, path: str) -> None:
        if self.energy_denormalizer is None:
            LOG.warning("Saving MatterTune MLIAP wrapper without energy denormalizer.")
        super().save(path)
