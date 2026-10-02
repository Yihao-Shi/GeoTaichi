"""Schema-versioned exact checkpoints for explicit FEM--DEM coupling.

The checkpoint contains every direct Taichi/NumPy state field owned by the
coupled explicit runtime, including FEM nodal and constitutive state, DEM body
state, broad-phase data, and persistent contact history. Immutable modeling
input is rebuilt by the caller and verified through a topology fingerprint.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.utils.FieldIO import field_to_numpy_prefix


SCHEMA_NAME = "geotaichi.fedem.checkpoint"
SCHEMA_VERSION = 1


def _owned_array(value) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        return np.array(array, copy=True)
    return np.ascontiguousarray(array)


def _hash_array(digest, name: str, value) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def _topology_fingerprint(coupling) -> str:
    digest = hashlib.sha256()
    mesh = coupling.fem.scene.mesh
    _hash_array(digest, "fem.rest_shape", mesh.rest_shape)
    _hash_array(digest, "fem.cells", mesh.cells)
    _hash_array(digest, "fem.surface_faces", coupling.surface_faces)
    _hash_array(digest, "fem.surface_body", coupling.surface_body)
    digest.update(str(coupling.dem.sims.scheme).encode("utf-8"))
    digest.update(str(type(coupling.fem.engine).__name__).encode("utf-8"))
    digest.update(str(type(coupling.contactor.model).__name__).encode("utf-8"))
    return digest.hexdigest()


def _component_objects(coupling) -> dict[str, Any]:
    fem_engine = coupling.fem.engine
    dem = coupling.dem
    objects = {
        "coupling.simulation": coupling.sims,
        "fem.state": fem_engine.state,
        "fem.engine": fem_engine,
        "fem.assembler": getattr(fem_engine, "classical_assembler", None),
        "fem.soft_contact": getattr(fem_engine, "soft_particle_contact", None),
        "dem.simulation": dem.sims,
        "dem.scene": dem.scene,
        "dem.neighbor": getattr(getattr(dem, "contactor", None), "neighbor", None),
        "dem.contact_pp": getattr(getattr(dem, "contactor", None), "physpp", None),
        "dem.contact_pw": getattr(getattr(dem, "contactor", None), "physpw", None),
        "coupling.patch": coupling.patch,
        "coupling.contactor": coupling.contactor,
        "coupling.neighbor": getattr(coupling.contactor, "neighbor", None),
        "coupling.contact_model": coupling.contactor.model,
    }
    neighbor = objects["coupling.neighbor"]
    # Broad-phase trees, traversal stacks and raw candidate buffers are
    # rebuild scratch. The compact contact records below are sufficient for
    # an exact next explicit step and avoid checkpointing raw BVH capacity.
    soft_contact = objects["fem.soft_contact"]
    if soft_contact is not None:
        # The PT/EE stencils are derived from geometry but are part of the
        # exact explicit contact state between Verlet rebuilds: the Python
        # counts select their active prefixes and the tangential histories in
        # fem.soft_contact are indexed by these stencils.  Archive only the
        # compact culling outputs needed to resume, not the much larger raw
        # broad-phase work arrays.
        culling = soft_contact.culling
        objects["fem.soft_contact_candidates"] = SimpleNamespace(
            point_triangle=culling.point_triangle,
            edge_edge=culling.edge_edge,
            point_triangle_measure=culling.point_triangle_measure,
            edge_edge_measure=culling.edge_edge_measure,
            point_triangle_count=culling.point_triangle_count,
            edge_edge_count=culling.edge_edge_count,
            raw_point_triangle_count=culling.raw_point_triangle_count,
            raw_edge_edge_count=culling.raw_edge_edge_count,
            compact_point_triangle_count=int(soft_contact.pt_count),
            compact_edge_edge_count=int(soft_contact.ee_count),
        )
    return {name: value for name, value in objects.items() if value is not None}


def _array_key(component: str, attribute: str, member: str | None = None) -> str:
    base = f"state/{component}/{attribute}"
    return base if member is None else f"{base}/{member}"


def _capture_field(
    arrays,
    manifest,
    component,
    attribute,
    field,
    prefix_count=None,
) -> bool:
    if not callable(getattr(field, "from_numpy", None)):
        return False
    if prefix_count is not None and len(getattr(field, "shape", ())) == 1:
        member_fields = getattr(field, "field_dict", None)
        if isinstance(member_fields, Mapping):
            members = []
            for member, member_field in sorted(member_fields.items()):
                key = _array_key(component, attribute, str(member))
                arrays[key] = field_to_numpy_prefix(member_field, prefix_count)
                members.append(str(member))
            manifest[f"{component}/{attribute}"] = {
                "kind": "struct_field",
                "members": members,
            }
            return True
        try:
            raw = field_to_numpy_prefix(field, prefix_count)
        except (AttributeError, NotImplementedError, RuntimeError, TypeError):
            return False
        arrays[_array_key(component, attribute)] = _owned_array(raw)
        manifest[f"{component}/{attribute}"] = {"kind": "field"}
        return True
    try:
        raw = field.to_numpy()
    except (AttributeError, NotImplementedError, RuntimeError, TypeError):
        return False
    if isinstance(raw, Mapping):
        members = []
        for member, value in sorted(raw.items()):
            if prefix_count is not None and value.ndim > 0:
                value = value[:prefix_count]
            key = _array_key(component, attribute, str(member))
            arrays[key] = _owned_array(value)
            members.append(str(member))
        manifest[f"{component}/{attribute}"] = {
            "kind": "struct_field",
            "members": members,
        }
        return True
    key = _array_key(component, attribute)
    if prefix_count is not None and raw.ndim > 0:
        raw = raw[:prefix_count]
    arrays[key] = _owned_array(raw)
    manifest[f"{component}/{attribute}"] = {"kind": "field"}
    return True


def _capture_object(
    arrays,
    manifest,
    component: str,
    value,
    component_prefixes=None,
) -> None:
    component_prefixes = component_prefixes or {}
    derived_scratch = {
        "fem.assembler": {
            "cell_hessian",
            "cell_stress",
        },
        "fem.soft_contact": {
            "history_state",
            "history_kind",
            "history_stencil",
            "history_overlap",
            "history_overflow",
            "history_entry_count",
        },
        "coupling.neighbor": {
            "history_state",
            "history_key",
            "history_overlap",
            "history_overflow",
            "history_entry_count",
        },
        "coupling.contactor": {
            "wall_history_state",
            "wall_history_key",
            "wall_history_gap",
            "wall_history_overlap",
            "wall_history_overflow",
            "wall_history_entry_count",
        },
    }
    for attribute, field in sorted(vars(value).items()):
        if attribute.startswith("_"):
            continue
        if attribute in derived_scratch.get(component, set()):
            # Rebuild-only hashes are recovered from the compact contact
            # arrays, which already own the persistent tangential overlap.
            continue
        prefix_count = component_prefixes.get(attribute)
        if prefix_count is None and component == "fem.soft_contact":
            if attribute.startswith("pt_"):
                prefix_count = int(value.pt_count)
            elif attribute.startswith("ee_"):
                prefix_count = int(value.ee_count)
        elif prefix_count is None and component == "fem.soft_contact_candidates":
            if attribute.startswith("point_triangle"):
                prefix_count = int(value.compact_point_triangle_count)
            elif attribute.startswith("edge_edge"):
                prefix_count = int(value.compact_edge_edge_count)
        elif prefix_count is None and component == "coupling.neighbor" and attribute == "contacts":
            prefix_count = int(value.contact_count)
        elif prefix_count is None and (
            component == "coupling.contactor" and attribute in {"wall_contacts", "wall_candidate_pairs"}
        ):
            prefix_count = int(value.wall_candidate_count)
        if _capture_field(
            arrays,
            manifest,
            component,
            attribute,
            field,
            prefix_count=prefix_count,
        ):
            continue
        if isinstance(field, np.ndarray):
            key = _array_key(component, attribute)
            arrays[key] = _owned_array(field)
            manifest[f"{component}/{attribute}"] = {"kind": "numpy_attribute"}
            continue
        if isinstance(field, (list, tuple)):
            try:
                candidate = np.asarray(field)
            except (TypeError, ValueError):
                continue
            if candidate.dtype != object:
                key = _array_key(component, attribute)
                arrays[key] = _owned_array(candidate)
                manifest[f"{component}/{attribute}"] = {
                    "kind": "sequence_attribute",
                    "sequence_type": type(field).__name__,
                }


def _dem_contact_prefixes(coupling) -> dict[str, dict[str, int]]:
    """Return active DEM contact prefixes needed for an exact restart."""

    dem = coupling.dem
    contactor = getattr(dem, "contactor", None)
    neighbor = getattr(contactor, "neighbor", None)
    if contactor is None or neighbor is None:
        return {}
    scene = dem.scene
    scheme = str(dem.sims.scheme)
    result = {}
    for component, model, particle_contact in (
        ("dem.contact_pp", getattr(contactor, "physpp", None), True),
        ("dem.contact_pw", getattr(contactor, "physpw", None), False),
    ):
        if model is None or getattr(model, "cplist", None) is None:
            continue
        if scheme == "DEM":
            total = int(scene.particleNum[0])
            prefix = getattr(
                neighbor,
                "hist_particle_particle" if particle_contact else "hist_particle_wall",
                None,
            )
        else:
            total = int(model.get_ls_contact_node_num(scene))
            prefix = getattr(
                neighbor,
                "hist_lsparticle_lsparticle" if particle_contact else "hist_lsparticle_wall",
                None,
            )
        if prefix is None:
            continue
        count = int(prefix[total])
        result[component] = {
            "cplist": count,
            "hist_cplist": count,
        }
    return result


def _scalar_state(coupling) -> dict[str, Any]:
    fem_engine = coupling.fem.engine
    dem_neighbor = getattr(getattr(coupling.dem, "contactor", None), "neighbor", None)
    cross_neighbor = getattr(coupling.contactor, "neighbor", None)
    soft_contact = getattr(fem_engine, "soft_particle_contact", None)
    return {
        "coupling.current_time": float(coupling.sims.current_time),
        "coupling.current_step": int(coupling.sims.current_step),
        "coupling.current_print": int(coupling.sims.current_print),
        "coupling.final_time": float(coupling.sims.time),
        "coupling.delta": float(coupling.sims.delta),
        "coupling.removed_wall_contact_energy": float(getattr(coupling, "removed_wall_contact_energy", 0.0)),
        "fem.time": float(fem_engine.time),
        "fem.step_count": int(fem_engine.step_count),
        "fem.total_step": int(fem_engine.total_step),
        "dem.current_time": float(coupling.dem.sims.current_time),
        "dem.current_step": int(coupling.dem.sims.current_step),
        "dem.current_print": int(coupling.dem.sims.current_print),
        "cross.contact_count": int(getattr(cross_neighbor, "contact_count", 0)),
        "cross.initialized": bool(coupling.contactor.initialized),
        "cross.wall_candidate_count": int(getattr(coupling.contactor, "wall_candidate_count", 0)),
        "engine.minimum_jacobian": float(getattr(coupling.enginer, "minimum_jacobian", 1.0)),
        "solver.last_save_time": float(getattr(coupling.solver, "last_save_time", coupling.sims.current_time)),
        "dem_neighbor.first_run": bool(getattr(dem_neighbor, "first_run", False)),
        "fem.soft_contact.pt_count": int(getattr(soft_contact, "pt_count", 0)),
        "fem.soft_contact.ee_count": int(getattr(soft_contact, "ee_count", 0)),
    }


def _metadata(coupling, manifest) -> dict[str, Any]:
    state = coupling.fem.engine.state
    scene = coupling.dem.scene
    dem_counts = {}
    for name in (
        "particleNum",
        "rigidNum",
        "surfaceNum",
        "gridNum",
        "wallNum",
    ):
        value = getattr(scene, name, None)
        if value is not None:
            dem_counts[name] = int(value[0])
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "checkpoint_phase": str(getattr(coupling, "checkpoint_phase", "unspecified")),
        "topology_sha256": _topology_fingerprint(coupling),
        "precision": str(np.dtype(state.numpy_type)),
        "fem_solver": type(coupling.fem.engine).__name__,
        "dem_scheme": str(coupling.dem.sims.scheme),
        "coupling_search": str(coupling.sims.search),
        "contact_model": type(coupling.contactor.model).__name__,
        "fem_nodes": int(coupling.fem.scene.mesh.number_of_nodes),
        "fem_cells": int(coupling.fem.scene.mesh.number_of_cells),
        "surface_facets": int(len(coupling.surface_faces)),
        "max_contact_pairs": int(coupling.sims.max_contact_pairs),
        "dem_counts": dem_counts,
        "scalar_state": _scalar_state(coupling),
        "array_manifest": manifest,
    }


def save_checkpoint(coupling, filename) -> Path:
    """Atomically save the initialized explicit coupled state to one NPZ."""
    if coupling.enginer is None or coupling.fem.engine is None:
        raise RuntimeError("call FEDEM.add_essentials before saving a checkpoint")
    if not coupling.contactor.initialized:
        raise RuntimeError("initialize FEDEM contact before saving a checkpoint")
    filename = Path(filename).expanduser().resolve()
    if filename.suffix.lower() != ".npz":
        filename = filename.with_suffix(".npz")
    filename.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, dict[str, Any]] = {}
    component_prefixes = _dem_contact_prefixes(coupling)
    for component, value in _component_objects(coupling).items():
        _capture_object(
            arrays,
            manifest,
            component,
            value,
            component_prefixes.get(component),
        )
    metadata = _metadata(coupling, manifest)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    temporary = filename.with_name(f".{filename.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, filename)
    finally:
        if temporary.exists():
            temporary.unlink()
    return filename


def _metadata_from_archive(archive) -> dict[str, Any]:
    if "metadata_json" not in archive.files:
        raise ValueError("FEDEM checkpoint has no metadata_json")
    metadata = json.loads(str(archive["metadata_json"].item()))
    if metadata.get("schema") != SCHEMA_NAME:
        raise ValueError("file is not a FEDEM coupled checkpoint")
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported FEDEM checkpoint schema {metadata.get('schema_version')}")
    return metadata


def _validate_metadata(coupling, metadata) -> None:
    expected = _topology_fingerprint(coupling)
    if metadata["topology_sha256"] != expected:
        raise ValueError("checkpoint topology does not match the rebuilt model")
    state = coupling.fem.engine.state
    checks = {
        "precision": str(np.dtype(state.numpy_type)),
        "fem_solver": type(coupling.fem.engine).__name__,
        "dem_scheme": str(coupling.dem.sims.scheme),
        "coupling_search": str(coupling.sims.search),
        "contact_model": type(coupling.contactor.model).__name__,
        "max_contact_pairs": int(coupling.sims.max_contact_pairs),
    }
    for name, expected_value in checks.items():
        if metadata.get(name) != expected_value:
            raise ValueError(
                f"checkpoint {name}={metadata.get(name)!r} does not match "
                f"the current model value {expected_value!r}"
            )
    for name, saved_count in metadata.get("dem_counts", {}).items():
        current = getattr(coupling.dem.scene, name, None)
        if current is None or int(current[0]) != int(saved_count):
            raise ValueError(f"checkpoint DEM count {name}={saved_count} does not match " f"the rebuilt model")


def _restore_array(field, value, logical_name: str) -> None:
    value = _owned_array(value)
    zero_initialized_capacity = logical_name.startswith(
        (
            "fem.soft_contact/",
            "fem.soft_contact_candidates/",
            "coupling.neighbor/contacts/",
            "coupling.contactor/wall_candidate_pairs",
            "dem.contact_pp/cplist/",
            "dem.contact_pp/hist_cplist/",
            "dem.contact_pw/cplist/",
            "dem.contact_pw/hist_cplist/",
        )
    )

    # Taichi exposes storage shape and scalar dtype without allocating a
    # device-to-host staging array.  Infer vector/matrix component dimensions
    # from the saved checkpoint, whose topology fingerprint was validated
    # before this routine is reached.
    current = None
    try:
        from taichi.lang.util import to_numpy_type

        expected_dtype = np.dtype(to_numpy_type(field.dtype))
        storage_shape = tuple(int(extent) for extent in field.shape)
    except (AttributeError, TypeError):
        current = field.to_numpy()
        if isinstance(current, Mapping):
            raise TypeError(f"{logical_name} is a struct field, not an array field")
        expected_dtype = current.dtype
        current_shape = current.shape
    else:
        storage_rank = len(storage_shape)
        if value.ndim < storage_rank:
            raise ValueError(
                f"checkpoint array {logical_name} has "
                f"{value.dtype}{value.shape}, expected storage rank "
                f"{storage_rank}"
            )
        current_shape = storage_shape + value.shape[storage_rank:]

    if expected_dtype != value.dtype:
        raise ValueError(
            f"checkpoint array {logical_name} has {value.dtype}{value.shape}, "
            f"expected {expected_dtype}{current_shape}"
        )
    if current_shape == value.shape:
        field.from_numpy(value)
        return

    compact_cross_contact = (
        logical_name.startswith("coupling.neighbor/contacts/")
        or logical_name == "coupling.contactor/wall_candidate_pairs"
        or logical_name.startswith("fem.soft_contact/pt_")
        or logical_name.startswith("fem.soft_contact/ee_")
        or logical_name.startswith("fem.soft_contact_candidates/point_triangle")
        or logical_name.startswith("fem.soft_contact_candidates/edge_edge")
        or logical_name.startswith("dem.contact_pp/cplist/")
        or logical_name.startswith("dem.contact_pp/hist_cplist/")
        or logical_name.startswith("dem.contact_pw/cplist/")
        or logical_name.startswith("dem.contact_pw/hist_cplist/")
    )
    if (
        compact_cross_contact
        and len(current_shape) == value.ndim
        and all(current_extent <= saved_extent for current_extent, saved_extent in zip(current_shape, value.shape))
    ):
        # Capacity was split from the raw BVH buffer after schema v1. Only
        # the saved active prefix can be used; the post-restore scalar check
        # below rejects a genuinely undersized current compact capacity.
        slices = tuple(slice(0, extent) for extent in current_shape)
        field.from_numpy(np.ascontiguousarray(value[slices]))
        return

    if len(current_shape) != value.ndim or any(
        current_extent < saved_extent for current_extent, saved_extent in zip(current_shape, value.shape)
    ):
        raise ValueError(
            f"checkpoint array {logical_name} has {value.dtype}{value.shape}, "
            f"expected {expected_dtype}{current_shape}"
        )

    if zero_initialized_capacity:
        # Contact-capacity fields are zero-initialized. Constructing the
        # enlarged target directly on the host avoids a device staging copy.
        current = np.zeros(current_shape, dtype=expected_dtype)
    elif current is None:
        # Only capacity growth whose fresh tail may be nonzero needs a device
        # readback. Exact-shape restores above never allocate this staging copy.
        current = field.to_numpy()
        if isinstance(current, Mapping):
            raise TypeError(f"{logical_name} is a struct field, not an array field")
    slices = tuple(slice(0, extent) for extent in value.shape)
    current[slices] = value
    field.from_numpy(current)


def _restore_object(archive, manifest, component: str, value) -> None:
    prefix = f"{component}/"
    for logical_name, description in manifest.items():
        if not logical_name.startswith(prefix):
            continue
        attribute = logical_name[len(prefix) :]
        if attribute in {
            "history_state",
            "history_kind",
            "history_stencil",
            "history_key",
            "history_overlap",
            "history_overflow",
            "history_entry_count",
        } and component in {"fem.soft_contact", "coupling.neighbor"}:
            # Legacy checkpoints stored rebuild-only hash scratch at the raw
            # candidate capacity. Current sparse hashes are reconstructed
            # from the restored per-contact tangential overlaps.
            continue
        if component == "coupling.contactor" and attribute in {
            "wall_contacts",
            "wall_history_state",
            "wall_history_key",
            "wall_history_gap",
            "wall_history_overlap",
            "wall_history_overflow",
            "wall_history_entry_count",
        }:
            # Wall records need a dense-to-compact schema-v1 migration after
            # the candidate-pair prefix and scalar count have been restored.
            continue
        if component == "fem.assembler" and attribute in {
            "cell_hessian",
            "cell_stress",
        }:
            # Element tangents and the obsolete cell-stress cache are derived
            # data. Older checkpoints may contain their former full buffers.
            continue
        if not hasattr(value, attribute):
            raise ValueError(f"current model has no checkpoint field {logical_name}")
        target = getattr(value, attribute)
        kind = description["kind"]
        if kind == "numpy_attribute":
            saved = _owned_array(archive[_array_key(component, attribute)])
            if target.dtype != saved.dtype:
                raise ValueError(f"checkpoint NumPy attribute mismatch: {logical_name}")
            if target.shape == saved.shape:
                target[...] = saved
            elif target.ndim == saved.ndim and all(
                current_extent >= saved_extent for current_extent, saved_extent in zip(target.shape, saved.shape)
            ):
                slices = tuple(slice(0, extent) for extent in saved.shape)
                target[slices] = saved
            else:
                raise ValueError(f"checkpoint NumPy attribute mismatch: {logical_name}")
        elif kind == "sequence_attribute":
            saved = archive[_array_key(component, attribute)].tolist()
            if description["sequence_type"] == "tuple":
                setattr(value, attribute, tuple(saved))
            else:
                target[:] = saved
        elif kind == "field":
            _restore_array(
                target,
                archive[_array_key(component, attribute)],
                logical_name,
            )
        elif kind == "struct_field":
            for member in description["members"]:
                if not hasattr(target, member):
                    raise ValueError(f"current struct has no checkpoint member {logical_name}/{member}")
                _restore_array(
                    getattr(target, member),
                    archive[_array_key(component, attribute, member)],
                    f"{logical_name}/{member}",
                )
        else:
            raise ValueError(f"unknown checkpoint array kind {kind!r}")


def _restore_scalar_state(coupling, state) -> None:
    coupling.sims.current_time = float(state["coupling.current_time"])
    coupling.sims.current_step = int(state["coupling.current_step"])
    coupling.sims.current_print = int(state["coupling.current_print"])
    coupling.sims.time = float(state["coupling.final_time"])
    coupling.sims.delta = float(state["coupling.delta"])
    coupling.removed_wall_contact_energy = float(state.get("coupling.removed_wall_contact_energy", 0.0))
    coupling.sims.dt[None] = coupling.sims.delta
    coupling.fem.engine.time = float(state["fem.time"])
    coupling.fem.engine.step_count = int(state["fem.step_count"])
    coupling.fem.engine.total_step = int(state["fem.total_step"])
    coupling.dem.sims.current_time = float(state["dem.current_time"])
    coupling.dem.sims.current_step = int(state["dem.current_step"])
    coupling.dem.sims.current_print = int(state["dem.current_print"])
    coupling.dem.sims.CurrentTime[None] = coupling.dem.sims.current_time
    cross_neighbor = getattr(coupling.contactor, "neighbor", None)
    cross_contact_count = int(state["cross.contact_count"])
    if cross_neighbor is None:
        if cross_contact_count != 0:
            raise ValueError(
                "checkpoint contains FEM--LSDEM contacts but the rebuilt " "model has no FEM--LSDEM neighbor module"
            )
    else:
        cross_neighbor.contact_count = cross_contact_count
    coupling.contactor.initialized = bool(state["cross.initialized"])
    if "cross.wall_candidate_count" in state:
        coupling.contactor.wall_candidate_count = int(state["cross.wall_candidate_count"])
    coupling.enginer.minimum_jacobian = float(state["engine.minimum_jacobian"])
    if coupling.solver is not None:
        coupling.solver.last_save_time = float(state["solver.last_save_time"])
    dem_neighbor = getattr(getattr(coupling.dem, "contactor", None), "neighbor", None)
    if dem_neighbor is not None:
        dem_neighbor.first_run = bool(state["dem_neighbor.first_run"])
    soft_contact = getattr(coupling.fem.engine, "soft_particle_contact", None)
    if soft_contact is not None and "fem.soft_contact.pt_count" in state:
        soft_contact.pt_count = int(state["fem.soft_contact.pt_count"])
        soft_contact.ee_count = int(state["fem.soft_contact.ee_count"])


def _restore_wall_contacts(
    archive,
    manifest,
    coupling,
    scalar_state,
) -> None:
    logical_name = "coupling.contactor/wall_contacts"
    description = manifest.get(logical_name)
    target = getattr(coupling.contactor, "wall_contacts", None)
    if description is None or target is None:
        return
    if description.get("kind") != "struct_field":
        raise ValueError("checkpoint FEM wall contacts are not a struct field")
    count = int(scalar_state.get("cross.wall_candidate_count", 0))
    capacity = int(coupling.contactor.wall_contact_count)
    if count > capacity:
        raise ValueError("checkpoint FEM wall candidate prefix exceeds the current " "max_fem_wall_pairs capacity")
    pairs = field_to_numpy_prefix(coupling.contactor.wall_candidate_pairs, count)
    dense_count = int(coupling.contactor.wall_dense_pair_count)
    for member in description["members"]:
        saved = _owned_array(archive[_array_key("coupling.contactor", "wall_contacts", member)])
        if saved.shape[0] == dense_count:
            compact = saved[pairs]
        else:
            compact = saved[:count]
        member_field = getattr(target, member)
        storage_shape = tuple(int(extent) for extent in member_field.shape)
        current_shape = storage_shape + compact.shape[1:]
        values = np.zeros(current_shape, dtype=compact.dtype)
        values[:count] = compact
        member_field.from_numpy(values)


def load_checkpoint(coupling, filename) -> dict[str, Any]:
    """Restore an initialized model and return validated checkpoint metadata."""
    if coupling.enginer is None or coupling.fem.engine is None:
        raise RuntimeError("call FEDEM.add_essentials before loading a checkpoint")
    if not coupling.contactor.initialized:
        coupling.enginer.pre_calculate()
    filename = Path(filename).expanduser().resolve()
    with np.load(filename, allow_pickle=False) as archive:
        metadata = _metadata_from_archive(archive)
        _validate_metadata(coupling, metadata)
        objects = _component_objects(coupling)
        manifest = metadata["array_manifest"]
        for component, value in objects.items():
            _restore_object(archive, manifest, component, value)
        scalar_state = metadata["scalar_state"]
        _restore_scalar_state(coupling, scalar_state)
        _restore_wall_contacts(
            archive,
            manifest,
            coupling,
            scalar_state,
        )
        soft_contact = getattr(coupling.fem.engine, "soft_particle_contact", None)
        if soft_contact is not None:
            has_counts = all(
                name in scalar_state
                for name in (
                    "fem.soft_contact.pt_count",
                    "fem.soft_contact.ee_count",
                )
            )
            has_stencils = all(
                name in manifest
                for name in (
                    "fem.soft_contact_candidates/point_triangle",
                    "fem.soft_contact_candidates/edge_edge",
                )
            )
            if has_counts != has_stencils:
                raise ValueError("checkpoint has an incomplete FEM soft-contact candidate state")
            if not has_counts:
                # Schema-v1 migration path for checkpoints that saved force
                # and history arrays but omitted derived culling stencils and
                # their Python prefix counts.
                soft_contact.rebuild_restart_candidates(coupling.fem.engine.position_field)
            else:
                soft_contact.rebuild_point_triangle_ranges()
            if int(soft_contact.pt_count) > int(soft_contact.pt_capacity) or int(soft_contact.ee_count) > int(
                soft_contact.ee_capacity
            ):
                raise ValueError("checkpoint FEM--FEM contact prefix exceeds the current " "filtered PT/EE capacity")
        cross_neighbor = getattr(coupling.contactor, "neighbor", None)
        if cross_neighbor is not None and hasattr(cross_neighbor, "contact_capacity"):
            if int(cross_neighbor.contact_count) > int(cross_neighbor.contact_capacity):
                raise ValueError(
                    "checkpoint FEM--LSDEM contact prefix exceeds the " "current max_filtered_contact_pairs capacity"
                )
        wall_candidate_state = (
            "coupling.contactor/wall_candidate_pairs",
            "coupling.contactor/wall_candidate_count_device",
            "coupling.contactor/wall_search_nodes",
        )
        has_wall_candidates = (
            all(name in manifest for name in wall_candidate_state) and "cross.wall_candidate_count" in scalar_state
        )
        if not has_wall_candidates:
            # Migration path for checkpoints written before the compact
            # FEM--facet wall broad-phase state was archived.
            coupling.contactor.rebuild_wall_candidates(coupling.dem.scene.wall)
    return metadata


__all__ = [
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "load_checkpoint",
    "save_checkpoint",
]
