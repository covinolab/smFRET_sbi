#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml
from tqdm import tqdm


DEFAULT_MAX_BATCH_BYTES = 256 * 1024 * 1024

# Schema-v2 (photon-level) datasets produced by generate_simulations.py, in
# canonical order. "theta" is first and acts as the sample-count anchor that
# every other dataset is validated against.
MERGE_DATASETS = ("theta", "interphoton_times", "channels", "lengths")

# Format-describing attributes carried from the source shards into the merged
# output (after validating they agree across shards). n_max_photons in
# particular is read back by data_loaders.H5Dataset.
FORMAT_ATTRS = ("schema_version", "n_max_photons", "num_traces_per_set")


@dataclass(frozen=True)
class DatasetLayout:
    tail_shape: tuple[int, ...]
    dtype: Any
    chunks: tuple[int, ...] | None
    compression: str | None
    compression_opts: Any
    shuffle: bool

    @classmethod
    def from_dataset(cls, dataset: h5py.Dataset) -> "DatasetLayout":
        return cls(
            tail_shape=tuple(dataset.shape[1:]),
            dtype=dataset.dtype,
            chunks=dataset.chunks,
            compression=dataset.compression,
            compression_opts=dataset.compression_opts,
            shuffle=bool(dataset.shuffle),
        )

    def output_chunks(self, total_rows: int) -> tuple[int, ...] | None:
        if self.chunks is None:
            return None
        return (min(self.chunks[0], total_rows),) + self.chunks[1:]


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    num_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge schema-v2 shard HDF5 files (theta / interphoton_times / channels / "
            "lengths) produced by generate_simulations.py into one consolidated data.h5 file."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing shard subdirectories or .h5 files",
    )
    parser.add_argument("output", type=Path, help="Output HDF5 file path")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="Optional output directory to create with config.yaml, metadata.yaml, and data.h5",
    )
    parser.add_argument(
        "--pattern",
        default="data.h5",
        help="Filename pattern to search for when input_dir is a directory tree",
    )
    parser.add_argument(
        "--compression-opts",
        type=int,
        default=None,
        help="Optional gzip compression level override for the merged output",
    )
    parser.add_argument(
        "--max-batch-bytes",
        type=int,
        default=DEFAULT_MAX_BATCH_BYTES,
        help=(
            "Upper bound (in bytes) on the uncompressed size of a single "
            "read/write batch per dataset. Controls peak RAM during merging."
        ),
    )
    return parser.parse_args()


def rows_per_batch(layout: DatasetLayout, max_bytes: int) -> int:
    tail = layout.tail_shape or (1,)
    per_row = layout.dtype.itemsize * int(np.prod(tail))
    return max(1, max_bytes // per_row)


def stream_copy(
    src_ds: h5py.Dataset,
    dst_ds: h5py.Dataset,
    dst_offset: int,
    n_rows: int,
    batch: int,
) -> None:
    for start in range(0, n_rows, batch):
        end = min(start + batch, n_rows)
        dst_ds[dst_offset + start : dst_offset + end] = src_ds[start:end]


def natural_sort_key(path: Path) -> list[Any]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", str(path))]


def find_shards(input_dir: Path, pattern: str) -> list[Path]:
    if input_dir.is_file():
        if input_dir.suffix not in {".h5", ".hdf5"}:
            raise ValueError(f"Input file must be an HDF5 file, got {input_dir}")
        return [input_dir]

    files = [p for p in input_dir.rglob(pattern) if p.suffix in {".h5", ".hdf5"}]
    if not files:
        raise FileNotFoundError(f"No HDF5 shards matching {pattern!r} found under {input_dir}")
    return sorted(files, key=natural_sort_key)


def read_format_attrs(f: h5py.File) -> dict[str, Any]:
    """Read the format-describing attributes, deriving any that are missing.

    generate_simulations.py always writes these, but we fall back to deriving
    ``n_max_photons``/``num_traces_per_set`` from the ``interphoton_times`` shape
    so a shard that omitted them still produces a self-describing merged file.
    """
    attrs: dict[str, Any] = {k: f.attrs.get(k) for k in FORMAT_ATTRS}

    ipt = f["interphoton_times"]
    if attrs["n_max_photons"] is None:
        attrs["n_max_photons"] = int(ipt.shape[-1])
    if attrs["num_traces_per_set"] is None:
        attrs["num_traces_per_set"] = int(ipt.shape[1]) if ipt.ndim == 3 else 1
    return attrs


def inspect_shards(
    files: list[Path],
) -> tuple[list[ShardInfo], dict[str, DatasetLayout], dict[str, Any]]:
    shard_infos: list[ShardInfo] = []
    layouts: dict[str, DatasetLayout] | None = None
    format_attrs: dict[str, Any] | None = None
    first_path: Path | None = None

    for path in files:
        with h5py.File(path, "r") as f:
            if "x" in f or any(name not in f for name in MERGE_DATASETS):
                missing = [name for name in MERGE_DATASETS if name not in f]
                raise ValueError(
                    f"{path} is not a schema-v2 shard "
                    f"(found 'x' dataset: {'x' in f}; missing: {missing or 'none'}). "
                    f"merge_h5_shards.py requires the photon-level datasets "
                    f"{MERGE_DATASETS} produced by the current generate_simulations.py."
                )

            theta = f["theta"]
            if theta.ndim != 2:
                raise ValueError(f"{path} theta dataset must be 2D, got shape {theta.shape}")

            n_samples = len(theta)
            for name in MERGE_DATASETS:
                if len(f[name]) != n_samples:
                    raise ValueError(
                        f"{path} has mismatched theta/{name} lengths: "
                        f"{n_samples} vs {len(f[name])}"
                    )

            current_layouts = {name: DatasetLayout.from_dataset(f[name]) for name in MERGE_DATASETS}
            current_attrs = read_format_attrs(f)

            if layouts is None:
                layouts = current_layouts
                format_attrs = current_attrs
                first_path = path
            else:
                for name in MERGE_DATASETS:
                    cur, ref = current_layouts[name], layouts[name]
                    if cur.tail_shape != ref.tail_shape:
                        raise ValueError(
                            f"{name} shape mismatch in {path}: {cur.tail_shape} "
                            f"vs {ref.tail_shape} (in {first_path})"
                        )
                    if cur.dtype != ref.dtype:
                        raise ValueError(
                            f"{name} dtype mismatch in {path}: {cur.dtype} "
                            f"vs {ref.dtype} (in {first_path})"
                        )
                for key in FORMAT_ATTRS:
                    if current_attrs[key] != format_attrs[key]:
                        raise ValueError(
                            f"Attribute '{key}' mismatch in {path}: {current_attrs[key]} "
                            f"vs {format_attrs[key]} (in {first_path}). "
                            "Refusing to merge shards from incompatible runs."
                        )

            shard_infos.append(ShardInfo(path=path, num_samples=n_samples))

    assert layouts is not None
    assert format_attrs is not None
    return shard_infos, layouts, format_attrs


def validate_configs(shard_infos: list[ShardInfo]) -> None:
    reference_text: str | None = None
    reference_path: Path | None = None

    for shard in shard_infos:
        config_path = shard.path.parent / "config.yaml"
        if not config_path.exists():
            continue

        text = config_path.read_text()
        if reference_text is None:
            reference_text = text
            reference_path = config_path
        elif text != reference_text:
            raise ValueError(
                f"Config mismatch between {reference_path} and {config_path}. "
                "Refusing to merge shards from different configurations."
            )


def total_samples(shard_infos: list[ShardInfo]) -> int:
    return sum(shard.num_samples for shard in shard_infos)


def create_dataset_kwargs(layout: DatasetLayout, total_rows: int, compression_override: int | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "dtype": layout.dtype,
        "chunks": layout.output_chunks(total_rows),
    }

    if layout.compression is not None:
        kwargs["compression"] = layout.compression
        kwargs["compression_opts"] = (
            compression_override if compression_override is not None else layout.compression_opts
        )
        kwargs["shuffle"] = layout.shuffle
    elif layout.shuffle:
        kwargs["shuffle"] = layout.shuffle

    return kwargs


def to_metadata_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def write_bundle_files(
    bundle_dir: Path,
    input_dir: Path,
    output_h5: Path,
    shard_infos: list[ShardInfo],
    n_total: int,
) -> None:
    template_dir = shard_infos[0].path.parent
    config_src = template_dir / "config.yaml"
    metadata_src = template_dir / "metadata.yaml"
    base_dir = input_dir.parent if input_dir.is_dir() else input_dir.parent

    if config_src.exists():
        shutil.copy2(config_src, bundle_dir / "config.yaml")

    metadata = {}
    if metadata_src.exists():
        with metadata_src.open("r") as f:
            metadata = yaml.safe_load(f) or {}

    metadata["generated_at"] = datetime.now().isoformat()
    metadata["num_simulations"] = n_total
    metadata["format"] = "h5"
    metadata["output_file"] = to_metadata_path(output_h5, base_dir)
    metadata["merged_from"] = [to_metadata_path(shard.path, base_dir) for shard in shard_infos]

    overrides = metadata.get("command_line_overrides", {})
    overrides["num_simulations"] = n_total
    overrides["output_dir"] = to_metadata_path(bundle_dir, base_dir)
    metadata["command_line_overrides"] = overrides

    with (bundle_dir / "metadata.yaml").open("w") as f:
        yaml.safe_dump(metadata, f, default_flow_style=False, sort_keys=False)


def merge_shards(
    shard_infos: list[ShardInfo],
    output_path: Path,
    layouts: dict[str, DatasetLayout],
    format_attrs: dict[str, Any],
    compression_override: int | None,
    max_batch_bytes: int,
) -> int:
    n_total = total_samples(shard_infos)
    batches = {name: rows_per_batch(layout, max_batch_bytes) for name, layout in layouts.items()}

    with h5py.File(output_path, "w", libver="latest") as out:
        for name, layout in layouts.items():
            shape = (n_total,) + layout.tail_shape
            out.create_dataset(
                name,
                shape=shape,
                maxshape=shape,
                **create_dataset_kwargs(layout, n_total, compression_override),
            )

        offset = 0
        with tqdm(total=len(shard_infos), unit="shard", desc="Merging shards") as pbar:
            for shard in shard_infos:
                with h5py.File(shard.path, "r") as src:
                    for name in layouts:
                        stream_copy(src[name], out[name], offset, shard.num_samples, batches[name])
                    offset += shard.num_samples

                pbar.update(1)
                pbar.set_postfix_str(f"{shard.path.parent.name} | samples={offset}/{n_total}")

        out.attrs["write_idx"] = offset
        out.attrs["num_simulations"] = offset
        out.attrs["generated_at"] = datetime.now().isoformat()
        out.attrs["source_shards"] = [str(shard.path) for shard in shard_infos]
        for name in layouts:
            out.attrs[f"{name}_shape"] = list(out[name].shape)
        for key in FORMAT_ATTRS:
            if format_attrs.get(key) is not None:
                out.attrs[key] = format_attrs[key]

    return n_total


def main() -> None:
    args = parse_args()
    output_path = args.output
    if args.bundle_dir is not None:
        args.bundle_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.bundle_dir / "data.h5"

    shard_files = [
        path
        for path in find_shards(args.input_dir, args.pattern)
        if path.resolve(strict=False) != output_path.resolve(strict=False)
    ]
    if not shard_files:
        raise FileNotFoundError("No input shard files remain after excluding the output path.")

    shard_infos, layouts, format_attrs = inspect_shards(shard_files)
    validate_configs(shard_infos)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = merge_shards(
        shard_infos=shard_infos,
        output_path=output_path,
        layouts=layouts,
        format_attrs=format_attrs,
        compression_override=args.compression_opts,
        max_batch_bytes=args.max_batch_bytes,
    )

    if args.bundle_dir is not None:
        write_bundle_files(
            bundle_dir=args.bundle_dir,
            input_dir=args.input_dir,
            output_h5=output_path,
            shard_infos=shard_infos,
            n_total=n_total,
        )
        print(f"Wrote merged bundle to {args.bundle_dir}")
    else:
        print(f"Wrote merged file to {output_path}")


if __name__ == "__main__":
    main()
