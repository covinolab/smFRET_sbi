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
import yaml
from tqdm import tqdm


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
        description="Merge shard HDF5 files produced by generate_simulations.py into one data.h5 file."
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
    return parser.parse_args()


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


def inspect_shards(files: list[Path]) -> tuple[list[ShardInfo], DatasetLayout, DatasetLayout]:
    shard_infos: list[ShardInfo] = []
    theta_layout: DatasetLayout | None = None
    x_layout: DatasetLayout | None = None

    for path in files:
        with h5py.File(path, "r") as f:
            if "theta" not in f or "x" not in f:
                raise ValueError(f"{path} does not contain top-level theta and x datasets")

            theta = f["theta"]
            x = f["x"]

            if theta.ndim != 2:
                raise ValueError(f"{path} theta dataset must be 2D, got shape {theta.shape}")
            if x.ndim < 2:
                raise ValueError(f"{path} x dataset must have sample axis plus feature dims, got {x.shape}")
            if len(theta) != len(x):
                raise ValueError(f"{path} has mismatched theta/x lengths: {len(theta)} vs {len(x)}")

            current_theta_layout = DatasetLayout.from_dataset(theta)
            current_x_layout = DatasetLayout.from_dataset(x)

            if theta_layout is None:
                theta_layout = current_theta_layout
                x_layout = current_x_layout
            else:
                if current_theta_layout.tail_shape != theta_layout.tail_shape:
                    raise ValueError(
                        f"Theta shape mismatch in {path}: {current_theta_layout.tail_shape} "
                        f"vs {theta_layout.tail_shape}"
                    )
                if current_x_layout.tail_shape != x_layout.tail_shape:
                    raise ValueError(
                        f"X shape mismatch in {path}: {current_x_layout.tail_shape} "
                        f"vs {x_layout.tail_shape}"
                    )
                if current_theta_layout.dtype != theta_layout.dtype:
                    raise ValueError(
                        f"Theta dtype mismatch in {path}: {current_theta_layout.dtype} "
                        f"vs {theta_layout.dtype}"
                    )
                if current_x_layout.dtype != x_layout.dtype:
                    raise ValueError(
                        f"X dtype mismatch in {path}: {current_x_layout.dtype} vs {x_layout.dtype}"
                    )

            shard_infos.append(ShardInfo(path=path, num_samples=len(theta)))

    assert theta_layout is not None
    assert x_layout is not None
    return shard_infos, theta_layout, x_layout


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
    theta_layout: DatasetLayout,
    x_layout: DatasetLayout,
    compression_override: int | None,
) -> int:
    n_total = total_samples(shard_infos)
    theta_dim = theta_layout.tail_shape[0]

    with h5py.File(output_path, "w", libver="latest") as out:
        theta_ds = out.create_dataset(
            "theta",
            shape=(n_total, theta_dim),
            maxshape=(n_total, theta_dim),
            **create_dataset_kwargs(theta_layout, n_total, compression_override),
        )
        x_ds = out.create_dataset(
            "x",
            shape=(n_total,) + x_layout.tail_shape,
            maxshape=(n_total,) + x_layout.tail_shape,
            **create_dataset_kwargs(x_layout, n_total, compression_override),
        )

        offset = 0
        with tqdm(total=len(shard_infos), unit="shard", desc="Merging shards") as pbar:
            for shard in shard_infos:
                with h5py.File(shard.path, "r") as src:
                    next_offset = offset + shard.num_samples
                    theta_ds[offset:next_offset] = src["theta"][:]
                    x_ds[offset:next_offset] = src["x"][:]
                    offset = next_offset

                pbar.update(1)
                pbar.set_postfix_str(f"{shard.path.parent.name} | samples={offset}/{n_total}")

        out.attrs["write_idx"] = offset
        out.attrs["num_simulations"] = offset
        out.attrs["generated_at"] = datetime.now().isoformat()
        out.attrs["theta_shape"] = list(theta_ds.shape)
        out.attrs["x_shape"] = list(x_ds.shape)
        out.attrs["source_shards"] = [str(shard.path) for shard in shard_infos]

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

    shard_infos, theta_layout, x_layout = inspect_shards(shard_files)
    validate_configs(shard_infos)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = merge_shards(
        shard_infos=shard_infos,
        output_path=output_path,
        theta_layout=theta_layout,
        x_layout=x_layout,
        compression_override=args.compression_opts,
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
