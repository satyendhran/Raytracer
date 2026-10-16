import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TIMINGS = (
    "setup_seconds",
    "trace_seconds",
    "filter_and_display_seconds",
    "copy_seconds",
    "save_seconds",
    "render_seconds",
    "total_seconds",
)
VARIANTS = (
    ("random", "random", "final", "pixels"),
    ("sobol", "sobol", "final", "pixels"),
    ("adaptive_tiles", "sobol", "normal", "tiles"),
    ("adaptive_pixels", "sobol", "normal", "pixels"),
)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare sampling and adaptive scheduling against a high-spp reference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scene", default="cornell_model")
    parser.add_argument("--width", type=positive_int, default=128)
    parser.add_argument("--spp", type=positive_int, default=32)
    parser.add_argument("--reference-spp", type=positive_int, default=512)
    parser.add_argument("--light-samples", type=positive_int, default=1)
    parser.add_argument("--nee-max-depth", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("output/quality_comparison"))
    parser.add_argument("--repeats", type=positive_int, default=2)
    parser.add_argument(
        "--warmup", action="store_true",
        help="Run each comparison configuration once before measurements; processes still include startup/compile.",
    )
    parser.add_argument(
        "--display-exposure", type=positive_float, default=None,
        help="Also measure ACES/gamma display metrics using this scene exposure, before PNG quantization.",
    )
    args = parser.parse_args(argv)
    if args.reference_spp < args.spp:
        parser.error("--reference-spp must be at least --spp")
    if args.light_samples > 8 or not 0 <= args.nee_max_depth <= 12:
        parser.error("Light samples must be 1..8 and NEE depth must be 0..12")
    if args.repeats > 9187:
        parser.error("--repeats must be at most 9187 to keep reference seed 9187 independent")
    return args


def load_archive(path):
    arrays = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in ("linear", "filtered", "sample_counts"):
            if name not in archive:
                raise ValueError(f"{path}: missing array {name}")
            value = archive[name]
            if value.dtype.kind not in "iuf" or not value.size:
                raise ValueError(f"{path}: {name} must be a nonempty real numeric array")
            value = value.astype(np.float64)
            if not np.isfinite(value).all():
                raise ValueError(f"{path}: {name} contains nonfinite values")
            arrays[name] = value
    shape = arrays["linear"].shape
    if len(shape) != 3 or shape[-1] != 3:
        raise ValueError(f"{path}: linear must have shape (width, height, 3), got {shape}")
    if arrays["filtered"].shape != shape:
        raise ValueError(f"{path}: filtered shape does not match linear shape")
    counts = arrays["sample_counts"]
    if counts.shape != shape[:2] or np.any(counts < 0):
        raise ValueError(f"{path}: sample_counts must be nonnegative with shape {shape[:2]}")
    return arrays


def tonemap(image, exposure):
    with np.errstate(over="ignore", invalid="ignore"):
        value = image * exposure
    invalid = np.any(~np.isfinite(value) | (np.abs(value) > 1e8), axis=-1)
    value = np.where(invalid[..., None], 0.0, np.clip(value, 0.0, 10000.0))
    mapped = value * (2.51 * value + 0.03)
    mapped /= value * (2.43 * value + 0.59) + 0.14
    return np.clip(mapped, 0.0, 1.0) ** (1.0 / 2.2)


def image_metrics(image, reference):
    if image.shape != reference.shape:
        raise ValueError(f"Image shape {image.shape} differs from reference {reference.shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        mse = float(np.mean(np.square(image - reference)))
    if not math.isfinite(mse):
        raise ValueError("MSE overflowed; image values are too large")
    return {
        "mse": mse,
        "psnr_db": None if mse == 0.0 else -10.0 * math.log10(mse),
        "perfect_match": mse == 0.0,
    }


def compare_images(arrays, reference, exposure):
    metrics = {}
    for name in ("linear", "filtered"):
        metrics[name] = image_metrics(arrays[name], reference)
        if exposure is not None:
            metrics[f"{name}_display"] = image_metrics(
                tonemap(arrays[name], exposure), tonemap(reference, exposure)
            )
    return metrics


def run_render(args, directory, name, sampler, quality, scheduler, spp, seed, denoise):
    stem = directory / name
    paths = {
        "png": str(stem.with_suffix(".png")),
        "npz": str(stem.with_suffix(".npz")),
        "stats": str(stem.with_suffix(".stats.json")),
        "log": str(stem.with_suffix(".log")),
    }
    command = [
        sys.executable, str(ROOT / "main.py"),
        "--scene", args.scene, "--width", str(args.width), "--spp", str(spp),
        "--quality", quality, "--sampler", sampler, "--seed", str(seed),
        "--adaptive-scheduler", scheduler, "--light-samples", str(args.light_samples),
        "--nee-max-depth", str(args.nee_max_depth), "--output", paths["png"],
        "--stats", paths["stats"], "--linear-output", paths["npz"],
    ]
    if denoise:
        command.append("--denoise")
    print(f"Rendering {name}: {spp} spp, seed {seed}", flush=True)
    with open(paths["log"], "w", encoding="utf-8") as log:
        start = perf_counter()
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        wall_seconds = perf_counter() - start
    record = {
        "name": name, "sampler": sampler, "quality": quality,
        "scheduler": scheduler, "requested_spp": spp, "seed": seed,
        "denoise": denoise, "command": command, "artifacts": paths,
        "wall_seconds": wall_seconds, "returncode": result.returncode,
    }
    if result.returncode:
        record["error"] = f"Renderer exited with code {result.returncode}; see {paths['log']}"
        return record, None
    try:
        with open(paths["stats"], encoding="utf-8") as file:
            stats = json.load(file)
        if not isinstance(stats, dict):
            raise ValueError("Renderer stats must be a JSON object")
        for key in (*TIMINGS, "mean_spp", "max_spp"):
            if key in stats:
                value = stats[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"Invalid renderer statistic {key}: {value!r}")
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid renderer statistic {key}: {value!r}")
        record["stats"] = stats
        arrays = load_archive(paths["npz"])
        counts = arrays["sample_counts"]
        record["sample_counts"] = {
            "mean_spp": float(counts.mean()), "max_spp": float(counts.max()),
            "min_spp": float(counts.min()), "total_samples": float(counts.sum()),
        }
        return record, arrays
    except (OSError, ValueError, KeyError, EOFError) as error:
        record["error"] = str(error)
        return record, None


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size), "mean": float(values.mean()),
        "std": float(values.std()), "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize(runs):
    summaries = {}
    for variant, _, _, _ in VARIANTS:
        selected = [run for run in runs if run["variant"] == variant]
        valid = [run for run in selected if "error" not in run]
        summary = {"successful_runs": len(valid), "failed_runs": len(selected) - len(valid)}
        if valid:
            summary["wall_seconds"] = describe([run["wall_seconds"] for run in valid])
            summary["timings"] = {
                key: describe([run["stats"][key] for run in valid if key in run["stats"]])
                for key in TIMINGS if any(key in run["stats"] for run in valid)
            }
            summary["sample_counts"] = {
                key: describe([run["sample_counts"][key] for run in valid])
                for key in valid[0]["sample_counts"]
            }
            summary["metrics"] = {}
            for space in valid[0]["metrics"]:
                measurements = [run["metrics"][space] for run in valid]
                finite_psnr = [item["psnr_db"] for item in measurements if item["psnr_db"] is not None]
                summary["metrics"][space] = {
                    "mse": describe([item["mse"] for item in measurements]),
                    "psnr_db": describe(finite_psnr) if finite_psnr else None,
                    "perfect_matches": sum(item["perfect_match"] for item in measurements),
                }
        summaries[variant] = summary
    return summaries


def write_report(path, report):
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run_", dir=output))
    report_path = output / "report.json"
    report = {
        "configuration": {**vars(args), "output_dir": str(output)},
        "methodology": {
            "wall_seconds": "Full subprocess lifetime, including Python/Taichi startup, compilation or cache loading, rendering, exports and shutdown; warmup does not subtract these costs.",
            "renderer_timings": "Reported separately by main.py; overlapping timings must not be summed and total_seconds excludes process startup.",
            "budget": "Identical requested spp caps; adaptive runs may use fewer actual samples. Actual counts are reported.",
            "reference": "Unfiltered independent-seed Sobol reference, final quality, seed 9187, no denoising.",
            "metrics": "RGB MSE; PSNR = -10 log10(MSE), fixed unit peak in both linear HDR and display space. Linear values are not clipped. Null PSNR with perfect_match means positive infinity; summary PSNR statistics include finite values only.",
            "display": "Optional ACES fit, renderer sanitization and gamma 1/2.2, explicit exposure, no quantization.",
            "orientation": "Native (width, height, 3); no flip or transpose.",
            "warmup": "One unmeasured subprocess per comparison configuration before the reference; not included in summaries.",
        },
        "artifacts_directory": str(directory), "warmups": [], "runs": [],
    }
    started = perf_counter()
    if args.warmup:
        for variant, sampler, quality, scheduler in VARIANTS:
            record, _ = run_render(
                args, directory, f"warmup_{variant}", sampler, quality,
                scheduler, args.spp, -1, True,
            )
            report["warmups"].append(record)
            write_report(report_path, report)
    reference_record, reference_arrays = run_render(
        args, directory, "reference", "sobol", "final", "pixels",
        args.reference_spp, 9187, False,
    )
    report["reference"] = reference_record
    write_report(report_path, report)
    if reference_arrays is not None:
        reference = reference_arrays["linear"]
        for repeat in range(args.repeats):
            for variant, sampler, quality, scheduler in VARIANTS:
                record, arrays = run_render(
                    args, directory, f"{variant}_{repeat}", sampler, quality,
                    scheduler, args.spp, repeat, True,
                )
                record.update(variant=variant, repeat=repeat)
                if arrays is not None:
                    try:
                        record["metrics"] = compare_images(arrays, reference, args.display_exposure)
                    except ValueError as error:
                        record["error"] = str(error)
                report["runs"].append(record)
                write_report(report_path, report)
    report["summaries"] = summarize(report["runs"])
    report["benchmark_wall_seconds"] = perf_counter() - started
    failed = any("error" in run for run in [reference_record, *report["warmups"], *report["runs"]])
    report["status"] = "failed" if failed else "complete"
    write_report(report_path, report)
    print(f"Report: {report_path}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
