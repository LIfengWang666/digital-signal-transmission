"""Regenerate the current signal-transmission project from its bundled source inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from reproduction.check_reproducibility import (
    main as check_reproducibility,
    require_environment,
)
from reproduction.build_pdf import build_pdf
from reproduction.check_degradation import main as check_models
from reproduction.audit_stage2 import main as audit_stage2
from reproduction.check_sampling_identity import main as check_sampling
from reproduction.check_r23_results import main as check_r23
from reproduction.check_r31_results import main as check_r31
from reproduction.check_r32_results import main as check_r32
from reproduction.check_r33_results import main as check_r33
from reproduction.check_r34_results import main as check_r34
from reproduction.check_r34_secondary import main as check_r34_secondary
from reproduction.check_r41_results import main as check_r41
from reproduction.check_r42_results import main as check_r42
from reproduction.prepare_report import main as prepare_report
from reproduction.prepare_report import report_source_digest
from reproduction.prepare_r21_report import main as prepare_r21_report
from reproduction.prepare_r22_report import main as prepare_r22_report
from reproduction.prepare_r23_report import main as prepare_r23_report
from reproduction.prepare_r31_report import main as prepare_r31_report
from reproduction.prepare_r32_report import main as prepare_r32_report
from reproduction.prepare_r33_report import main as prepare_r33_report
from reproduction.prepare_r34_report import main as prepare_r34_report
from reproduction.prepare_r41_report import main as prepare_r41_report
from reproduction.prepare_r42_report import main as prepare_r42_report
from reproduction.prepare_r4_report import main as prepare_r4_report
from reproduction.prepare_integrity_report import prepare_r61
from reproduction.verify_inputs import main as verify_inputs
from reproduction.execution_inputs import (
    begin_execution,
    bind_metric_inputs,
    bind_pilot_inputs,
)
from stage1.r11.characterise_reference import main as measure_r11
from stage1.r12.sampling_rate import main as measure_r12
from stage1.r13.aliasing_experiment import main as measure_r13
from stage1.r14.sampling_report import main as report_r14
from stage2.r21.filtering_approach import main as measure_r21
from stage2.r22.plot_parameters import main as plot_r22
from stage2.r23.validation import main as measure_r23
from stage2.r23.plot_validation import main as plot_r23
from stage3.r31.quantisation import main as measure_r31
from stage3.r31.plot_quantisation import main as plot_r31
from stage3.r32.code_checks import main as check_codes
from stage3.r32.pilot_comparison import main as measure_r32
from stage3.r32.plot_comparison import main as plot_r32
from stage3.r33.error_range import main as measure_r33
from stage3.r33.plot_range import main as plot_r33
from stage3.r34.plot_validation import main as plot_r34
from stage4.r41.evidence import main as measure_r41
from stage4.r42.freeze_validation import main as freeze_r4_validation
from stage4.r43.assessment import main as assess_r43

ROOT = Path(__file__).resolve().parent


def isolated_plot(module: str) -> None:
    """Draw a stage figure without inheriting another stage's Matplotlib style.

    Args:
        module: Submitted plotting module with its own main entry point.

    Returns:
        None. Execute unchanged plotting source in a new Python process.
    """
    subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)


def isolated_parameter_scan() -> None:
    """Reproduce the reviewed R2.2 BLAS reduction with four MKL threads.

    Returns:
        None. Run the unchanged R2.2 source in its validated environment;
        all other numerical modules retain their one-thread setting.
    """
    environment = os.environ.copy()
    environment.update(
        OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4"
    )
    subprocess.run(
        [sys.executable, "-m", "stage2.r22.filter_parameters"],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def main() -> None:
    """Read CLI options, regenerate data/assets, optionally compile, and
    return None.

    Returns:
        None. Write the described outputs or raise on invalid data.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Regenerate data and TeX assets without compiling PDF.",
    )
    parser.add_argument(
        "--r4-workers",
        type=int,
        default=6,
        help="Processes for the fixed1400-bundle R4 study;1 runs serially.",
    )
    parser.add_argument(
        "--r5-workers",
        type=int,
        default=6,
        help="Processes for the fixed R5 pilot/confirmation and replay.",
    )
    parser.add_argument(
        "--tectonic",
        type=Path,
        help="Path to the Tectonic executable if it is not on PATH.",
    )
    args = parser.parse_args()
    require_environment()
    begin_execution()
    verify_inputs()
    measure_r11()
    measure_r12()
    measure_r13()
    report_r14()
    check_sampling()
    check_models()
    measure_r21()
    isolated_parameter_scan()
    plot_r22()
    measure_r23()
    check_r23()
    audit_stage2()
    plot_r23()
    measure_r31()
    check_r31()
    plot_r31()
    check_codes()
    measure_r32()
    check_r32()
    plot_r32()
    measure_r33()
    check_r33()
    plot_r33()
    subprocess.run(
        [sys.executable, "-m", "stage3.r34.coding_gain"],
        cwd=ROOT,
        check=True,
    )
    check_r34()
    check_r34_secondary(ROOT)
    plot_r34()
    bind_metric_inputs()
    measure_r41()
    check_r41()
    isolated_plot("stage4.r41.plot_metrics")
    bind_pilot_inputs()
    subprocess.run(
        [sys.executable, "-m", "stage4.r42.pilot"], cwd=ROOT, check=True
    )
    check_r42()
    freeze_r4_validation()
    isolated_plot("stage4.r42.plot_configurations")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "stage4.r43.evaluation",
            "--workers",
            str(args.r4_workers),
        ],
        cwd=ROOT,
        check=True,
    )
    assess_r43()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "reproduction.check_r4_results",
            "--workers",
            str(args.r4_workers),
        ],
        cwd=ROOT,
        check=True,
    )
    isolated_plot("stage4.r43.plot_evaluation")
    for phase in (0, 1):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "stage5.counterfactual",
                "--phase",
                str(phase),
                "--workers",
                str(args.r5_workers),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "reproduction.check_r5_results",
                "--phase",
                str(phase),
                "--workers",
                str(args.r5_workers),
            ],
            cwd=ROOT,
            check=True,
        )
    subprocess.run(
        [sys.executable, "-m", "stage5.evidence"], cwd=ROOT, check=True
    )
    subprocess.run(
        [sys.executable, "-m", "stage5.constraint"], cwd=ROOT, check=True
    )
    subprocess.run(
        [sys.executable, "-m", "reproduction.check_r52_results"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "stage5.plot_reflection", "--part", "all"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "stage5.packing_diagnostic"],
        cwd=ROOT,
        check=True,
    )
    prepare_report()
    prepare_r21_report()
    prepare_r22_report()
    prepare_r23_report()
    prepare_r31_report()
    prepare_r32_report()
    prepare_r33_report()
    prepare_r34_report()
    prepare_r41_report()
    prepare_r42_report()
    prepare_r4_report()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "reproduction.prepare_r5_report",
            "--part",
            "all",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "stage3.r34.plot_validation",
            "--horizontal",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "reproduction.prepare_publication_figures"],
        cwd=ROOT,
        check=True,
    )
    check_reproducibility()
    prepare_r61()
    if args.data_only:
        return
    compiler = args.tectonic or shutil.which("tectonic")
    if compiler is None:
        local = ROOT.parent / ".tools" / "tectonic" / "tectonic.exe"
        compiler = local if local.exists() else None
    if compiler is None:
        raise FileNotFoundError(
            "Install Tectonic or provide --tectonic PATH; "
            "data and TeX sources are ready."
        )
    outcome = build_pdf(
        Path(compiler).resolve(), ROOT / "report", ROOT / ".build-cache"
    )
    pdf = ROOT / "report" / "main.pdf"
    manifest = {
        "status": "Digital signal transmission report compiled",
        "report_source_sha256": report_source_digest(),
        "compiled_pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "compiler": outcome["compiler"],
        "build_policy": outcome["build_policy"],
    }
    (ROOT / "metadata" / "report_build.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("Report PDF compiled from the supplied LaTeX sources.")


if __name__ == "__main__":
    main()
