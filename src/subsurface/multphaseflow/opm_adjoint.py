"""Serial PET wrapper for Flow builds with adjoint record/replay and the linw objective."""

from contextlib import closing
from datetime import timedelta
from pathlib import Path
import re
import shlex
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from resdata.summary import Summary
from resdata.resfile import ResdataFile

from .opm import flow
from .vjp import scalar_seeds


class FlowAdjoint(flow):
    """Use the same Flow executable, deck renderer and flags for predictions and VJPs."""

    def __init__(self, input_file):
        super().__init__(input_file)
        self.executable = input_file.get("adjoint_executable", "flow_adjoint")
        if self.options["mpi"] or self.options["mpiarray"] or input_file.get("hpc") or self.upscale:
            raise ValueError("FlowAdjoint supports serial, non-upscaled decks only")
        self.flags = shlex.split(self.options["sim_flag"])
        if any(f.startswith(("--adjoint-", "--output-dir", "--enable-storage-cache")) for f in self.flags):
            raise ValueError("FlowAdjoint owns archive, output and storage-cache flags")
        if self.options["parsing-strictness"]:
            self.flags.append("--parsing-strictness=" + self.options["parsing-strictness"])

    def _flow(self, deck, output, *flags):
        output.mkdir(exist_ok=True)
        with (output / "flow.log").open("w+") as log:
            result = subprocess.run([self.executable, str(deck), f"--output-dir={output}",
                                     *self.flags, "--enable-storage-cache=false", *flags],
                                    stdout=log, stderr=subprocess.STDOUT, timeout=self.options.get("sim_limit"))
            if result.returncode:
                log.seek(0, 2)
                log.seek(max(0, log.tell() - 2000))
                raise RuntimeError(f"flow_adjoint exited {result.returncode}: {log.read()}")

    def call_sim(self, folder, wait_for_proc=False):
        folder = Path(folder).resolve()
        try:
            self._flow(folder / (self.file + ".DATA"), folder)
        except (RuntimeError, subprocess.TimeoutExpired):
            return False
        return True

    def run_vjp(self, state, weights):
        cells, seeds = scalar_seeds(weights, self.true_order[1], self.all_data_types)
        supported = {"poro", "permx", "permy", "permz", "log_permx", "log_permy", "log_permz"}
        if not state or set(state) - supported:
            raise ValueError(f"Expected direct active-cell fields from {sorted(supported)}, got {list(state)}")
        if len({key.removeprefix("log_") for key in state}) != len(state):
            raise ValueError("Only one PET variable may control a deck field")
        phases = {"WOPR": "oil", "WWPR": "water", "WGPR": "gas"}
        for _, key in cells:
            kind, _, well = key.partition(":")
            if kind.upper() not in phases or not re.fullmatch(r"[\w.-]+", well):
                raise ValueError(f"Unsupported production rate: {key}")
        gradients = {key: np.zeros((seeds.shape[1], np.size(x))) for key, x in state.items()}
        with TemporaryDirectory(prefix="vjp_", dir=".") as tmp:
            root = Path(tmp).resolve()
            self._runMako(str(root) + "/", dict(state, member=0, level=self.level))
            deck = root / (self.file + ".DATA")
            archive = f"--adjoint-file={root / 'archive'}"
            self._flow(deck, root, archive, "--adjoint-save=true")
            with closing(ResdataFile(str(root / (self.file + ".INIT")))) as init:
                for key, x in state.items():
                    field = key.removeprefix("log_").upper()
                    expected = np.exp(x) if key.startswith("log_") else np.asarray(x)
                    actual = np.asarray(init[field][0])
                    if (expected.shape != actual.shape or not np.all(expected > 0)
                            or not np.allclose(actual, expected, rtol=2e-5, atol=0)):
                        raise ValueError(f"{key} must render directly in active-cell order (check units and clipping)")
            summary = Summary(str(root / self.file))
            dates = pd.DatetimeIndex(summary.dates)
            steps = root / "archive" / "adjoint" / "substep"
            if sum(dates > summary.start_time) != len(list(steps.iterdir())):
                raise ValueError("linw requires a summary for every recorded substep; check summary thinning")
            key_map = {k.upper(): k for k in summary.keys()}
            rows, values = [], []
            for label, key in cells:
                key = key_map[key.upper()]
                date = (pd.Timestamp(summary.start_time) + timedelta(days=float(label))
                        if self.true_order[0] == "days" else pd.Timestamp(label))
                if date != date.normalize() or date not in dates or np.sum(abs(dates - date) <= pd.Timedelta(hours=6)) != 1:
                    raise ValueError(f"linw needs an isolated midnight report at {date} (±6 hour lookup)")
                if summary.unit(key).upper() != "SM3/DAY":
                    raise ValueError(f"Expected metric surface rate for {key}, got {summary.unit(key)}")
                values.append(summary.iget(key, dates.get_loc(date)))
                kind, well = key.split(":")
                rows.append(f"{date:%Y-%m-%d} {well} {phases[kind]}")
            for j, column in enumerate(seeds.T):
                if not column.any():
                    continue
                objective = root / "weights.txt"
                objective.write_text("".join(f"{row} {-86400*w:.17g}\n" for row, w in zip(rows, column) if w))
                with TemporaryDirectory(dir=root) as replay:
                    out = Path(replay)
                    self._flow(deck, out, archive, "--adjoint-mode=gradient", f"--adjoint-objective=linw:{objective}")
                    prefix = out / (self.file + ".ADJOINT_GRADIENTS_PV.txt")
                    with prefix.open() as f:
                        actual = float(f.readline().split("J = ")[1].split(")")[0])
                    # UNSMRY is single precision: scale roundoff by the terms before cancellation.
                    if not np.isclose(actual, column @ values, rtol=2e-5,
                                      atol=1e-8 + 1e-7 * (abs(column) @ np.abs(values))):
                        raise ValueError("linw replay does not match the weighted forward summary")
                    perm = np.loadtxt(out / (self.file + ".ADJOINT_GRADIENTS_PERM.txt"), ndmin=2)
                    for name, x in state.items():
                        x = np.asarray(x)
                        if perm.shape != (len(x), 3):
                            raise ValueError(f"{name} must follow Flow's serial active-cell order")
                        if name == "poro":
                            g = np.loadtxt(prefix, ndmin=1) / x
                        else:
                            axes = "xyz".index(name[-1])
                            g = perm[:, axes] * 9.869233e-16
                            if name.startswith("log_"):
                                g *= np.exp(x)
                        if not np.isfinite(g).all():
                            raise ValueError(f"Nonfinite replay gradient for {name}")
                        gradients[name][j] = g
        return gradients
