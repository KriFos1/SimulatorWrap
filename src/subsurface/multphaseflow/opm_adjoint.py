"""Serial PET wrapper for OPM Flow with opm-adjoint: forecasts, and weighted adjoints through its JSON objectives."""

from contextlib import closing
import json
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

PHASES = {"WOPR": "oil", "WWPR": "water", "WGPR": "gas"}
FIELDS = {"poro", "permx", "permy", "permz", "log_permx", "log_permy", "log_permz"}


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
        """Gradients of weighted sums of producer rates: ``{name: (k, n)}`` for the ``k`` weight columns."""
        cells, seeds = scalar_seeds(weights, self.true_order[1], self.all_data_types)
        if not state or set(state) - FIELDS:
            raise ValueError(f"Expected direct cell fields from {sorted(FIELDS)}, got {list(state)}")
        if len({key.removeprefix("log_") for key in state}) != len(state):
            raise ValueError("Only one PET variable may control a deck field")
        for _, key in cells:
            kind, _, well = key.partition(":")
            if kind.upper() not in PHASES or not re.fullmatch(r"[\w.-]+", well):
                raise ValueError(f"Unsupported production rate: {key}")
        gradients = {key: np.zeros((seeds.shape[1], np.size(x))) for key, x in state.items()}
        with TemporaryDirectory(prefix="vjp_", dir=".") as tmp:
            root = Path(tmp).resolve()
            self._runMako(str(root) + "/", dict(state, member=0, level=self.level))
            deck = root / (self.file + ".DATA")
            archive = f"--adjoint-file={root / 'archive'}"
            self._flow(deck, root, archive, "--adjoint-save=true")
            summary = Summary(str(root / self.file))
            days = np.asarray(summary.days, dtype=float)
            days = days[days > 0]
            if len(days) != len(list((root / "archive" / "adjoint" / "substep").iterdir())):
                raise ValueError("Expected one summary row per recorded substep; check summary thinning")
            key_map = {k.upper(): k for k in summary.keys()}
            start = pd.Timestamp(summary.start_time)
            terms, rates, steps = [], [], []
            for label, key in cells:
                key = key_map[key.upper()]
                if summary.unit(key).upper() != "SM3/DAY":
                    raise ValueError(f"Expected metric surface rate for {key}, got {summary.unit(key)}")
                time = (float(label) if self.true_order[0] == "days"
                        else (pd.Timestamp(label) - start) / pd.Timedelta(days=1))
                i = int(np.argmin(abs(days - time)))
                step = days[i] - (days[i - 1] if i else 0.0)
                after = days[i + 1] - days[i] if i + 1 < len(days) else step
                # Summary times are single precision: the window must be resolved by them.
                if abs(days[i] - time) > 1e-3 or min(step, after) < 16 * np.spacing(np.float32(days[i])):
                    raise ValueError(f"No simulator step ends on the report at {label}")
                kind, well = key.split(":")
                # A term is active on the substep whose end lies in its window.
                terms.append({"kind": "rate", "wells": [well], "phase": PHASES[kind],
                              "window_days": [days[i] - step / 2, days[i] + after / 2]})
                rates.append(summary.numpy_vector(key)[len(summary.days) - len(days) + i])
                steps.append((i, step * 86400.0))
            rates = np.asarray(rates, dtype=float)

            # A rate term is weight * dt * q, with q the internal rate (m3/s, negative for a
            # producer). Its unit-weight value gives dt of each report's substep in double
            # precision; the summary's time differences only check it.
            values = self._objective_terms(deck, root, archive, terms)
            seconds = {}
            for (i, step), value, rate in zip(steps, values, rates):
                if rate:
                    seconds.setdefault(i, []).append(-86400.0 * value / rate)
            resolution = 86400.0 * 2 * np.spacing(np.float32(days[-1]))
            for i, estimates in seconds.items():
                step = next(step for j, step in steps if j == i)
                if not np.allclose(estimates, step, rtol=1e-3, atol=resolution):
                    raise ValueError("Rate objectives do not match the forward summary (units, sign or timing)")
                seconds[i] = float(np.median(estimates))
            unit = np.array([-86400.0 / seconds.get(i, step) for i, step in steps])

            with closing(ResdataFile(str(root / (self.file + ".INIT")))) as init:
                cells_total = int(np.prod(init["INTEHEAD"][0][8:11]))
                deck_fields = {key: np.asarray(init[key.removeprefix("log_").upper()][0]) for key in state}

            for j, column in enumerate(seeds.T):
                if not column.any():
                    continue
                config = root / "objective.json"
                config.write_text(json.dumps({"schema_version": 1, "objective": {"terms": [
                    dict(term, weight=w * u) for term, w, u in zip(terms, column, unit) if w]}}))
                with TemporaryDirectory(dir=root) as replay:
                    out = Path(replay)
                    self._flow(deck, out, archive, "--adjoint-mode=gradient", f"--adjoint-config={config}")
                    prefix = str(out / self.file) + ".ADJOINT_GRADIENTS"
                    actual = json.loads(Path(prefix + ".json").read_text())["objective"]["value"]
                    # UNSMRY is single precision: scale roundoff by the terms before cancellation.
                    if not np.isclose(actual, column @ rates, rtol=2e-5,
                                      atol=1e-8 + 1e-6 * (abs(column) @ abs(rates))):
                        raise ValueError("Replayed objective does not match the weighted forward summary")
                    poro = np.loadtxt(prefix + "_PORO.csv", delimiter=",", skiprows=1, ndmin=2)
                    perm = np.loadtxt(prefix + "_PERM.csv", delimiter=",", skiprows=1, ndmin=2)
                order = np.argsort(poro[:, 0])
                active = poro[order, 0].astype(int)
                if not np.array_equal(active, np.sort(perm[:, 0]).astype(int)):
                    raise ValueError("Porosity and permeability gradients cover different cells")
                poro, perm = poro[order, 1], perm[np.argsort(perm[:, 0]), 1:] * 9.869233e-16
                for name, x in state.items():
                    x = np.asarray(x, dtype=float)
                    if x.size not in (len(active), cells_total):
                        raise ValueError(f"{name} must have one value per active cell or per grid cell")
                    at = active if x.size == cells_total else slice(None)
                    expected = np.exp(x[at]) if name.startswith("log_") else x[at]
                    if (expected.shape != deck_fields[name].shape or not np.all(expected > 0)
                            or not np.allclose(deck_fields[name], expected, rtol=2e-5, atol=0)):
                        raise ValueError(f"{name} must render directly in grid order (check units and clipping)")
                    g = poro / expected if name == "poro" else perm[:, "xyz".index(name[-1])]
                    if name.startswith("log_"):
                        g = g * expected
                    if not np.isfinite(g).all():
                        raise ValueError(f"Nonfinite replay gradient for {name}")
                    gradients[name][j, at] = g
        return gradients

    def _objective_terms(self, deck, root, archive, terms):
        """Values of unit-weight objective ``terms`` on the recorded trajectory."""
        config = root / "terms.json"
        config.write_text(json.dumps({"schema_version": 1, "objective": {"terms": terms}}))
        with TemporaryDirectory(dir=root) as replay:
            self._flow(deck, Path(replay), archive, "--adjoint-mode=objective", f"--adjoint-config={config}")
            log = (Path(replay) / "flow.log").read_text()
        values = re.findall(r"Adjoint objective term\s+\d+\s+\S+\s+J_term = (\S+)", log)
        if len(terms) == 1:  # the tool lists the terms only when there are several
            values = re.findall(r"Adjoint objective J = (\S+)", log)
        if len(values) != len(terms):
            raise RuntimeError("Could not read the objective terms from the flow_adjoint log")
        return np.array(values, dtype=float)
