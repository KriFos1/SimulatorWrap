"""Integration: point OPM_ADJOINT_CASE at doc/tutorials/MODEL_1D_DEBUG of OPM/opm-adjoint."""

import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from subsurface.multphaseflow.opm_adjoint import FlowAdjoint
from misc.structures.layout import DataLayout


# Fixed steps: one substep per report. Adaptive steps: several, of which the last one carries the datum.
@pytest.mark.skipif(not os.environ.get("OPM_ADJOINT_CASE"), reason="requires flow_adjoint and its tutorial deck")
@pytest.mark.parametrize("steps,flags", [((2, 4), "--enable-adaptive-time-stepping=false"), ((40, 80), "")])
def test_weighted_producer_rate_matches_forward_directional_difference(tmp_path, monkeypatch, steps, flags):
    source = Path(os.environ["OPM_ADJOINT_CASE"]).resolve()
    deck = (source / "MODEL_1D_DEBUG.DATA").read_text()
    deck = re.sub(r"INCLUDE\s+(\w+\.txt)\s*/", lambda m: (source / m[1]).read_text(), deck)
    deck = re.sub(r"(?ms)^PERMX\s.*?/", "PERMX\n${' '.join(map(str, permx))} /", deck)
    deck = re.sub(r"(?ms)^PORO\s.*?/", "PORO\n${' '.join(map(str, poro))} /", deck)
    deck = re.sub(r"(?ms)^TSTEP\s.*?/", f"TSTEP\n {steps[0]} {steps[1] - steps[0]} /", deck)
    monkeypatch.chdir(tmp_path)
    Path("TEST.mako").write_text(deck)
    sim = FlowAdjoint({"runfile": "TEST", "datatype": ["wopr:prod"], "reporttype": "days", "reportpoint": list(steps),
                       "adjoint_executable": os.environ.get("FLOW_ADJOINT", "flow_adjoint"),
                       "simoptions": {"sim_flag": f"{flags} --tolerance-cnv=1e-7 --tolerance-mb=1e-9"}})
    sim.setup_fwd_run()
    x, d = np.full(3, 100.), np.array([.2, 1., -.3])
    w = np.array([[1., 0., -.5], [0., 1., 1.]])  # two Jacobian rows and a projected row
    layout = DataLayout.from_frame(pd.DataFrame({"wopr:prod": [0., 0.]}, index=list(steps)))
    p = np.full(3, .1)
    g = sim.run_vjp({"permx": x, "poro": p}, layout.to_frame(w))

    def forward(y, z):
        records = sim.run_fwd_sim({"permx": y, "poro": z}, 0)
        return np.array([r["wopr:prod"] for r in records]) @ w

    delta = .2
    fd = (forward(x + delta*d, p) - forward(x - delta*d, p)) / (2*delta)
    assert np.linalg.norm(fd) > 1e-5
    np.testing.assert_allclose(g["permx"] @ d, fd, rtol=.02, atol=1e-5)
    delta = .0002
    fd = (forward(x, p + delta*d) - forward(x, p - delta*d)) / (2*delta)
    assert np.linalg.norm(fd) > 1e-5
    np.testing.assert_allclose(g["poro"] @ d, fd, rtol=.02, atol=1e-5)


def test_the_solver_of_the_adjoint_is_the_users_choice_and_the_run_mode_is_not():
    options = {"runfile": "TEST", "datatype": ["wopr:prod"], "reporttype": "days", "reportpoint": [2, 4]}
    sim = FlowAdjoint({**options, "simoptions": {"sim_flag": "--adjoint-linear-solver=umfpack"}})
    assert "--adjoint-linear-solver=umfpack" in sim.flags
    with pytest.raises(ValueError, match="FlowAdjoint owns"):
        FlowAdjoint({**options, "simoptions": {"sim_flag": "--adjoint-mode=gradient"}})
