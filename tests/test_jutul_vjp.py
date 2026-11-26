import numpy as np
import pandas as pd

from test_jutul_darcy import REPORT_DATES, _copy_case_folder, _tiny_folder
from subsurface.multphaseflow.jutul_darcy import JutulDarcy


def test_weighted_rates_match_forward_difference(tmp_path, monkeypatch):
    case = _copy_case_folder(tmp_path, _tiny_folder(), "tiny")
    monkeypatch.chdir(case)
    sim = JutulDarcy({"runfile": "RUNFILE.mako", "reportpoint": REPORT_DATES, "reporttype": "dates",
                      "datatype": ["WOPR:PRO2", "WWPR:PRO1"], "perm_copied": True})
    x = np.log(np.load("PERMX.npy"))
    direction = np.zeros_like(x)
    direction[94] = 1
    assert x[94] + .01 < 6  # differentiating the interior of this clipped template
    labels = [REPORT_DATES[-1], REPORT_DATES[2]]
    weights = pd.DataFrame({"WWPR:PRO1": [None, np.array([.2, -.7])],
                            "WOPR:PRO2": [np.array([1., .5]), None]}, index=labels)
    g = sim.run_vjp({"log_permx": x}, weights)["log_permx"]

    def f(y):
        frame = sim.run_fwd_sim({"log_permx": y})
        return frame.at[labels[0], "WOPR:PRO2"] * np.array([1., .5]) + frame.at[labels[1], "WWPR:PRO1"] * np.array([.2, -.7])

    fd = (f(x + .01*direction) - f(x - .01*direction)) / .02
    assert np.linalg.norm(fd) > 1e-4
    np.testing.assert_allclose(g @ direction, fd, rtol=.05, atol=1e-4)
