import numpy as np
import pandas as pd

from accretion import ultimate_fitting as fitting
from accretion import ultimate_fitting_types as fitting_types


def test_basic_fitting_uses_supplied_model_params_as_fit_overrides(monkeypatch):
    captured = {}

    def fake_fitting(**kwargs):
        captured.update(kwargs)
        return {"success": False}

    monkeypatch.setattr(fitting_types, "ultimate_fitting_regularized", fake_fitting)

    initial_params = {"Mdot": 10.0, "Av": 7.0}
    fitting_types.basic_fitting(pd.DataFrame(), initial_params=initial_params)

    assert captured["initial_params"] is initial_params
    assert captured["evaluate_only"] is False


def test_basic_fitting_only_skips_optimizer_when_explicit(monkeypatch):
    captured = {}

    def fake_fitting(**kwargs):
        captured.update(kwargs)
        return {"success": False}

    monkeypatch.setattr(fitting_types, "ultimate_fitting_regularized", fake_fitting)
    fitting_types.basic_fitting(
        pd.DataFrame(),
        initial_params={"Mdot": 10.0, "Av": 7.0},
        evaluate_only=True,
    )

    assert captured["evaluate_only"] is True


def test_evaluate_only_uses_exact_values_without_optimizer(monkeypatch):
    day = pd.DataFrame(
        {
            "Filter": ["J", "H", "K"],
            "Lambda": [1.25, 1.65, 2.20],
            "Flux": [2.0, 2.0, 2.0],
            "Fluxerr": [0.2, 0.2, 0.2],
            "Mag": [1.0, 1.0, 1.0],
            "ZP": [1.0, 1.0, 1.0],
        }
    )
    monkeypatch.setattr(fitting, "get_daily_data", lambda _df, _filters: {123.0: day})
    monkeypatch.setattr(
        fitting,
        "accretion_model",
        lambda frequencies, *_args, **_kwargs: np.log(np.full(len(frequencies), 2.0)),
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("optimizer must not run in evaluate-only mode")

    monkeypatch.setattr(fitting, "minimize", fail_if_called)

    results = fitting.ultimate_fitting_regularized(
        required_filters=("J", "H", "K"),
        fit_filters=("J", "H", "K"),
        df=day,
        initial_params={"Mdot": 10.0, "Av": 7.0},
        evaluate_only=True,
    )

    params = results["daily_params"][123.0]
    assert results["success"] is True
    assert results["fit_info"]["evaluate_only"] is True
    assert results["fit_info"]["optimizer_nfev"] == 0
    assert results["fit_info"]["optimizer_method"] == "fixed parameters (no optimization)"
    assert results["fit_info"]["dof"] == 3
    assert params["logMdot"] == 1.0
    assert params["Av"] == 7.0
    assert params["logMdot_err"] == 0.0
    assert params["Av_err"] == 0.0
