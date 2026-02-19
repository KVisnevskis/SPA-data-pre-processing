from pathlib import Path
import numpy as np

from preprocessing.optitrack_raw import load_optitrack_raw_csv


def _pick_existing(repo_root: Path, candidates: list[str]) -> Path:
    for rel in candidates:
        p = repo_root / rel
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exist: {candidates}")


def test_load_optitrack_fixed_orientation_first_row():
    repo_root = Path(__file__).resolve().parents[1]
    path = _pick_existing(
        repo_root,
        [
            "sample_data/sample_optitrack_fixed_orientation.csv",
        ],
    )

    df = load_optitrack_raw_csv(path)
    r0 = df.iloc[0]

    assert np.isclose(r0["Time"], 0.0)

    # Base rotation (quat x,y,z,w)
    assert np.isclose(r0["BR_X"], -0.000036, atol=1e-6)
    assert np.isclose(r0["BR_Y"], -0.000013, atol=1e-6)
    assert np.isclose(r0["BR_Z"], -0.000019, atol=1e-6)
    assert np.isclose(r0["BR_W"], -1.0, atol=1e-6)

    # Base position
    assert np.isclose(r0["BP_X"], 0.252055, atol=1e-6)
    assert np.isclose(r0["BP_Y"], 0.305910, atol=1e-6)
    assert np.isclose(r0["BP_Z"], 0.244279, atol=1e-6)

    # Tip rotation
    assert np.isclose(r0["TR_X"], -0.000674, atol=1e-6)
    assert np.isclose(r0["TR_Y"], -0.000074, atol=1e-6)
    assert np.isclose(r0["TR_Z"], 0.000002, atol=1e-6)
    assert np.isclose(r0["TR_W"], -1.0, atol=1e-6)

    # Tip position
    assert np.isclose(r0["TP_X"], 0.239623, atol=1e-6)
    assert np.isclose(r0["TP_Y"], 0.169286, atol=1e-6)
    assert np.isclose(r0["TP_Z"], 0.263956, atol=1e-6)

    # Sampling sanity: second row time step should be ~1/240 s
    dt = df["Time"].iloc[1] - df["Time"].iloc[0]
    assert np.isclose(dt, 1 / 240, atol=1e-6)
