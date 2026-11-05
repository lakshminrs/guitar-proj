import numpy as np

from app.analysis import geometry as G
from app.theory import CHORDS, expected_events, get_exercise


def test_fret_positions_equal_temperament():
    assert G.fret_x(0) == 0
    assert abs(G.fret_x(12) - 0.5) < 1e-12           # 12th fret = half the scale length
    assert abs(G.fret_x(24) - 0.75) < 1e-12
    assert np.all(np.diff(G.FRET_X) > 0)
    assert np.all(np.diff(np.diff(G.FRET_X)) < 0)     # frets get closer together


def test_chord_midi_matches_spec():
    assert CHORDS["G"].midi == [43, 47, 50, 55, 59, 67]
    assert CHORDS["D"].midi[:2] == [-1, -1]


def test_barre_targets_collapse_index_finger():
    t = CHORDS["F"].fingering_targets()
    barres = [x for x in t if x["barre"]]
    assert len(barres) == 1 and barres[0]["fret"] == 1
    assert {(x["finger"], x["string"], x["fret"]) for x in t if not x["barre"]} == {(3, 5, 3), (4, 4, 3), (2, 3, 2)}


def _calib(offset=(0.1, 0.3), length=0.7, width=0.12, mirrored=False):
    ox, oy = offset
    nut_x, f12_x = (ox + length, ox + length * 0.4) if mirrored else (ox, ox + length * 0.6)
    return {"nut_low": (nut_x, oy), "nut_high": (nut_x, oy + width),
            "f12_low": (f12_x, oy - 0.01), "f12_high": (f12_x, oy + width + 0.01)}


def test_homography_roundtrip_and_cells():
    H = G.homography_from_points(_calib())
    assert H is not None
    # canonical -> image -> canonical
    pts = np.array([[0.1, 0.2], [0.3, 0.9], [0.45, 0.5]])
    back = G.image_to_canon(H, G.apply_h(H, pts))
    assert np.allclose(back, pts, atol=1e-9)
    # every cell centre maps back to its own (string, fret)
    for s in range(1, 7):
        for f in range(1, 13):
            x, y = G.cell_center(s, f)
            img = G.apply_h(H, [[x, y]])
            assert G.canon_to_cell(*G.image_to_canon(H, img)[0]) == (s, f)


def test_mirroring_detected():
    assert not G.is_mirrored(G.homography_from_points(_calib()))
    assert G.is_mirrored(G.homography_from_points(_calib(mirrored=True)))


def test_expected_events_pattern():
    ex = get_exercise(1)
    evs = expected_events(ex)
    assert len(evs) == 6 * len(ex["bars"])          # D-DU-UDU = 6 strums per bar
    first = evs[:6]
    assert [e.direction for e in first] == list("DDUUDU")
    assert [e.beat_pos for e in first] == [0, 1, 1.5, 2.5, 3, 3.5]
    assert evs[6].is_change and not evs[1].is_change


def test_one_euro_smooths_jitter():
    f = G.OneEuro(min_cutoff=1.0, beta=0.0)
    rng = np.random.default_rng(0)
    raw = 0.5 + rng.normal(0, 0.01, 300)
    out = np.array([f(np.array([v]), i / 30)[0] for i, v in enumerate(raw)])
    assert out[50:].std() < raw[50:].std() * 0.6
