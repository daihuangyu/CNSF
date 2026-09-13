from track_mt3.evaluation import load_operating_point


def test_frozen_operating_points_cover_all_neural_methods():
    path = "configs/evaluation/operating_points.yaml"
    assert {"existence", "tracking"} <= load_operating_point(
        path, "Track-MT3"
    ).keys()
    assert {"existence", "tracking"} <= load_operating_point(
        path, "Track-MT3-CM"
    ).keys()
    assert {
        "candidate",
        "output",
        "existence",
        "confirmation_hits",
        "retention",
    } <= load_operating_point(path, "CNSF").keys()
