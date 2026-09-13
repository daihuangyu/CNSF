from .outputs import FramePrediction

__all__ = ["FramePrediction", "TrackMT3"]


def __getattr__(name: str):
    if name == "TrackMT3":
        from .track_mt3 import TrackMT3

        return TrackMT3
    raise AttributeError(name)
