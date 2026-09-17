from __future__ import annotations

from fractions import Fraction

import av
import numpy as np

from stage3.scripts.build_manifest import video_end_time


def make_video(path, count: int = 100, fps: int = 20) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.time_base = Fraction(1, fps)
        for index in range(count):
            image = np.full((48, 64, 3), index % 255, np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_video_end_time_caps_signal_duration(tmp_path) -> None:
    path = tmp_path / "shorter-than-signals.mp4"
    make_video(path)

    signal_end = 12.0
    duration = min(signal_end, video_end_time(path))

    starts = np.arange(0.0, duration, 3.0)
    assert duration == 5.0
    assert starts.tolist() == [0.0, 3.0]
    assert starts[-1] < video_end_time(path)
