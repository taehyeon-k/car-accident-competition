from __future__ import annotations

from fractions import Fraction

import av
import numpy as np

from stage3.data.timing import decode_dacon_stage3_video, decode_external_training_video, nearest_grid_indices


def make_video(path, count=12, fps=20):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(count):
            image = np.full((48, 64, 3), i * 10, np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_external_20hz_resamples_to_10hz(tmp_path):
    path = tmp_path / "twenty.mp4"
    make_video(path, count=12, fps=20)
    decoded = decode_external_training_video(path, max_selection_error=0.03)
    assert len(decoded.frames) == 6
    assert np.allclose(np.diff(decoded.target_times), 0.1)
    assert decoded.valid.all()


def test_vfr_nearest_grid():
    pts = np.asarray([0.0, 0.049, 0.101, 0.151, 0.198, 0.252, 0.301])
    indices, target, valid = nearest_grid_indices(pts, hz=10, max_error=0.01)
    assert indices.tolist() == [0, 2, 4, 6]
    assert np.allclose(target, [0, 0.1, 0.2, 0.3])
    assert valid.all()


def test_dacon_decodes_every_frame(tmp_path):
    path = tmp_path / "dacon.mp4"
    make_video(path, count=9, fps=20)
    decoded = decode_dacon_stage3_video(path)
    assert len(decoded.frames) == 9
    assert decoded.source_indices.tolist() == list(range(9))
    assert np.allclose(decoded.target_times, np.arange(9) * 0.1)
