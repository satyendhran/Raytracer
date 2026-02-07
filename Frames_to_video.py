"""
Frame-to-video utility for rendered PNG sequences.

Reads all .png files from a folder, sorts them by name, and writes a
single MP4 file at a fixed frame rate.
"""

import imageio
from pathlib import Path


def Frames_to_video(Foldername: str):
    """
    Create a video from a folder of PNG frames.

    Parameters
    ----------
    Foldername : str
        Directory containing PNG frame files.

    Returns
    -------
    bool
        True on successful write.
    """
    frames = sorted(Path(Foldername).glob("*.png"))

    with imageio.get_writer("output2.mp4", fps=30) as writer:
        for frame in frames:
            writer.append_data(imageio.imread(frame))
    return True
Frames_to_video("output")
