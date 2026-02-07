import imageio
from pathlib import Path


def Frames_to_video(Foldername : str):
    frames = sorted(Path(Foldername).glob("*.png"))

    with imageio.get_writer("output2.mp4", fps=30) as writer:
        for frame in frames:
            writer.append_data(imageio.imread(frame))
    return True
Frames_to_video("output")