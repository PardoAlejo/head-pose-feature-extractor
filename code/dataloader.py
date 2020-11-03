import torch as th
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import ffmpeg
import logging

class FrameLoader(Dataset):
    """Pytorch video loader."""

    def __init__(
            self,
            videos_csv,
            durations_csv,
            framerate=3,
            width=960,
            height=540,
            centercrop=False,
    ):
        """
        Args:
        """
        self.csv = pd.read_csv(videos_csv)
        logging.info(f'Extracting faces for {len(self.csv)} videos')
        self._set_video_duration(durations_csv)
        self.centercrop = centercrop
        self.width = width
        self.height = height
        self.framerate = framerate

    def __len__(self):
        return len(self.csv)

    def _get_video_dim(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        return height, width

    def _get_video_dim(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        return height, width

    def _get_output_dim(self, h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)

    def _set_video_duration(self, durations_csv):
        durations_df = pd.read_csv(durations_csv)
        self.durations = {row[1].videoid:int(row[1].duration) for row in durations_df.iterrows()}

    def __getitem__(self, idx):
        video_path = self.csv['video_path'].values[idx]
        output_file = self.csv['feature_path'].values[idx]

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        duration = self.durations[video_name]*self.framerate
        if not(os.path.isfile(output_file)) and os.path.isfile(video_path):
            # print('Decoding video: {}'.format(video_path))
            try:
                h, w = self._get_video_dim(video_path)
            except:
                print('ffprobe failed at: {}'.format(video_path))
                return th.zeros(1), output_file

            # height, width = self._get_output_dim(self.height, self.width)
            cmd = (
                ffmpeg
                .input(video_path)
                .filter('fps', fps=self.framerate)
                .filter('scale', self.width, self.height)
            )
            out, _ = (
                cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
            )
            video = np.frombuffer(out, np.uint8).reshape([-1, self.height, self.width, 3])
            video = th.from_numpy(video.astype('float32'))[:duration,:,:,:]
        else:
            video = th.zeros(1)
            
        return video, output_file
