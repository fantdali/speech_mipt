import numpy as np


class MelSpectrogram:
    def __init__(
        self,
        sample_rate: float,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        f_min: float,
        f_max: float,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_len = win_length
        self.hop_len = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        n_frames = 1 + (len(waveform) - self.win_len) // self.hop_len

        frames = np.stack(
            [
                waveform[i * self.hop_len : i * self.hop_len + self.win_len]
                for i in range(n_frames)
            ]
        )

        frames *= np.hanning(self.win_len)

        pad_total = self.n_fft - self.win_len
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        frames = np.pad(frames, ((0, 0), (pad_left, pad_right)))

        frames = np.fft.rfft(frames, self.n_fft)
        power_spec = (np.abs(frames) ** 2) / self.n_fft

        mel_points = np.linspace(
            MelSpectrogram.hz_to_mel(self.f_min),
            MelSpectrogram.hz_to_mel(self.f_max),
            self.n_mels + 2,
        )
        hz_points = MelSpectrogram.mel_to_hz(mel_points)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(
            int
        )

        mel_filters = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        for i in range(1, self.n_mels + 1):
            f_l, f_c, f_r = bin_points[i - 1 : i + 2]

            for k in range(f_l, f_c):
                mel_filters[i - 1, k] = (k - f_l) / (f_c - f_l)
            for k in range(f_c, f_r):
                mel_filters[i - 1, k] = (f_r - k) / (f_r - f_c)

        slaney_norm = 2.0 / (hz_points[2:] - hz_points[:-2])
        mel_filters *= slaney_norm[:, np.newaxis]

        return np.dot(power_spec, mel_filters.T).T

    @staticmethod
    def hz_to_mel(f_hz):
        return 2595.0 * np.log10(1.0 + f_hz / 700.0)

    @staticmethod
    def mel_to_hz(mel):
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    @staticmethod
    def power_to_db(
        power_spec: np.ndarray,
        eps: float = 1e-10,
    ):
        return 10 * np.log10(np.maximum(power_spec, eps) / np.max(power_spec))
