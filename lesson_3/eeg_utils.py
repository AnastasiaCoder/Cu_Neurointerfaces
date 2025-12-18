"""
eeg_utils.py

Набор утилит для обработки ЭЭГ-данных в реальном времени с использованием библиотеки MNE.
Предназначен для буферизации потоковых данных, преобразования в формат MNE, вычисления спектральной плотности мощности (PSD)
и интеграции мощности в заданном частотном диапазоне.

Включает поддержку каузальной (real-time) фильтрации с сохранением состояния фильтра.
"""

import numpy as np
import mne
from mne.time_frequency import psd_array_welch
from scipy.signal import butter, sosfilt, sosfilt_zi

SAMPLE_RATE = 250.0


class RealTimeFilter:
    """
    Каузальный (real-time) IIR-фильтр с сохранением состояния между вызовами.
    Поддерживает полосовой, нижний и верхний срез.
    
    Пример:
        rt_filter = RealTimeFilter(sfreq=250, l_freq=7, h_freq=30, n_channels=2)
        filtered_block = rt_filter.filter_block(raw_block)
    """

    def __init__(self, sfreq: float, l_freq: float = None, h_freq: float = None, n_channels: int = 1, order: int = 4):
        """
        Инициализирует фильтр.

        Параметры
        ----------
        sfreq : float
            Частота дискретизации (Гц).
        l_freq : float, optional
            Нижняя граница частоты (для высокочастотного или полосового фильтра).
        h_freq : float, optional
            Верхняя граница частоты (для низкочастотного или полосового фильтра).
        n_channels : int
            Количество каналов для фильтрации.
        order : int
            Порядок фильтра Баттерворта (по умолчанию 4).
        """
        self.sfreq = sfreq
        self.n_channels = n_channels
        self.order = order

        # Определяем тип фильтра
        if l_freq is not None and h_freq is not None:
            btype = 'bandpass'
            freqs = [l_freq, h_freq]
        elif l_freq is not None:
            btype = 'highpass'
            freqs = l_freq
        elif h_freq is not None:
            btype = 'lowpass'
            freqs = h_freq
        else:
            raise ValueError("Укажите хотя бы l_freq или h_freq")

        # Создаём фильтр в форме Second-Order Sections (SOS) — устойчиво для IIR
        self.sos = butter(order, freqs, btype=btype, fs=sfreq, output='sos')
        # Инициализируем состояние фильтра для каждого канала
        self.zi = [sosfilt_zi(self.sos) for _ in range(n_channels)]

    def filter_block(self, block: np.ndarray):
        """
        Фильтрует блок данных каузально (без использования будущих отсчётов).

        Параметры
        ----------
        block : np.ndarray
            Массив формы (n_channels, n_samples)

        Возвращает
        ----------
        np.ndarray
            Отфильтрованный массив той же формы.
        """
        n_ch, n_s = block.shape
        assert n_ch == self.n_channels, f"Ожидалось {self.n_channels} каналов, получено {n_ch}"
        
        filtered = np.zeros_like(block)
        for ch in range(n_ch):
            filtered[ch], self.zi[ch] = sosfilt(self.sos, block[ch], zi=self.zi[ch])
        return filtered

    def reset(self):
        """Сбрасывает состояние фильтра (например, при старте новой записи)."""
        self.zi = [sosfilt_zi(self.sos) for _ in range(self.n_channels)]


class RingBuffer:
    """Кольцевой (циклический) буфер для хранения последних N отсчётов ЭЭГ-сигнала по нескольким каналам.

    Позволяет эффективно накапливать данные в скользящем окне фиксированной длины.
    При переполнении старые отсчёты автоматически перезаписываются новыми.
    Обеспечивает хронологический порядок данных при извлечении, даже если буфер "завёрнут".
    """

    def __init__(self, n_channels: int, maxlen: int):
        """Инициализирует кольцевой буфер.

        Параметры
        ----------
        n_channels : int
            Количество каналов ЭЭГ.
        maxlen : int
            Максимальное количество отсчётов, которое может хранить буфер (длина окна в отсчётах).
        """
        self.n_channels = n_channels
        self.maxlen = int(maxlen)
        self.data = np.zeros((n_channels, self.maxlen), dtype=float)
        self.idx = 0      # Индекс следующей позиции для записи
        self.count = 0    # Фактическое количество записанных отсчётов (<= maxlen)

    def append_block(self, block: np.ndarray):
        """Добавляет блок новых ЭЭГ-данных в буфер.

        Параметры
        ----------
        block : np.ndarray
            Массив данных формы (n_channels, n_samples), где каждый столбец — новый отсчёт по всем каналам.
        """
        n_ch, n_s = block.shape
        assert n_ch == self.n_channels, "Число каналов в переданном блоке должно совпадать с числом каналов буфера."
        for i in range(n_s):
            self.data[:, self.idx] = block[:, i]
            self.idx = (self.idx + 1) % self.maxlen
            self.count = min(self.maxlen, self.count + 1)

    def get(self):
        """Возвращает все накопленные данные в виде непрерывного массива в хронологическом порядке.

        Возвращает
        ----------
        np.ndarray
            Массив формы (n_channels, n_samples_present), где n_samples_present = min(count, maxlen).
            Данные упорядочены от самого старого к самому новому отсчёту.
        """
        if self.count < self.maxlen:
            # Буфер ещё не заполнен — возвращаем только часть с данными
            return self.data[:, :self.count].copy()
        # Буфер заполнен и "завёрнут": сначала идут данные от idx до конца, затем от начала до idx
        return np.concatenate((self.data[:, self.idx:], self.data[:, :self.idx]), axis=1)


def to_mne_raw(data: np.ndarray, ch_names: list, sfreq: float = SAMPLE_RATE):
    """Преобразует сырые ЭЭГ-данные в объект MNE RawArray для совместимости с экосистемой MNE.

    Параметры
    ----------
    data : np.ndarray
        Двумерный массив ЭЭГ-данных формы (n_channels, n_samples).
    ch_names : list of str
        Список имён каналов (должен совпадать по длине с n_channels).
    sfreq : float, необязательно
        Частота дискретизации данных в герцах. По умолчанию 250.0 Гц.

    Возвращает
    ----------
    mne.io.RawArray
        Объект сырых данных, совместимый с MNE.
    """
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


def compute_psd_mne(data: np.ndarray, sfreq: float = SAMPLE_RATE, fmin=1.0, fmax=50.0, n_fft=None):
    """Вычисляет спектральную плотность мощности (PSD) для каждого ЭЭГ-канала методом Уэлча.

    Параметры
    ----------
    data : np.ndarray
        Двумерный массив формы (n_channels, n_samples).
    sfreq : float, необязательно
        Частота дискретизации. По умолчанию 250.0 Гц.
    fmin : float, необязательно
        Минимальная частота для анализа (в Гц). По умолчанию 1.0 Гц.
    fmax : float, необязательно
        Максимальная частота для анализа (в Гц). По умолчанию 50.0 Гц.
    n_fft : int, необязательно
        Длина окна БПФ. По умолчанию устанавливается на 2-секундное окно (2 * sfreq).

    Возвращает
    ----------
    freqs : np.ndarray
        Массив частот, соответствующих оценке PSD (в Гц).
    psd : np.ndarray
        Массив PSD формы (n_channels, n_freqs), где каждая строка — спектр одного канала.
    """
    n_times = data.shape[1]

    if n_fft is None:
        n_fft = min(int(sfreq * 2), n_times)
    psd_all = []
    for ch in range(data.shape[0]):
        psd, freqs = psd_array_welch(
            data[ch, :], sfreq=sfreq, fmin=fmin, fmax=fmax, n_per_seg=min(250, n_times),n_fft=n_fft, verbose=False
        )
        psd_all.append(psd)
    return freqs, np.vstack(psd_all)


def integrate_band(freqs: np.ndarray, psd: np.ndarray, low: float, high: float):
    """Интегрирует спектральную плотность мощности (PSD) в заданном частотном диапазоне.

    Параметры
    ----------
    freqs : np.ndarray
        Массив частот (в Гц), соответствующих PSD.
    psd : np.ndarray
        PSD-данные: либо одномерный массив (n_freqs,) для одного канала,
        либо двумерный массив (n_channels, n_freqs) для нескольких каналов.
    low : float
        Нижняя граница частотного диапазона (в Гц).
    high : float
        Верхняя граница частотного диапазона (в Гц).

    Возвращает
    ----------
    float или np.ndarray
        Если psd одномерный — возвращает скаляр (мощность для одного канала).
        Если psd двумерный — возвращает массив (мощность для каждого канала).
        Возвращает 0.0 или массив нулей, если диапазон [low, high] выходит за пределы freqs.
    """
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        if psd.ndim == 1:
            return 0.0
        else:
            return np.zeros(psd.shape[0])
    if psd.ndim == 1:
        return float(np.trapz(psd[mask], freqs[mask]))
    else:
        return np.trapz(psd[:, mask], freqs[mask], axis=1)


def bandpower_from_raw_block(block, ch_names, fmin, fmax, sfreq=SAMPLE_RATE, n_fft=None):
    """Вычисляет спектральную плотность мощности (PSD) и интегральную мощность в заданном диапазоне для блока ЭЭГ-данных.

    Параметры
    ----------
    block : np.ndarray
        Двумерный массив ЭЭГ-данных формы (n_channels, n_samples).
    ch_names : list of str
        Список имён каналов (для внутреннего использования, если потребуется MNE-объект).
    fmin : float
        Нижняя граница частотного диапазона для интеграции (в Гц).
    fmax : float
        Верхняя граница частотного диапазона для интеграции (в Гц).
    sfreq : float, необязательно
        Частота дискретизации. По умолчанию 250.0 Гц.
    n_fft : int, необязательно
        Длина окна БПФ для PSD. По умолчанию — 2-секундное окно.

    Возвращает
    ----------
    freqs : np.ndarray
        Массив частот (в Гц), соответствующих PSD.
    psd : np.ndarray
        Массив PSD формы (n_channels, n_freqs).
    bp : np.ndarray
        Массив интегральной мощности формы (n_channels,), по одному значению на канал.
    """
    freqs, psd = compute_psd_mne(block, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft)
    bp = integrate_band(freqs, psd, fmin, fmax)  # Мощность по каждому каналу
    return freqs, psd, bp