import time
import threading
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import socket

from CapsuleSDK.Capsule import Capsule
from CapsuleSDK.DeviceLocator import DeviceLocator
from CapsuleSDK.DeviceType import DeviceType
from CapsuleSDK.Device import Device
from CapsuleSDK.EEGTimedData import EEGTimedData
from CapsuleSDK.Resistances import Resistances

from eeg_utils import *

# Конфиг
PLATFORM = 'win'
EEG_WINDOW_SECONDS = 4.0
CHANNELS = 2
BUFFER_LEN = int(SAMPLE_RATE * EEG_WINDOW_SECONDS)
TARGET_SERIAL = '821619'

# Настройки машинки и порога 
ESP32_IP = "172.20.10.12"  # замените на IP вашей ESP32
UDP_PORT = 9999            # должен совпадать с main.py на ESP32
ALPHA_THRESHOLD = 0.19e-10  # порог мощности альфа-ритма (подстройте под данные)
CALIBRATION_DURATION = 10.0  # секунд "тишины" при старте

ALPHA_LOW = 8.0   # Гц
ALPHA_HIGH = 12.0 

# UDP-сокет для управления
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_to_esp32(cmd):
    """Отправляет команду на ESP32."""
    try:
        udp_sock.sendto((cmd + "\n").encode(), (ESP32_IP, UDP_PORT))
        print(f"[UDP] Sent: {cmd}")
    except OSError as e:
        print(f"[UDP] Error: {e}")

# Инициализация
device = None
device_locator = None
calibration_start_time = None
is_calibrated = False
current_direction = "S"  # "F", "B", "S"

# Глобальные переменные для графиков
alpha_avg_history = []  # история средней альфа-мощности
time_history = []
progress_value = 0  # 0..100
last_accum_time = None
progress_step_sec = 0.1  # шаг обновления прогресса

# Переменные для импеданса
impedance_values = []  # текущие значения импеданса
impedance_history = []  # история изменений импеданса
impedance_time_history = []  # временные метки для истории

class EventFiredState:
    def __init__(self): self._awake = False
    def is_awake(self): return self._awake
    def set_awake(self): self._awake = True
    def sleep(self): self._awake = False

device_list_event = EventFiredState()
device_conn_event = EventFiredState()
device_eeg_event = EventFiredState()
device_resistances_event = EventFiredState()

ring = RingBuffer(n_channels=CHANNELS, maxlen=BUFFER_LEN)
channel_names = []

rt_filter = RealTimeFilter(sfreq=250, l_freq=7, h_freq=13, n_channels=CHANNELS)

def on_device_list(locator, info, fail_reason):
    global device
    if len(info) == 0:
        print("No devices found in this scan.")
        return
    print(f"Found {len(info)} device(s). Using first device:")
    info0 = info[0]
    print("Serial:", info0.get_serial())
    device = Device(locator, info0.get_serial(), locator.get_lib())
    device_list_event.set_awake()

def on_connection_status_changed(d, status):
    global channel_names
    ch_obj = device.get_channel_names()
    channel_names = [ch_obj.get_name_by_index(i) for i in range(len(ch_obj))]
    print(f"Channel names: {channel_names}")
    device_conn_event.set_awake()

def on_eeg(d, eeg: EEGTimedData):
    global ring
    samples = eeg.get_samples_count()
    ch = eeg.get_channels_count()
    if samples <= 0: return

    block = np.zeros((ch, samples), dtype=float)
    for i in range(samples):
        for c in range(ch):
            block[c, i] = eeg.get_processed_value(c, i)
    
    # Фильтрация — один раз, после сбора блока
    filtered_block = rt_filter.filter_block(block)
    
    if filtered_block.shape[0] >= CHANNELS:
        ring.append_block(filtered_block[:CHANNELS, :])
    else:
        padded = np.zeros((CHANNELS, filtered_block.shape[1]), dtype=float)
        padded[:filtered_block.shape[0], :] = filtered_block
        ring.append_block(padded)
    if not device_eeg_event.is_awake():
        device_eeg_event.set_awake()

# Функция-обработчик для сопротивлений (импедансов) электродов
def on_resistances(resistances_obj: Resistances):
    """Обрабатывает данные о сопротивлении электродов."""
    global impedance_values, impedance_history, impedance_time_history
    
    try:
        # Получаем значения импеданса для каждого канала (в Омах)
        values = [resistances_obj.get_value(i) for i in range(len(resistances_obj))]
        impedance_values = values
        current_time = time.time()
        
        # Сохраняем историю для отображения
        impedance_history.append(values)
        impedance_time_history.append(current_time)
        
        # Ограничиваем размер истории
        MAX_HISTORY = 50
        if len(impedance_history) > MAX_HISTORY:
            impedance_history = impedance_history[-MAX_HISTORY:]
            impedance_time_history = impedance_time_history[-MAX_HISTORY:]
        
        # Выводим информацию в консоль
        print("\n" + "="*60)
        print("ELECTRODE IMPEDANCE (Ω)")
        print("="*60)
        
        for i, (channel_name, imp) in enumerate(zip(channel_names, values)):
            # Оценка качества контакта
            if imp < 5000:
                quality = "EXCELLENT"
                color_code = "\033[92m"  # зеленый
            elif imp < 10000:
                quality = "GOOD"
                color_code = "\033[93m"  # желтый
            elif imp < 30000:
                quality = "ACCEPTABLE"
                color_code = "\033[33m"  # оранжевый
            else:
                quality = "POOR"
                color_code = "\033[91m"  # красный
            
            reset_code = "\033[0m"
            print(f"{channel_name:20s}: {imp:8.0f} Ω {color_code}[{quality}]{reset_code}")
        
        print("="*60)
        
        # Проверяем наличие проблем с контактом
        poor_contacts = [i for i, imp in enumerate(values) if imp > 30000]
        
        # Активируем событие для обновления графика
        device_resistances_event.set_awake()
        
    except Exception as e:
        print(f"Error processing impedance data: {e}")

def non_blocking_cond_wait(wake_event, name, total_sleep_time):
    print(f"Waiting {name} up to {total_sleep_time}s...")
    steps = int(total_sleep_time * 50)
    for _ in range(steps):
        if device_locator is not None:
            try:
                device_locator.update()
            except:
                pass
        if wake_event.is_awake():
            return True
        time.sleep(0.02)
    return False

# Создание 4 графиков + кнопка для импеданса
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(5, 1, height_ratios=[2, 2, 2, 1, 0.5])

ax_eeg = fig.add_subplot(gs[0])
ax_psd = fig.add_subplot(gs[1])
ax_alpha = fig.add_subplot(gs[2])
ax_progress = fig.add_subplot(gs[3])

# 1. График ЭЭГ
lines_eeg = []
for i in range(CHANNELS):
    ln, = ax_eeg.plot([], [], label=f'Ch{i}', lw=1)
    lines_eeg.append(ln)
ax_eeg.set_ylabel("Amplitude (µV)")
ax_eeg.set_title("EEG Channels")
ax_eeg.legend(loc='upper right')
ax_eeg.grid(True)

# 2. График PSD
lines_psd = []
for i in range(CHANNELS):
    ln, = ax_psd.plot([], [], label=f'PSD Ch{i}', lw=1)
    lines_psd.append(ln)
ax_psd.set_xlabel("Frequency (Hz)")
ax_psd.set_ylabel("PSD (µV²/Hz)")
ax_psd.set_title("PSD Channels")
ax_psd.legend(loc='upper right')
ax_psd.grid(True)
ax_psd.set_xlim(0, 40)
ax_psd.set_ylim(0, 1e-10)

# 3. График альфа-мощности
line_alpha_avg, = ax_alpha.plot([], [], 'b-', lw=2, label='Avg Alpha')
thr_line = ax_alpha.axhline(ALPHA_THRESHOLD, color='gray', linestyle='--', 
                           linewidth=1, label='Threshold')
ax_alpha.set_xlabel("Time (s)")
ax_alpha.set_ylabel("Alpha Power (µV²/Hz)")
ax_alpha.set_title("Average Alpha Power (8–12 Hz)")
ax_alpha.set_ylim(0, 1e-10)
ax_alpha.legend(loc='upper right')
ax_alpha.grid(True)

# 4. График прогресса
ax_progress.set_xlim(0, 100)
ax_progress.set_ylim(-0.5, 0.5)
ax_progress.set_xlabel("Progress (%)")
ax_progress.set_ylabel("")
ax_progress.set_title("Alpha > Threshold → +1 every 0.1s (after 10s)")
bar_container = ax_progress.barh([0], [0], height=0.8, color='tab:green', alpha=0.8)
progress_bar = bar_container[0]
ax_progress.set_yticks([])
ax_progress.grid(True, axis='x')

# Функция для отображения графика импеданса
def show_impedance_plot():
    """Показывает график импеданса в отдельном окне."""
    if not impedance_values:
        print("No impedance data available yet.")
        return
    
    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
    
    colors = []
    for imp in impedance_values:
        if imp < 5000:
            colors.append('green')
        elif imp < 10000:
            colors.append('yellow')
        elif imp < 30000:
            colors.append('orange')
        else:
            colors.append('red')
    
    bars = ax_imp.bar(channel_names, impedance_values, color=colors, alpha=0.7)
    ax_imp.axhline(y=5000, color='green', linestyle='--', alpha=0.5, label='Excellent (<5kΩ)')
    ax_imp.axhline(y=10000, color='yellow', linestyle='--', alpha=0.5, label='Good (<10kΩ)')
    ax_imp.axhline(y=30000, color='red', linestyle='--', alpha=0.5, label='Poor (>30kΩ)')
    
    # Добавляем значения на столбцы
    for bar, imp in zip(bars, impedance_values):
        height = bar.get_height()
        ax_imp.text(bar.get_x() + bar.get_width()/2., height + max(impedance_values)*0.02,
                   f'{imp:.0f} Ω', ha='center', va='bottom', rotation=0, fontsize=9)
    
    ax_imp.set_ylabel('Impedance (Ω)')
    ax_imp.set_title('Electrode Impedance - Current Values')
    ax_imp.legend()
    ax_imp.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Показываем график в отдельном окне
    plt.show(block=False)

# Кнопка для проверки импеданса
from matplotlib.widgets import Button
ax_button = plt.axes([0.85, 0.01, 0.14, 0.04])  # [left, bottom, width, height]
btn_impedance = Button(ax_button, 'Show Impedance')

def on_impedance_clicked(event):
    show_impedance_plot()

btn_impedance.on_clicked(on_impedance_clicked)

def update_plot(_):
    global alpha_avg_history, time_history, progress_value, last_accum_time
    global calibration_start_time, is_calibrated, current_direction
    
    try:
        device_locator.update()
    except Exception as e:
        print(f"[SDK] Update error: {e}")
    
    buf = ring.get()
    current_time = time.time()
    
    # Инициализация времени калибровки
    if calibration_start_time is None:
        calibration_start_time = current_time
    
    if buf.shape[1] == 0:
        return lines_eeg + lines_psd + [line_alpha_avg] + [progress_bar]
    
    # Обновление EEG
    t = np.linspace(-EEG_WINDOW_SECONDS, 0, buf.shape[1])
    for i in range(CHANNELS):
        lines_eeg[i].set_data(t, buf[i, :])
        if len(channel_names) > i:
            lines_eeg[i].set_label(channel_names[i])
        else:
            lines_eeg[i].set_label(f'Ch{i}')
    
    all_data_eeg = buf.flatten()
    ymin, ymax = all_data_eeg.min(), all_data_eeg.max()
    if ymin == ymax:
        ymin -= 1e-6; ymax += 1e-6
    pad = 0.1*(ymax - ymin)
    ax_eeg.set_ylim(ymin-pad, ymax+pad)
    ax_eeg.set_xlim(-EEG_WINDOW_SECONDS, 0)
    ax_eeg.legend(loc='upper right')
    
    # Вычисление PSD
    try:
        freqs, psd = compute_psd_mne(
            buf, 
            sfreq=SAMPLE_RATE, 
            fmin=1.0, 
            fmax=50.0, 
            n_fft=int(SAMPLE_RATE * 2)
        )
        
        # Обновление PSD
        num_ch = min(psd.shape[0], CHANNELS)
        for i in range(num_ch):
            lines_psd[i].set_data(freqs, psd[i, :])
            lines_psd[i].set_label(f'{channel_names[i] if i < len(channel_names) else f"Ch{i}"}')
        
        ax_psd.set_ylim(0, 1e-10)
        ax_psd.set_xlim(0, 40)
        ax_psd.legend(loc='upper right')
        
        # Вычисление альфа-мощности
        current_alpha = []
        for i in range(num_ch):
            alpha_pow = integrate_band(freqs, psd[i, :], ALPHA_LOW, ALPHA_HIGH)
            current_alpha.append(alpha_pow)
        # avg_alpha = np.mean(current_alpha) if current_alpha else 0.0
        avg_alpha = current_alpha[0]
        
        # Обновление истории альфа-мощности
        time_history.append(current_time)
        alpha_avg_history.append(avg_alpha)
        
        MAX_HISTORY = 100
        if len(time_history) > MAX_HISTORY:
            time_history[:] = time_history[-MAX_HISTORY:]
            alpha_avg_history[:] = alpha_avg_history[-MAX_HISTORY:]
        
        # Обновление графика альфа-мощности
        if time_history:
            t0 = time_history[0]
            t_rel = [t - t0 for t in time_history]
            line_alpha_avg.set_data(t_rel, alpha_avg_history)
            ax_alpha.set_xlim(t_rel[0], t_rel[-1])
        
        # Проверка калибровки
        elapsed = current_time - calibration_start_time
        if not is_calibrated and elapsed >= CALIBRATION_DURATION:
            is_calibrated = True
            print("Калибровка завершена. Машинка готова к управлению.")
        
        # Управление машинкой ПО ПОРОГУ
        if is_calibrated:
            if avg_alpha > ALPHA_THRESHOLD:
                # Альфа выше порога - машинка едет ВПЕРЕД
                # if current_direction != "F":
                send_to_esp32("F,100")
                current_direction = "F"
            else:
                # Альфа ниже порога - машинка едет НАЗАД
                # if current_direction != "B":
                send_to_esp32("B,100")
                current_direction = "B"
        
        # Прогресс-бар
        if len(time_history) > 0 and (current_time - time_history[0]) >= CALIBRATION_DURATION:
            now = time.time()
            if last_accum_time is None:
                last_accum_time = now
            if now - last_accum_time >= progress_step_sec:
                if avg_alpha > ALPHA_THRESHOLD:
                    progress_value = min(100, progress_value + 1)
                else:
                    progress_value = max(0, progress_value - 1)
                last_accum_time = now
            progress_bar.set_width(progress_value)
        else:
            progress_bar.set_width(0)
        
    except Exception as e:
        print(f"[Plot] Error: {e}")
    
    return lines_eeg + lines_psd + [line_alpha_avg] + [progress_bar]

def main():
    global device_locator, device
    
    if PLATFORM == 'win':
        capsuleLib = Capsule(r'C:\Users\n\Desktop\project\BSc-Neuroscience-and-Neurointerfaces-2024\lesson_2\CapsuleClient.dll')
    else:
        capsuleLib = Capsule('./libCapsuleClient.dylib')
    
    device_locator = DeviceLocator(capsuleLib.get_lib())
    device_locator.set_on_devices_list(on_device_list)
    device_locator.request_devices(device_type=DeviceType.Band, seconds_to_search=10)
    
    if not non_blocking_cond_wait(device_list_event, 'device list', 12):
        print("No device found. Exiting.")
        return
    
    device.set_on_connection_status_changed(on_connection_status_changed)
    device.set_on_eeg(on_eeg)
    # Регистрация обработчика для сопротивлений
    device.set_on_resistances(lambda dev, res: on_resistances(res))
    
    device.connect(bipolarChannels=True)
    
    if not non_blocking_cond_wait(device_conn_event, 'device connection', 40):
        print("Failed to connect.")
        return
    
    device.start()
    print("Device started. Opening plot...")
    
    # Начальная остановка машинки
    send_to_esp32("S")
    
    # Фоновая функция для обновления устройства
    running = True
    def updater():
        while running:
            try:
                device_locator.update()
            except Exception:
                pass
            time.sleep(0.01)
    
    t = threading.Thread(target=updater, daemon=True)
    t.start()
    
    # Создание анимации
    ani = FuncAnimation(fig, update_plot, interval=100, blit=False, cache_frame_data=False)
    
    # Отображение графика
    plt.tight_layout()
    plt.show()
    
    running = False
    send_to_esp32("S")
    udp_sock.close()
    device.stop()
    device.disconnect()
    print("Stopped.")

if __name__ == '__main__':
    main()
