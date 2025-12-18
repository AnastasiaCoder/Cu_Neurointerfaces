# keyboard_control.py — Управление машинкой по Wi-Fi с клавиатуры
# W — вперёд, S — назад, A — медленнее, D — быстрее, Q — выход
# Автоматически показывает текущую скорость и направление

import socket
import threading
import time
import sys

#  Настройки подключения 
ESP32_IP = "10.33.25.75" # замените на IP вашей ESP32
UDP_PORT = 9999            # должен совпадать с UDP_PORT в main.py на ESP32

#  Глобальные переменные управления 
current_direction = "S"    # "F" = вперёд, "B" = назад, "S" = стоп
current_speed = 50         # начальная скорость в процентах (0–100)
is_running = True          # флаг для остановки программы
status_lock = threading.Lock()  # для безопасного обновления статуса

def send_command():
    """Отправляет текущую команду на ESP32."""
    global current_direction, current_speed
    if current_direction == "S":
        cmd = "S"
    else:
        cmd = f"{current_direction},{current_speed}"
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto((cmd + "\n").encode('utf-8'), (ESP32_IP, UDP_PORT))
    except OSError as e:
        print(f"\n[!] Ошибка отправки: {e}")
    finally:
        sock.close()

def clear_line():
    """Очищает текущую строку в терминале (для обновления статуса)."""
    if sys.platform.startswith('win'):
        # Windows: используем возврат каретки
        print('\r', end='', flush=True)
    else:
        # macOS/Linux: очищаем строку ANSI-кодом
        print('\r\033[K', end='', flush=True)

def update_status():
    """Обновляет строку статуса в реальном времени."""
    with status_lock:
        direction_str = {
            "F": "ВПЕРЁД",
            "B": "НАЗАД",
            "S": "СТОП"
        }[current_direction]
        clear_line()
        print(f"Скорость: {current_speed:3d}% | Направление: {direction_str}", end='', flush=True)

def key_listener():
    """Обрабатывает нажатия клавиш без Enter."""
    global current_direction, current_speed, is_running
    
    print("Управление:")
    print("  W — вперёд     S — назад")
    print("  A — медленнее  D — быстрее")
    print("  Q — выход")
    print("\n" + "="*40)
    
    # Показываем начальный статус
    update_status()

    try:
        # Windows
        import msvcrt
        while is_running:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
                handle_key(key)
    except ImportError:
        # macOS / Linux
        import tty, termios, select
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            while is_running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    handle_key(key)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def handle_key(key):
    """Обрабатывает нажатую клавишу."""
    global current_direction, current_speed, is_running
    
    changed = False
    if key == 'w':
        current_direction = "F"
        changed = True
    elif key == 's':
        current_direction = "B"
        changed = True
    elif key == 'a':
        new_speed = max(0, current_speed - 10)
        if new_speed != current_speed:
            current_speed = new_speed
            changed = True
    elif key == 'd':
        new_speed = min(100, current_speed + 10)
        if new_speed != current_speed:
            current_speed = new_speed
            changed = True
    elif key == 'q':
        current_direction = "S"
        send_command()
        is_running = False
        clear_line()
        print("Выход...")
        return

    if changed:
        send_command()
        update_status()  # обновляем статус сразу после изменения

#  Запуск программы 
if __name__ == "__main__":
    print("Подключение к машинке...")
    send_command()
    print("Готово! Управляйте с клавиатуры.\n")

    # Скрываем курсор (опционально, для красоты)
    if not sys.platform.startswith('win'):
        print('\033[?25l', end='', flush=True)  # скрыть курсор

    # Запускаем обработчик клавиш
    listener_thread = threading.Thread(target=key_listener, daemon=True)
    listener_thread.start()

    try:
        # Keep-alive: отправляем команду каждые 0.8 сек (для watchdog ESP32)
        while is_running:
            time.sleep(0.8)
            send_command()
    except KeyboardInterrupt:
        pass
    finally:
        current_direction = "S"
        send_command()
        # Восстанавливаем курсор
        if not sys.platform.startswith('win'):
            print('\033[?25h', end='', flush=True)
        clear_line()
        print("Остановлено.")