# Отправляет команды на ESP32 с помощью UDP
# Последовательность: вперёд 3 сек - назад 3 сек - стоп

import socket
import time

#  Настройки подключения 
ESP32_IP = "10.33.25.75"  # замените на IP вашей ESP32
UDP_PORT = 9999            # должен совпадать с UDP_PORT в main.py на ESP32
DEFAULT_SPEED = 50         # начальная скорость в процентах (0-100)

#  Создаём UDP-сокет 
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send(cmd):
    """
    Отправляет команду на ESP32.
    Важно: команда должна заканчиваться символом новой строки '\n',
    потому что ESP32 использует .strip() и ожидает строку в формате "F,50\n".
    """
    full_cmd = cmd + "\n"
    sock.sendto(full_cmd.encode('utf-8'), (ESP32_IP, UDP_PORT))

#  Основная последовательность управления 

print("Едем вперёд...")
send("F,20")       # Вперёд на 50% скорости
time.sleep(5)      # Ждём

print("Едем назад...")
send("B,20")       # Назад на 50% скорости
time.sleep(5)      # Ждём 

print("Останавливаемся.")
send("S")          # Стоп

sock.close()