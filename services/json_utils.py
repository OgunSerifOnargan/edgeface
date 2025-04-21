from json import load
import os
def read_json(json_file_path):
    with open(json_file_path, "r") as file:
        config = load(file)
    return config


# Dosyaların değişim zamanlarını saklamak için bir sözlük
last_modified_times = {}

def check_file_changed(CONFIG_FILE):
    """Dosyanın değişip değişmediğini kontrol et"""
    global last_modified_times
    try:
        # Dosyanın son değişim zamanını al
        current_modified_time = os.path.getmtime(CONFIG_FILE)

        # Dosya daha önce izlenmemişse, son değişim zamanını kaydet
        if CONFIG_FILE not in last_modified_times:
            last_modified_times[CONFIG_FILE] = current_modified_time
            return False

        # Eğer dosya değiştiyse, değişim zamanını güncelle
        if current_modified_time > last_modified_times[CONFIG_FILE]:
            last_modified_times[CONFIG_FILE] = current_modified_time
            return True  # Dosya değişti

    except FileNotFoundError:
        print(f"{CONFIG_FILE} bulunamadı.")
        return False
    
    return False