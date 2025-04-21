from platform import system
from os.path import abspath, join, dirname, pardir
import sys
from subprocess import check_output
from psutil import cpu_percent, virtual_memory
from time import time, sleep
from gc import collect

# Add parent directory to sys.path to allow importing from parent directory
parent_dir = abspath(join(dirname(__file__), pardir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from services.utils import initialize_logger
from services.json_utils import read_json

# Function to get CPU temperature on Raspberry Pi using vcgencmd
def get_cpu_temperature_rpi():
    try:
        temp_str = check_output(['vcgencmd', 'measure_temp']).decode()
        return float(temp_str.split('=')[1].split('\'')[0])
    except Exception as e:
        return None

# Function to get CPU temperature on macOS using osx-cpu-temp
def get_cpu_temperature_macos():
    try:
        temp_str = check_output(['osx-cpu-temp']).decode()
        return float(temp_str.split()[0])
    except Exception as e:
        return None

# Function to get system metrics
def get_system_metrics():
    cpu_usage = cpu_percent(interval=1)
    memory = virtual_memory()
    ram_usage = memory.percent

    if system() == "Linux":
        cpu_temp = get_cpu_temperature_rpi()
    elif system() == "Darwin":
        cpu_temp = get_cpu_temperature_macos()
    else:
        cpu_temp = None

    return cpu_usage, ram_usage, cpu_temp

# Main function
def monitor_system(stop_event, interval=20):
    config_system = read_json("facefinder/config/system.json")
    loop_counter = 0
    #setproctitle("PYsysMonitor")
    logger = initialize_logger('facefinder/logs/systemLog_rp5')
    if config_system["DEBUG_MODE"]:
        print("raspi_sys_monitor init")
    while not stop_event.is_set():
        loop_counter += 1
        st = time()
        cpu_usage, ram_usage, cpu_temp = get_system_metrics()
        et = time()
        logger.info(f"CPU Usage: {cpu_usage}%")
        logger.info(f"RAM Usage: {ram_usage}%")
        if cpu_temp is not None:
            logger.info(f"CPU Temperature: {cpu_temp}°C")
        else:
            logger.warning("CPU Temperature: Not available")
        sleep(interval)
        if loop_counter >= 100:
            loop_counter = 0
            collect()
