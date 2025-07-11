import multiprocessing
import time
from collections import deque

import numpy as np

from pyomyo import Myo, emg_mode

# ------------ Myo Setup ---------------
q = multiprocessing.Queue()


def worker(q):
    m = Myo(mode=emg_mode.FILTERED)
    m.connect()

    def add_to_queue(emg, movement):
        q.put(emg)

    m.add_emg_handler(add_to_queue)

    def print_battery(bat):
        print("Battery level:", bat)

    m.add_battery_handler(print_battery)

    # Orange logo and bar LEDs
    m.set_leds([128, 0, 0], [128, 0, 0])
    # Vibrate to know we connected okay
    m.vibrate(1)

    """worker function"""
    while True:
        m.run()
    print("Worker Stopped")


# -------- Sample Rate Calculation -----------
def calculate_sample_rate():
    sample_times = deque(maxlen=100)  # Store last 100 timestamps for rate calculation
    last_print_time = time.time()

    while True:
        start_time = time.perf_counter()
        yield

        # Record the time after getting data
        sample_times.append(time.perf_counter())

        # Calculate and print sample rate every second
        current_time = time.time()
        if current_time - last_print_time >= 1.0:
            if len(sample_times) > 1:
                # Calculate average time between samples in the window
                time_diffs = np.diff(sample_times)
                avg_sample_rate = 1.0 / np.mean(time_diffs)
                print(
                    f"\rSample Rate: {avg_sample_rate:.2f} Hz | Samples in window: {len(sample_times)-1}",
                    end="",
                )
            last_print_time = current_time


# -------- Main Program Loop -----------
if __name__ == "__main__":
    p = multiprocessing.Process(target=worker, args=(q,))
    p.start()

    # Initialize sample rate calculator
    rate_calculator = calculate_sample_rate()
    next(rate_calculator)  # Start the generator

    try:
        while True:
            while not q.empty():
                emg = list(q.get())
                # Print EMG data (commented out for cleaner output)
                # print(emg)

                # Update sample rate calculation
                next(rate_calculator)

    except KeyboardInterrupt:
        print("\nQuitting")
        p.terminate()
        p.join()
