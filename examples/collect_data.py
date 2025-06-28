import multiprocessing
import os
import time

import numpy as np
import pandas as pd

from pyomyo import Myo, emg_mode


def cls():
    os.system("cls" if os.name == "nt" else "clear")


# Myo Setup
q = multiprocessing.Queue()


def worker(q):
    m = Myo(mode=emg_mode.FILTERED)
    m.connect()

    def add_to_queue(emg, movement):
        q.put(emg)

    m.add_emg_handler(add_to_queue)
    m.set_leds([128, 128, 0], [128, 128, 0])
    m.vibrate(1)

    while True:
        m.run()


if __name__ == "__main__":
    p = multiprocessing.Process(target=worker, args=(q,))
    p.start()

    gestures = ["rock", "paper", "point"]  # グー、パー、指差し
    data = []

    try:
        print("Myo armband connected. Starting data collection...")
        for i, gesture in enumerate(gestures):
            print(
                f"\nPrepare for {gesture} pose. Data collection will start in 3 seconds..."
            )
            time.sleep(3)
            print(f"Collecting data for {gesture} for 10 seconds...")
            start_time = time.time()
            while time.time() - start_time < 10:
                if not q.empty():
                    emg = list(q.get())
                    data.append(emg + [i])  # Add label to EMG data
                cls()
            print(f"Finished collecting data for {gesture}.")

        df = pd.DataFrame(data, columns=[f"emg{i}" for i in range(8)] + ["label"])
        df.to_csv("myo_gesture_data.csv", index=False)
        print("\nData collection complete. Saved to myo_gesture_data.csv")

    except KeyboardInterrupt:
        print("\nData collection interrupted.")
    finally:
        p.terminate()
        p.join()
        print("Myo process terminated.")
