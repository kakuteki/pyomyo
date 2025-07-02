"""
Can plot EMG data in 2 different ways
change DRAW_LINES to try each.
Press Ctrl + C in the terminal to exit 
"""

import csv  # Import csv module
import multiprocessing
import time  # Import time module

from pyomyo import Myo, emg_mode

# ------------ Myo Setup ---------------
q = multiprocessing.Queue()


def worker(q):
    m = Myo(mode=emg_mode.PREPROCESSED)
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


# -------- Main Program Loop -----------
if __name__ == "__main__":
    p = multiprocessing.Process(target=worker, args=(q,))
    p.start()

    # Open a CSV file to save the data
    with open("emg_data.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(["Timestamp"] + [f"Sensor_{i+1}" for i in range(8)])

        try:
            while True:
                # Get the emg data and save it with a timestamp
                while not (q.empty()):
                    emg = list(q.get())
                    timestamp = time.time()  # Get the current timestamp
                    writer.writerow([timestamp] + emg)
                    print(f"Timestamp: {timestamp}, Sensor: {', '.join(map(str, emg))}")

        except KeyboardInterrupt:
            print("Quitting")
            quit()
