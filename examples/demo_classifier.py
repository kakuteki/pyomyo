import multiprocessing
import os
import time

import joblib
import numpy as np

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
    # Load the trained model
    try:
        classifier = joblib.load("gesture_classifier.pkl")
        print("Classifier loaded successfully.")
    except FileNotFoundError:
        print(
            "Error: gesture_classifier.pkl not found. Please run train_model.py first."
        )
        exit()

    gestures = ["Rock (グー)", "Paper (パー)", "Point (指差し)"]

    p = multiprocessing.Process(target=worker, args=(q,))
    p.start()

    try:
        print("\nMyo armband connected. Starting real-time gesture prediction...")
        print("Make sure your Myo armband is on and connected.")
        while True:
            if not q.empty():
                emg = np.array(q.get()).reshape(
                    1, -1
                )  # Reshape for single sample prediction
                prediction = classifier.predict(emg)
                predicted_gesture_index = prediction[0]
                predicted_gesture = gestures[predicted_gesture_index]
                cls()
                print(f"Predicted Gesture: {predicted_gesture}")
            time.sleep(0.01)  # Small delay to prevent busy-waiting

    except KeyboardInterrupt:
        print("\nGesture prediction interrupted.")
    finally:
        p.terminate()
        p.join()
        print("Myo process terminated.")
