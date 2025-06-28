import pygame
from pygame.locals import *

from .classifier_base import Classifier
from .enums import emg_mode
from .myo_classifier_integration import MyoClassifier


def text(scr, font, txt, pos, clr=(255, 255, 255)):
    scr.blit(font.render(txt, True, clr), pos)


class EMGHandler(object):
    def __init__(self, m):
        self.recording = -1
        self.m = m
        self.emg = (0,) * 8

    def __call__(self, emg, moving):
        self.emg = emg
        if self.recording >= 0:
            self.m.cls.store_data(self.recording, emg)


def run_gui(myo_classifier_instance, hnd, scr, font, w, h):
    # Handle keypresses
    for ev in pygame.event.get():
        if ev.type == QUIT or (ev.type == KEYDOWN and ev.unicode == "q"):
            raise KeyboardInterrupt()
        elif ev.type == KEYDOWN:
            # Mapping for poses: 0=Rest, 1=Fist, 2=Wave In, 3=Wave Out
            if K_0 <= ev.key <= K_3:  # Only allow keys 0-3 for training
                hnd.recording = ev.key - K_0
            elif ev.unicode == "r":
                hnd.cl.read_data()
            elif ev.unicode == "e":
                print("Pressed e, erasing local data")
                myo_classifier_instance.cls.delete_data()
        elif ev.type == KEYUP:
            if K_0 <= ev.key <= K_3:  # Only allow keys 0-3 for training
                hnd.recording = -1

    # Plotting
    scr.fill((0, 0, 0), (0, 0, w, h))
    r = myo_classifier_instance.history_cnt.most_common(1)[0][0]

    # Define pose labels
    pose_labels = {0: "Rest", 1: "Fist", 2: "Wave In", 3: "Wave Out"}

    # Display training instructions
    text(scr, font, "Press 0-3 to train poses:", (20, h - 100), (255, 255, 0))
    text(
        scr,
        font,
        "0: Rest, 1: Fist, 2: Wave In, 3: Wave Out",
        (20, h - 70),
        (255, 255, 0),
    )
    text(
        scr,
        font,
        "Train each pose in multiple arm orientations for robustness!",
        (20, h - 40),
        (255, 255, 0),
    )
    text(
        scr, font, "Press 'e' to erase all training data.", (20, h - 10), (255, 255, 0)
    )

    for i in range(len(pose_labels)):  # Iterate through defined poses
        x = 0
        y = 0 + 30 * i
        # Set the barplot color
        clr = myo_classifier_instance.cls.color if i == r else (255, 255, 255)

        # Display count of training samples for each pose
        txt = font.render(
            "%5d" % (myo_classifier_instance.cls.Y == i).sum(), True, (255, 255, 255)
        )
        scr.blit(txt, (x + 20, y))

        # Display pose number and label
        txt = font.render("%d: %s" % (i, pose_labels.get(i, "Unknown")), True, clr)
        scr.blit(txt, (x + 110, y))

        # Plot the barchart plot
        scr.fill(
            (0, 0, 0),
            (
                x + 130,
                y + txt.get_height() / 2 - 10,
                len(myo_classifier_instance.history) * 20,
                20,
            ),
        )
        scr.fill(
            clr,
            (
                x + 130,
                y + txt.get_height() / 2 - 10,
                myo_classifier_instance.history_cnt[i] * 20,
                20,
            ),
        )

    pygame.display.flip()


if __name__ == "__main__":
    pygame.init()
    w, h = 800, 400  # Increased height to accommodate instructions
    scr = pygame.display.set_mode((w, h))
    font = pygame.font.Font(None, 30)

    m = MyoClassifier(Classifier())
    hnd = EMGHandler(m)
    m.add_emg_handler(hnd)
    m.connect()

    m.add_raw_pose_handler(print)

    # Set Myo LED color to model color
    m.set_leds(m.cls.color, m.cls.color)
    # Set pygame window name
    pygame.display.set_caption(m.cls.name)

    try:
        while True:
            m.run()
            run_gui(m, hnd, scr, font, w, h)

    except KeyboardInterrupt:
        pass
    finally:
        m.disconnect()
        print()
        pygame.quit()
