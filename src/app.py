from Classifier import Classifier
import time
import pyttsx3
import os


tts_engine = pyttsx3.init()
time_since_last_detection = [0]
announcement_timeout_seconds = 10


def found_item_callback(label, confidence, coordinates):
    """Callback function for when an item is found"""
    print('{} found, {} confident'.format(label, confidence))
    current_time = int(time.time())
    if (current_time - time_since_last_detection[0]) > announcement_timeout_seconds:
        announce_item_found(label)
        time_since_last_detection[0] = int(time.time())


def announce_item_found(label):
    """Announce out loud that an item was found"""
    speech = "Attention. There is a {} outside.".format(label)
    cmd = 'say "{}"'.format(speech)
    os.system(cmd)


def main():
    """Run the application"""
    c = Classifier()
    try:
        allowed_labels = [
            'dog',
            'person',
            'cat',
        ]
        c.set_allowed_labels(allowed_labels)
        c.set_callback_item_found(found_item_callback)
        c.classify_from_live_stream()
        # c.classify_video_from_file('./data/test.m4v', './data/test.avi')
    except KeyboardInterrupt:
        print('Exiting...')


if __name__ == '__main__':
    main()
