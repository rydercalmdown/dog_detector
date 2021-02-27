import numpy as np
import imutils
import time
import cv2
import os
from progress.bar import Bar
from imutils.video import VideoStream, FPS


class Classifier():
    """Class for running the classification app"""

    def __init__(self):
        self.set_up()

    def __del__(self):
        self.clean_up()

    def set_up(self):
        """Sets up resources"""
        self.set_defaults()
        self.load_labels()
        self.set_label_colours()
        self.load_model_to_memory()

    def set_defaults(self):
        self.progress_bar = None
        self.video_writer = None
        self.video_stream = None
        self.frame_width = None
        self.frame_height = None
        self.show_preview = False
        self.write_livestream_to_file = False
        self.allowed_labels = []
        self.num_frames = 0
        self.network = None
        self.frame_processing_time = 0
        self.confidence_threshold = 0.5
        self.non_maxima_supression_threshold = 0.3
        self.model_weights_path = './models/yolov3.weights'
        self.model_config_path = './models/yolov3.cfg'
        self.output_video_path = './data/output.avi'
    
    def set_callback_item_found(self, callback):
        """Sets the callback function for when an item is found"""
        self.callback_item_found = callback

    def initialize_progress_bar(self):
        """Initialize the progress bar"""
        if self.num_frames:
            self.progress_bar = Bar('Processing video', max=self.num_frames)
    
    def set_allowed_labels(self, labels):
        """Filter results to only these labels"""
        self.allowed_labels = labels
    
    def write_frame_to_video(self, frame):
        if self.write_livestream_to_file:
            if self.video_writer is None:
                self.initialize_video_writer(frame, 4)
                self.video_writer.write(frame)
    
    def classify_from_live_stream(self):
        """Runs classification on a live streaming input"""
        if os.path.exists('/.dockerenv'):
            print('Unable to run live stream in docker')
            exit(1)
        self.initialize_live_video_stream()
        while True:
            print("Running frame")
            self.box_list = []
            self.confidence_list = []
            self.class_id_list = []
            frame = self.video_stream.read()
            # frame = imutils.resize(frame, width=400)
            self.process_frame(frame)
            results = self.apply_non_maxima_supression()
            if len(results) > 0:
                for i in results.flatten():
                    label = self.labels[self.class_id_list[i]]
                    confidence = self.confidence_list[i]
                    if self.allowed_labels:
                        if str(label).lower().strip() not in self.allowed_labels:
                            self.write_frame_to_video(frame)
                            continue
                    (x, y) = (self.box_list[i][0], self.box_list[i][1])
                    (w, h) = (self.box_list[i][2], self.box_list[i][3])
                    self.draw_bounding_box(frame, i, x, y, w, h)
                    if self.callback_item_found:
                        self.callback_item_found(label, confidence, [x, y, w, h])
                    self.write_frame_to_video(frame)
            if self.show_preview:
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            self.fps.update()

    def classify_video_from_file(self, input_path, output_path):
        """Process the input video"""
        self.input_video_path = input_path
        self.output_video_path = output_path
        self.initialize_video_stream_from_file()
        print('Processing {}'.format(self.input_video_path))
        while True:
            self.box_list = []
            self.confidence_list = []
            self.class_id_list = []
            (grabbed, frame) = self.video_stream.read()
            if not grabbed:
                # Break the loop when no more frames
                if self.progress_bar:
                    self.progress_bar.finish()
                break
            if self.video_writer is None:
                self.initialize_video_writer(frame)
            self.process_frame(frame)
            results = self.apply_non_maxima_supression()
            if len(results) > 0:
                for i in results.flatten():
                    # Get bounding box coordinates
                    (x, y) = (self.box_list[i][0], self.box_list[i][1])
                    (w, h) = (self.box_list[i][2], self.box_list[i][3])
                    self.draw_bounding_box(frame, i, x, y, w, h)
            self.video_writer.write(frame)
            if self.progress_bar:
                self.progress_bar.next()

    def clean_up(self):
        """Cleans up resources"""
        print('Shutting down')
        try:
            if self.video_writer:
                self.video_writer.release()
            if self.video_stream:
                self.video_stream.release()
        except Exception:
            print('Error when shutting down')

    def load_labels(self):
        """Loads labels for model"""
        labels_path = './models/coco.names'
        self.labels = open(labels_path).read().strip().split('\n')

    def set_label_colours(self):
        """Sets a list of colours for each label"""
        self.label_colours = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")

    def load_model_to_memory(self):
        """Load the ML model into memory"""
        print('Loading model into memory')
        self.network = cv2.dnn.readNetFromDarknet(self.model_config_path, self.model_weights_path)
        self.layer_names = [self.network.getLayerNames()[i[0] - 1] for i in self.network.getUnconnectedOutLayers()]
        print('Model loaded')

    def initialize_video_writer(self, sample_frame, fps=30):
        """Initializes video writer, if None"""
        height_width = (sample_frame.shape[1], sample_frame.shape[0])
        writer_class = cv2.VideoWriter_fourcc(*"MJPG")
        self.video_writer = cv2.VideoWriter(self.output_video_path, writer_class, fps, height_width, True)
    
    def initialize_live_video_stream(self):
        self.video_stream = VideoStream(src=0).start()
        time.sleep(2.0)
        self.fps = FPS().start()

    def initialize_video_stream_from_file(self):
        self.video_stream = cv2.VideoCapture(self.input_video_path)
        try:
            if imutils.is_cv2():
                prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT
            else:
                prop = cv2.CAP_PROP_FRAME_COUNT
            self.num_frames = int(self.video_stream.get(prop))
            print('{} frames in video'.format(self.num_frames))
            self.initialize_progress_bar()
        except Exception:
            self.num_frames = -1
            print("Unknown number of frames in video")

    def is_within_acceptable_confidence(self, confidence):
        """Determines if prediction is within acceptable confidence"""
        return confidence >= self.confidence_threshold

    def apply_non_maxima_supression(self):
        """Applies non_maxima supression to results"""
        return cv2.dnn.NMSBoxes(
            self.box_list,
            self.confidence_list,
            self.confidence_threshold,
            self.non_maxima_supression_threshold
        )

    def build_bounding_box(self, detection):
        """Builds a bounding box on the current prediction"""
        box = detection[0:4] * np.array([self.frame_width, self.frame_height, self.frame_width, self.frame_height])
        (center_x, center_y, prediction_width, prediction_height) = box.astype("int")
        # get top left corner
        x = int(center_x - (prediction_width / 2))
        y = int(center_y - (prediction_height / 2))
        return [x, y, int(prediction_width), int(prediction_height)]

    def draw_bounding_box(self, frame, i, x, y, w, h):
        """Draws a bounding box on the frame"""
        colour = [int(c) for c in self.label_colours[self.class_id_list[i]]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        text = "{}: {:.4f}".format(self.labels[self.class_id_list[i]], self.confidence_list[i])
        font_scale = 1.5
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour, 3)

    def process_frame(self, frame):
        """Processes each frame of the video"""
        if self.frame_width is None or self.frame_height is None:
            (self.frame_height, self.frame_width) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.network.setInput(blob)
        start_time = time.time()
        outputs = self.network.forward(self.layer_names)
        end_time = time.time()
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if self.is_within_acceptable_confidence(confidence):
                    self.box_list.append(self.build_bounding_box(detection))
                    self.confidence_list.append(float(confidence))
                    self.class_id_list.append(class_id)
        self.frame_processing_time = end_time - start_time
