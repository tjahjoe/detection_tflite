import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import cv2
import sys
np.set_printoptions(threshold=sys.maxsize)

def categoriez(frame, output_data, shape):
  confidence = output_data[0][:, 4]
  class_probs = output_data[0][:, 5:]

  class_max_probs = np.max(class_probs, axis=1)
  high_conf_boxes = output_data[0][(confidence > 0.3) & (class_max_probs > 0.3)]

  for detec in high_conf_boxes:
    x_center, y_center, width, height, conf = detec[:5]

    x_min = int((x_center - width / 2) * shape[0])
    y_min = int((y_center - height / 2) * shape[1])
    x_max = int((x_center + width / 2) * shape[0])
    y_max = int((y_center + height / 2) * shape[1])

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


def detection(frame, frame_process, input_details, interpreter, output_details):
  interpreter.set_tensor(input_details[0]['index'], frame_process)
  interpreter.invoke()
  output_data = interpreter.get_tensor(output_details[0]['index'])
  shape = frame_process.shape[1:3]

  categoriez(frame, output_data, shape)


def preparation(frame, input_details, interpreter, output_details):
  frame_process = frame.astype(np.float32) / 255.0
  frame_process = np.expand_dims(frame_process, axis=0)

  detection(frame, frame_process, input_details, interpreter, output_details)


def open_cam( input_details, interpreter, output_details):
  cap = cv2.VideoCapture(0)

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    frame_size = input_details[0]['shape'][1:3]
    frame = cv2.resize(frame, frame_size)
    frame = cv2.flip(frame, 1)

    preparation(frame, input_details, interpreter, output_details)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
      break
  
  cap.release()
  cv2.destroyAllWindows()

def load_model():
  interpreter = tf.lite.Interpreter(model_path="yolov5n-fp16.tflite")
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  open_cam(input_details, interpreter, output_details) 

if __name__ == '__main__':
  load_model()

