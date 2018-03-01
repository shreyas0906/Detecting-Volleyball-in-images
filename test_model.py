import numpy as np
import os, os.path
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import csv
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util
from argparse import ArgumentParser

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

parser = ArgumentParser()
parser.add_argument('--im_dir', required=True, help='''Path to the directory of test images to run the model on''')
parser.add_argument('--outname', required=True, help='''Filename to store the predictions''')
args = parser.parse_args()


# trained model folder
MODEL_NAME = 'model_8023_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'ball-detect.pbtxt')

NUM_CLASSES = 1

# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

#Saving the output to csv files
#@parameters of the function:
#boxes: The normalized image coordinates of the detection
#scores: The scores of the object detections
#width: Image width
#height: Image height

line_0 = [['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'number_of_detection', 'score']]
with open(args.outname, 'a') as result_file:
    writer = csv.writer(result_file, dialect='excel', lineterminator='\n')
    writer.writerows(line_0)

def save_output(boxes, scores, width, height, image_path):

    image_file = image_path.split('/') # Split on '/' to find the image file name
    image_file = image_file[-1] # Since the last item in the list will be the name of the image
    scr = []
    num_of_detect = 0
    xmin = 0; xmax = 0; ymin = 0; ymax = 0
    threshold = 0.9 # Setting the score threshold to be 0.9
    for i in range(0,4):
        if scores[0][i] > threshold: # Finding the number of detections which are more than the threshold
          num_of_detect += 1
          scr.append(scores[0][i] * 100)

    for j in range(0, num_of_detect):
      ymin = boxes[0][j][0] * height # Since we have the normalized image coordinates
      xmin = boxes[0][j][1] * width  # We are using the image dimensions
      ymax = boxes[0][j][2] * height # To scale the object detections
      xmax = boxes[0][j][3] * width


    # Saving only the image coordinates of the highest scores
    line = [[image_file, int(xmin), int(ymin), int(xmax), int(ymax), num_of_detect, scr]]
    # Writing to the csv file.
    with open(args.outname,'a') as result_file:
        writer = csv.writer(result_file, dialect='excel', lineterminator = '\n')
        writer.writerows(line)


# PATH_TO_TEST_IMAGES_DIR = 'test_images'
images_names = 'volleyball_frame_{}.png' # Please specify the image name. Example: 'volleyball_frame_0.png' without the number
# TEST_IMAGE_PATHS = [ os.path.join(args.im_dir, images_names.format(i)) for i in range(0, 3) ] #Please specify the number of images
# num_of_images = len([name for name in os.listdir(args.im_dir) if (os.path.isfile(name) and name.endswith('.png'))])
TEST_IMAGE_PATHS = [ os.path.join(args.im_dir, images_names.format(i)) for i in range(0, len([name for name in os.listdir(args.im_dir) if name.endswith('.png')]))]
# len([name for name in os.listdir('.') if (os.path.isfile(name) and name.endswith('.png'))]) --> finds the total numper of images with a .png extension

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
# print(num_of_images)
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    i = 0
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      width , height = image.size
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      save_output(boxes, scores, width, height,image_path)
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=4)
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np); plt.savefig('result' + str(i) +'.png'); i+=1;
