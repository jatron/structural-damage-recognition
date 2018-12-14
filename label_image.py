# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#  Modifications: to run over all files in a particular directory
#  sorts the files in alphabetical order, and prints out the image index and the predicted value
#  Example:  python -m scripts.label_image_mh \
#  --graph=tf_files/retrained_graph.pb \
#  --labels=tf_files/struct_labels.txt \
#  --path=tf_files/struct_photos/damaged 
#  Best to cat or append the output to a results file  >> results
#
#  Minnie Ho, 09/11/18
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import datasets, neighbors, linear_model, metrics

import argparse
import sys
import time

import numpy as np
import tensorflow as tf

# Used to read in files as tensors
import os
import re

def sorted_nicely( l ):
    """ Sorts the given iterable in the way that is expected.
 
    Required arguments:
    l -- The iterable to be sorted.
 
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader', dct_method="INTEGER_ACCURATE")   # added to be bit accurate to im.write
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)
#  result = sess.run(image_reader)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__ == "__main__":
  file_name = "tf_files/flower_photos/daisy/3475870145_685a19116d.jpg"
#  model_file = "./mobilenet_1.0_224_graph.pb"
  model_file = "../tf_files/inceptionv3_4000_graph.pb"
  label_file = "../tf_files/retrained_labels.txt"
  results_file = "../data/results.txt"
  
  path = "../data/task2/test500" 
#  input_height = 224
#  input_width = 224
#  input_mean = 128
#  input_std = 128
#  input_layer = "input"
#  output_layer = "final_result"
  input_height = 299
  input_width = 299
  input_mean = 128
  input_std = 128
  input_layer = 'Mul'
  output_layer = "final_result"

  parser = argparse.ArgumentParser()
  parser.add_argument("--path", help="image to be processed")
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  parser.add_argument("--results_file", help="name of results file") 
  args = parser.parse_args()

  num_files = 1

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.path:
    path = args.path
    folder = os.fsencode(path)
    filenames = []
    num_files = 0
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith(('.jpg')):
            filenames.append(filename)
            num_files = num_files + 1
    if not filenames:
        print("List is empty")
        filenames = file_name
    sfilenames = sorted_nicely(filenames)
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

# Read and sort all filenames in a folder
  folder = os.fsencode(path)        
  filenames = []
  num_files = 0
  for file in os.listdir(folder):
      filename = os.fsdecode(file)
      if filename.endswith(('.jpg')):
         filenames.append(filename)
         num_files = num_files + 1
  if not filenames:
      print("List is empty")
      filenames = file_name
  sfilenames = sorted_nicely(filenames)

  graph = load_graph(model_file)
  
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);
  
  
#  num_k = 1
  labels = load_labels(label_file)
  
#  num_files=10
#  all_results = np.zeros((num_files*num_k, 2))
  all_results = np.zeros((num_files, 2))   # Store the top result and the probability
#  mark_time=0
#  epoch = 0
  print("Num files is "+str(num_files))
  start = time.time()  
  
  train_images = np.load('../data/X_train.npy')
  train_label = np.load('../data/Y_train.npy')
  y_test = train_label[:num_files]

 # for i in range(num_files):
  for file_ind in range(num_files):     
      file_name=os.path.join(path,sfilenames[file_ind])
      file_num = re.findall(r'\d+', file_name)
 #     print(file_name)
 #     file_name='../tf_files/struct_photos/damaged_image1.jpg'
      t = read_tensor_from_image_file(file_name,
                                      input_height=input_height,
                                      input_width=input_width,
                                      input_mean=input_mean,
                                      input_std=input_std)
    
      if ((file_ind%50)==0):
          print("File index is "+str(file_ind)+" File number is "+str(file_num))
          print('\nEvaluation time: {:.3f}s\n'.format(time.time()-start))
#         np.savetxt(results_file, all_results, fmt='%d, %.3f')
          
      with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0],
                          {input_operation.outputs[0]: t})
        
      results = np.squeeze(results)
      top_k = results.argsort()[-1:][::-1]
      
#      for j in range(len(top_k)):
#          all_results[int(file_num[0])+j,:] = np.array([top_k[j], results[top_k[j]]])
      for res_ind in top_k:   # need to flip the results, since I stupidly put the labels in the wrong order
          if (res_ind==0):
              all_results[file_ind,:] = np.array([1, results[res_ind]])  
          else:
              all_results[file_ind,:] = np.array([0, results[res_ind]])  
      print("Number is "+str(file_ind)+" Image is classified as "+labels[res_ind]+" Probability is "+str(results[res_ind]))
  
  y_pred = all_results[:,0]    
  cm = metrics.confusion_matrix(y_test,y_pred)
  accuracy = sum(np.diag(cm))/np.sum(cm)
          
  np.savetxt(results_file, all_results, fmt='%d, %.3f')


