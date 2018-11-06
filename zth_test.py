# 使用自动抽核的视频进行处理


import cv2
import numpy as np
from scipy.misc import imread
import tensorflow as tf

from FlowNet2_src import FlowNet2, LONG_SCHEDULE
from FlowNet2_src import flow_to_image


if __name__ == '__main__':
  # Graph construction
  im1_pl = tf.placeholder(tf.float32, [1, 256, 256, 3])                # 图像的维度
  im2_pl = tf.placeholder(tf.float32, [1, 256, 256, 3])

  flownet2 = FlowNet2()
  inputs = {'input_a': im1_pl, 'input_b': im2_pl}
  flow_dict = flownet2.model(inputs, LONG_SCHEDULE, trainable=False)
  pred_flow = flow_dict['flow']

  # Feed forward
  #im1 = imread('FlowNet2_src/example/0img0.ppm')/255.
  #im2 = imread('FlowNet2_src/example/0img1.ppm')/255.
  im1 = imread('FlowNet2_src/2_crop/3970.jpg')/255.           #(256,256)
  im2 = imread('FlowNet2_src/2_crop/3971.jpg')/255.

  im1 = np.array([im1]).astype(np.float32)                   # （1,256,256）
  print(im1.shape)
  im1 = np.stack([im1,im1,im1],axis=-1)                                 # 增加一个维度 （1,256,256,3）
  print(im2.shape)
  im2 = np.array([im2]).astype(np.float32)
  im2 = np.stack([im2,im2,im2],axis=-1)
  print(im1.shape)
  print(im2.shape)


  ckpt_file = 'FlowNet2_src/checkpoints/FlowNet2/flownet-2.ckpt-0'
  saver = tf.train.Saver()

  with tf.Session() as sess:
    saver.restore(sess, ckpt_file)
    # Double check loading is correct
    #for var in tf.all_variables():
    #  print(var.name, var.eval(session=sess).mean())
    feed_dict = {im1_pl: im1, im2_pl: im2}
    pred_flow_val = sess.run(pred_flow, feed_dict=feed_dict)
    print('preflow[0]',pred_flow_val[0].shape)
  # Visualization
  import matplotlib.pyplot as plt
  flow_im = flow_to_image(pred_flow_val[0]) # 三通道的光流图
  print('flow_to_image', flow_im.shape)
  print(flow_im)
  plt.imshow(flow_im)
  plt.show()

