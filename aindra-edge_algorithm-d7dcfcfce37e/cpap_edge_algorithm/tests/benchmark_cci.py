import os
import sys
import time
import numpy as np
import multiprocessing as mp
import sklearn.metrics as skm
from argparse import ArgumentParser
from aindra_ds import cacx
from edge_algorithm.src.wsi_analyzer import process_slide, create_logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# max_gpu_mem = 2048*5
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [
#     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=max_gpu_mem)])
# logical_gpus = tf.config.experimental.list_logical_devices('GPU')

from edge_algorithm.src.wsi_analyzer import process_slide, create_logger

def parse_ds(src_dir):

    files, labels =list(), list()
    for dir in os.listdir(src_dir):
        date_dir = os.path.join(src_dir, dir)
        for dir2 in os.listdir(date_dir):
            slide_dir = os.path.join(date_dir, dir2)
            if os.path.exists(os.path.join(slide_dir, 'pyramid', 'tiles.dzi')):
                files.append(slide_dir)
                labels.append(['1.0, 0.0'])
    return files, labels


wsis, labels = parse_ds('/media/Data/aindra_iad')
for wsi in wsis:
    slide = cacx.Slide(slide_dir=wsi, slide_type=cacx.SLIDE_TYPE_PAP_LBC, tile_dir='/tmp')

    tic = time.time()
    out_q = mp.Queue()
    process_slide(slide, None, create_logger('/tmp/log.txt'), out_q, fswrite=False)
    result_slide = out_q.get(timeout=1)
    toc = time.time()

    #print("Time taken : {}s".format(toc - tic))
    print(wsi, result_slide.diagnosis)
    #assert (result_slide.diagnosis is not None)

