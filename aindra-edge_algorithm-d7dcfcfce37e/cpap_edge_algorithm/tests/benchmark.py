import os
import sys
import json
import numpy as np
import sklearn.metrics as skm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# max_gpu_mem = 2048*5
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [
#     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=max_gpu_mem)])
# logical_gpus = tf.config.experimental.list_logical_devices('GPU')

sys.path.insert(0, '..')
from src import infer as inf

def parse_ds(src_dir, ypath=None, npath=None):

    files, labels =list(), list()
    for root_iter, _, files_iter in os.walk(src_dir):
        for item in files_iter:
            if item == 'meta.json':

                if ypath and ypath not in root_iter:
                    continue
                if npath and npath in root_iter:
                    continue

                with open(os.path.join(root_iter, item), 'r') as f:
                    try:
                        meta = json.load(f)
                    except Exception as e:
                        continue

                    if meta['label'].lower() == 'normal':
                        label = [1.0, 0.0]
                    else:
                        label = [0.0, 1.0]

                    levels = list()
                    for slide_file in os.listdir(root_iter):
                        if slide_file.endswith('.16f'):
                            slide_file = slide_file.split('_')[0]
                            slide_file = slide_file.split('-')[1]
                            levels.append(int(slide_file))
                    levels.sort()

                    files.append(os.path.join(root_iter, 'level-{}_feature.16f'.format(levels[-1])))
                    labels.append(label)
    return files, labels


files, labels = parse_ds('/media/ssd/oracle/features/media/dataset/images/aindra')
logits = inf._predict(files, None, '/home/hari/Documents/cacx/tmp/logs/deploy_0.0.0b2/time-2020-08-07_11:41:56.477/ckpts/maxpool_saved_model.h5')

labels = np.asarray(labels)
logits = logits > 0.5
accuracy = skm.accuracy_score(labels, logits)
print(accuracy)


