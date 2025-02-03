import os
# Uncomment to enable TensorFlow logging
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import time
import psutil
import random
import subprocess
import multiprocessing
from multiprocessing import Queue, Lock, Process

import tensorflow as tf
from aindra_ds import cacx
from edge_algorithm.src.wsi_analyzer import queue_processor

case_path = sys.argv[1]

in_queue, out_queue = Queue(), Queue()
manager = multiprocessing.Manager()
quit_event = manager.Event()


log_file = "/tmp/algo_log.log"
lock = Lock()
p = Process(target=queue_processor, args=(in_queue, out_queue, lock, quit_event, None, log_file))
p.start()

ex_msg = cacx.Slide(slide_dir=case_path, tile_dir=case_path, slide_type=cacx.SLIDE_TYPE_PAP_HISTO)

tic = time.time()
in_queue.put(ex_msg)

# Simulate locking mechanisms
net_usage = [0.0, 0.0]
while out_queue.empty():
    sleep_time = random.randint(10, 20)

    if random.random() > 0.5:
        with lock:
            print("Lock is live for {}".format(sleep_time))
            if True:
                if tf.test.gpu_device_name():
                    for i in range(sleep_time):
                        time.sleep(1)
                        try:
                            gpu_util = subprocess.check_output('nvidia-smi')
                            gpu_util = gpu_util.split('%')[1]
                            gpu_util = gpu_util.split('|')[-1]
                            gpu_util = int(gpu_util)
                            net_usage[1] += gpu_util
                        except TypeError:
                            pass
                else:
                    net_usage[1] += psutil.cpu_percent(interval=sleep_time)
    else:
        print("Lock is dead for {}".format(sleep_time))
        if tf.test.gpu_device_name():
            for i in range(sleep_time):
                    time.sleep(1)
                    try:
                        gpu_util = subprocess.check_output('nvidia-smi')
                        gpu_util = gpu_util.split('%')[1]
                        gpu_util = gpu_util.split('|')[-1]
                        gpu_util = int(gpu_util)
                        net_usage[1] += gpu_util
                    except TypeError:
                        pass
        else:
            net_usage[0] += psutil.cpu_percent(interval=sleep_time)

out_msg = out_queue.get()
quit_event.set()
p.join()

print("Process usage was {} w/o lock and {} with lock".format(round(net_usage[0]), round(net_usage[1])))