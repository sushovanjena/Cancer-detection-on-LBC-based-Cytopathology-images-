import logging
formatter = logging.Formatter('%(asctime)s - %(process)d - %(thread)d - %(threadName)s - %(processName)s - %(name)s '
                              '- %(filename)s - %(funcName)s - %(lineno)d - %(levelname)s - %(message)s')
logger = logging.getLogger("astra_log")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

import os
import sys
import time
import queue as queue
from datetime import datetime

import traceback
import tensorflow as tf
import multiprocessing as mp
from logging.handlers import RotatingFileHandler

import numpy as np
import cv2

from . import testing_cbam
from . import stitch as stitcher


def process_slide(slide, out_queue, _, progress, clog_path, lock, fswrite=True):
    logger.debug("Processing slide with tiles at {}".format(slide.pyramid_dir))

    # noinspection PyBroadException
    try:
        output_dir = os.path.join(slide.slide_dir, "heatmap_tiles")
        tile_folder = os.path.join(slide.slide_dir, "heatmap_pyr")
        os.makedirs(tile_folder,exist_ok=True)
        pyramid_levels_dir = os.path.join(slide.pyramid_dir, "tiles_files")
        # Checking if the pyramid generation has started
        logger.debug("Checking if {} is created".format(pyramid_levels_dir))
        while True:
            if os.path.exists(pyramid_levels_dir):
                pyramid_levels = os.listdir(pyramid_levels_dir)
                if len(pyramid_levels) > 0:
                    break
            else:
                time.sleep(1.0)
        logger.debug("{} is created".format(pyramid_levels_dir))

        pyramid_levels = map(int, pyramid_levels)
        base_pyramid_level = max(pyramid_levels)
        base_pyr_dir_path = os.path.join(slide.pyramid_dir, "tiles_files", str(base_pyramid_level))
        logger.debug("Found {} as final layer".format(base_pyramid_level))
        next_pyr_dir_path = os.path.join(slide.pyramid_dir, "tiles_files", str(base_pyramid_level-1))
        logger.debug("Found {} as pre final layer".format(next_pyr_dir_path))


        if os.path.exists(base_pyr_dir_path):
            logger.debug("{} exists, starting to process".format(base_pyramid_level))

            model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models')

            # BiT model path
            # tile_model = os.path.join(model_path, 'tfhub_modules/bit-m-50x1')

            # logger.debug("Extracting features using model at {}".format(tile_model))
            # features, fidxs = infer.extract_features(base_pyr_dir_path, next_pyr_dir_path, tile_model, logger, lock)

            wsi_model = os.path.join(model_path, 'best_model.pth')
            # logger.debug("Acquiring lock for final inference".format(tile_model))
            with lock:
                logger.debug("Starting inference using model at {}".format(wsi_model))
                classification_result = testing_cbam.heatmap(base_pyr_dir_path, output_dir, wsi_model)

                heatmap = mp.Process(target = stitcher.stitch_slide, args = (output_dir, tile_folder, clog_path, lock))
                heatmap.start()
                heatmap.join()
                print("heatmap dzi created")
            slide.diagnosis = classification_result


            
            logger.info('Successfully processed {}, result-{}'.format(slide.pyramid_dir, classification_result))

            with open(os.path.join(slide.slide_dir, "algo_output.txt"), "w") as f:
                logger.debug("Writing results to {}".format(slide.seg_annot_file))
                f.write(classification_result)
                # if classification_result[0] > classification_result[1]:
                #     f.write("Normal")
                # else:
                #     f.write("Abnormal")

                

            out_queue.put(slide)
            logger.info('slide object pushed in queue')

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Exception in processing with error {} with trace back {}".format(e, tb))
        out_queue.put(RuntimeError(slide.uid))


def create_logger(filename=None, level=logging.INFO):
    if filename:
        fh = RotatingFileHandler(filename, maxBytes=100*1024*1024, backupCount=10)
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)

    return logger




def queue_processor(queue_in, queue_out, process_lock, quit_event, model_path, log_file, progress=None):

    logger = create_logger(log_file)
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    clog_path = log_file + '_' + dt_string + '.clog'
    logger.debug("Clog path is {}".format(clog_path))
    logger.info("Staring Astra Queue API")
    logger.warning("model_path argument will be removed in next API change")
    
    if progress is None:
        logger.warning("Please use progress argument to get back progress")
        progress = mp.Value('d', lock=True)
    else:
        logger.warning("Progress variable is work in progress")

    while not quit_event.is_set():

        try:
            logger.debug("Waiting for slide")
            slide = queue_in.get(timeout=1)
            progress.value = 0.0

            # noinspection PyBroadException
            try:
                logger.info("Received for processing tgt tiles {}".format(slide.pyramid_dir))
                worker = mp.Process(target=process_slide, args=(slide, queue_out, logger, progress,clog_path, process_lock))
                worker.start()
                worker.join()

            except Exception as e:
                tb = traceback.format_exc()
                logger.exception("Exception in processing with error {} with trace back {}".format(e, tb))
                queue_out.put(RuntimeError(slide.uid))

            progress.value = 1.0

        except queue.Empty:
            continue



