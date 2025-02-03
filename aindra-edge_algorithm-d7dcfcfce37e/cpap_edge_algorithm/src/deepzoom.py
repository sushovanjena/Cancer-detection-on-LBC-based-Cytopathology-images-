import os
import time
import shutil
import multiprocessing
from PIL import Image
import cv2
import json
import pickle
import itertools
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from multiprocessing import Process, Pool, Lock, Queue

from . import blending 
from . import registration as reg


class BlendImgError(RuntimeError):
    pass


def chunks(src_imgs, n):
    for i in range(0, len(src_imgs), n):
        yield src_imgs[i:i+n]


def save_mean_corr_img(img_path, mean_img):
    for img_path in img_path:
        src_img_path, tgt_img_path = img_path
        img = cv2.imread(src_img_path)
        blur = cv2.GaussianBlur(img, (0, 0), 2.0)
        img = cv2.addWeighted(img, 2.0, blur, -1.0, 0, img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        corr_img = np.asarray(img, dtype=float)

        corr_img[:, :, 2] = np.multiply(corr_img[:, :, 2], mean_img)

        corr_img[:, :, 1] = np.multiply(corr_img[:, :, 1], 1.5)
        corr_img[:, :, 2] = np.multiply(corr_img[:, :, 2], 0.95)

        corr_img[corr_img > 254] = 254
        corr_img = np.asarray(corr_img, dtype=np.uint8)
        corr_img = cv2.cvtColor(corr_img, cv2.COLOR_HSV2BGR)

        gamma_bg = 0.9
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma_bg) * 255.0, 0, 255)

        corr_img[:, :, 0] = cv2.LUT(corr_img[:, :, 0], lookUpTable)
        corr_img[:, :, 1] = cv2.LUT(corr_img[:, :, 1], lookUpTable)

        gamma_r = 0.8
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma_r) * 255.0, 0, 255)

        corr_img[:, :, 2] = cv2.LUT(corr_img[:, :, 2], lookUpTable)

        if not os.path.exists(tgt_img_path):
            cv2.imwrite(tgt_img_path, corr_img)
            
def save_mean_corr_img_rchannel(img_path_list, mean_img):
    for img_path in img_path_list:
        src_img_path, tgt_img_path = img_path
        img = cv2.imread(src_img_path)
        blur = cv2.GaussianBlur(img, (0, 0), 2.0)
        img = cv2.addWeighted(img, 2.0, blur, -1.0, 0, img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        corr_img = np.asarray(img, dtype=float)

        corr_img[:, :, 2] = np.multiply(corr_img[:, :, 2], mean_img)

        corr_img[:, :, 1] = np.multiply(corr_img[:, :, 1], 1.5)
        corr_img[:, :, 2] = np.multiply(corr_img[:, :, 2], 0.95)

        corr_img[corr_img > 254] = 254
        corr_img = np.asarray(corr_img, dtype=np.uint8)
        corr_img = cv2.cvtColor(corr_img, cv2.COLOR_HSV2BGR)

        gamma_r = 0.6
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma_r) * 255.0, 0, 255)

        corr_img[:, :, 2] = cv2.LUT(corr_img[:, :, 2], lookUpTable)

        if not os.path.exists(tgt_img_path):
            cv2.imwrite(tgt_img_path, corr_img)


def mean_corr(a_b):
    return save_mean_corr_img_rchannel(*a_b)


def calc_mean_img(imgs):
    img_size = np.shape(cv2.imread(imgs[0]))

    imgs_sum = np.zeros((img_size[0], img_size[1]), dtype=float)

    for src_image in imgs:
        img = cv2.imread(src_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        imgs_sum = np.add(imgs_sum, img[:, :, 2])

    mean_img = imgs_sum / len(imgs)
    mean_val = cv2.mean(mean_img)
    gain_coeff = mean_val[0] / mean_img
    return gain_coeff

def correct_mean_new(n_cores, src_sub_lists, src_tgt_sub_lists):
    """
    src_path -- > folder which contains tiles
    tgt_tiles_dir -- > target folder
    """
    

    worker_pool = Pool(n_cores)

    result = worker_pool.map(calc_mean_img, src_sub_lists)
    # print(type(result), (result[0]))
    gain_coeff = np.mean(result, axis=0)
    # print(gain_coeff)
    worker_pool.map(mean_corr, zip(src_tgt_sub_lists, itertools.repeat(gain_coeff)))
    worker_pool.close()
    worker_pool.join()

def correct_mean(src_path, tgt_tiles_dir):
    #print('src_path :' ,src_path)
    src_images = os.listdir(src_path)
    src_imgs = list()
    src_tgt_pair = list()

    for src_image in src_images:
        if '.webp' in src_image:
            src_imgs.append(os.path.join(src_path, src_image))
            src_tgt_pair.append((os.path.join(src_path, src_image), os.path.join(tgt_tiles_dir, src_image)))

    # n_cores = int(multiprocessing.cpu_count()/2)
    n_cores = 20
    chunk_size = int(len(src_imgs) / 8)
    #print(chunk_size)

    src_sub_lists = list(chunks(src_imgs, chunk_size))
    src_tgt_sub_lists = list(chunks(src_tgt_pair, chunk_size))

    worker_pool = Pool(n_cores)

    result = worker_pool.map(calc_mean_img, src_sub_lists)
    # print(type(result), (result[0]))
    gain_coeff = np.mean(result, axis=0)
    # print(gain_coeff)
    worker_pool.map(mean_corr, zip(src_tgt_sub_lists, itertools.repeat(gain_coeff)))
    worker_pool.close()
    worker_pool.join()

# def create_tx_pairs(src_path):
#     r_max = 0
#     c_max = 0
#     file_dict = dict()
#     src_images = os.listdir(src_path)
    
#     # Iterate through the images and extract row, col information
#     for src_image in src_images:
#         if '.jpeg' in src_image:
#             name, ext = src_image.split(".")
#             c, r = name.split("_")  # Change to r, c instead of c, r
#             c, r = int(c), int(r)
#             r_max = max(r_max, r)
#             c_max = max(c_max, c)
#             file_dict['{}_{}'.format(c, r)] = src_image
    
#     # Read the image based on the max row, col for shape extraction
#     img = cv2.imread(os.path.join(src_path, file_dict['{}_{}'.format(r_max, c_max)]))
#     r_max = r_max + 1
#     c_max = c_max + 1
#     print("r_max, c_max in create tx pairs",c_max, r_max)
#     tx_triplets = list()
#     horizontal_tx_pairs = list()
#     vertical_tx_pairs = list()
    
#     # Process the image pairs
#     for src_image in src_images:
#         if '.jpeg' in src_image:
#             name, ext = src_image.split(".")
#             c, r = name.split("_")  # Change to r, c instead of c, r
#             c, r = int(c), int(r)

#             central_image = os.path.join(src_path, file_dict['{}_{}'.format(r, c)])
            
#             # Find the left and top neighbor images based on the new (r, c) focus
#             # For even rows, the left neighbor is at c+1, for odd rows it's at c-1
#             if c % 2 == 0 and '{}_{}'.format(r, c + 1) in file_dict:
#                 left_image = os.path.join(src_path, file_dict['{}_{}'.format(r, c + 1)])
#             elif c % 2 != 0 and '{}_{}'.format(r, c - 1) in file_dict:
#                 left_image = os.path.join(src_path, file_dict['{}_{}'.format(r, c - 1)])
#             else:
#                 left_image = ""

#             # For the top neighbor, we look at the row above (r+1)
#             if '{}_{}'.format(r + 1, c_max - 1 - c) in file_dict:
#                 top_image = os.path.join(src_path, file_dict['{}_{}'.format(r + 1, c_max - 1 - c)])
#             else:
#                 top_image = ""

#             # Add triplets or pairs depending on which neighbors exist
#             if os.path.exists(left_image) and os.path.exists(top_image):
#                 tx_triplets.append([central_image, left_image, top_image])
#             elif os.path.exists(left_image):
#                 horizontal_tx_pairs.append([central_image, left_image])
#             elif os.path.exists(top_image):
#                 vertical_tx_pairs.append([central_image, top_image])

#     # return file_dict, (r_max, c_max), img.shape, tx_triplets, horizontal_tx_pairs, vertical_tx_pairs
#     return (r_max, c_max), img.shape

def create_tx_pairs(src_path):
    c_max = 0
    r_max = 0
    file_dict = dict()
    src_images = os.listdir(src_path)
    for src_image in src_images:
        if '.webp' in src_image:
            name, ext = src_image.split(".")
            c, r = name.split("_")
            c, r = int(c), int(r)
            c_max = max(c_max, c)
            r_max = max(r_max, r)
            file_dict['{}_{}'.format(c, r)] = src_image
    img = cv2.imread(os.path.join(src_path, file_dict['{}_{}'.format(c_max, r_max)]))
    c_max = c_max + 1
    r_max = r_max + 1

    tx_triplets = list()
    horizontal_tx_pairs = list()
    vertical_tx_pairs = list()
    for src_image in src_images:
        if '.webp' in src_image:
            name, ext = src_image.split(".")
            c, r = name.split("_")
            c, r = int(c), int(r)

            central_image = os.path.join(src_path, file_dict['{}_{}'.format(c, r)])
            # Need to check for neighbour images which have overlap
            # Top and left overlapping images are paired with center image
            # Need to check for odd and even row to check the overlapping pattern, top left is origin
            if r % 2 == 0 and '{}_{}'.format(c + 1, r) in file_dict:
                left_image = os.path.join(src_path, file_dict['{}_{}'.format(c + 1, r)])
            elif r % 2 != 0 and '{}_{}'.format(c - 1, r) in file_dict:
                left_image = os.path.join(src_path, file_dict['{}_{}'.format(c - 1, r)])
            else:
                left_image = ""

            if '{}_{}'.format(c_max - 1 - c, r + 1) in file_dict:
                top_image = os.path.join(src_path, file_dict['{}_{}'.format(c_max - 1 - c, r + 1)])
            else:
                top_image = ""

            if os.path.exists(left_image) and os.path.exists(top_image):
                tx_triplets.append([central_image, left_image, top_image])
            elif os.path.exists(left_image):
                horizontal_tx_pairs.append([central_image, left_image])
            elif os.path.exists(top_image):
                vertical_tx_pairs.append([central_image, top_image])

    return (r_max, c_max), img.shape

def register(data_queue,triplet_files, horizontal_files, vertical_files,batch_size, pause_lock):
    ld_library_path = os.getenv("LD_LIBRARY_PATH")
    print("LD_LIBRARY_PATH:", ld_library_path)

    tx_graph = reg.tx_est_graph(batch_size=batch_size)

    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
    with tf.Session(graph=tx_graph, config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
        ld_library_path = os.getenv("LD_LIBRARY_PATH")
        print("LD_LIBRARY_PATH:", ld_library_path)
        init = tx_graph.get_collection("init")
        sess.run(init)
        
        triplet_txs = reg.compute_triplet_txs(triplet_files, tx_graph, sess, pause_lock)
        horizontal_txs = reg.compute_horizontal_txs(horizontal_files, tx_graph, sess, pause_lock)
        vertical_txs = reg.compute_vertical_txs(vertical_files, tx_graph, sess, pause_lock)

        pair_translations = horizontal_txs + vertical_txs + triplet_txs
        with open('/tmp/pair_tx.pkl', 'wb') as f:
            pickle.dump(pair_translations, f)
        data_queue.put('/tmp/pair_tx.pkl')

    sess.close()


def register_wrapper(triplet_files, horizontal_files, vertical_files, pause_lock=Lock(), num_threads=4,batch_size =4,
                     gpu_mem_fraction=0.3):
    """
    Computes pair wise translation between images in a folder
    The images are assumed to be named as {column}_{row}.webp
    The translations are returned in the argument pair_translations.
    pair_translations : [[current_image, reference image, horizontal tx, vertical tx, variance]...]
    """
    
    data_queue = Queue()
    worker = Process(target=register,
                     args=(data_queue,
                           triplet_files, horizontal_files, vertical_files,
                           batch_size, pause_lock))

    worker.start()
    worker.join()
    pair_tx_path = data_queue.get()
    with open(pair_tx_path, 'rb') as f:
        pair_translations = pickle.load(f)

    return pair_translations

def save_image(src_path, dest_path):
    try:
        # Open the source image file
        with Image.open(src_path) as img:
            # Save the image to the destination path
            img.save(dest_path)
        print(f"Image saved to {dest_path}")
    except Exception as e:
        print(f"Error saving image {src_path} to {dest_path}: {e}")

def blend_q_processor(logger, pause_lock, blend_list, dir_path, is_alpha_blending, scale, nworkers,overlap,global_positions):
    print("inside bqp")
    aug_blend_queue = Queue()
    exception_queue = Queue()

    processed_cntr = 0
    if overlap == 0:
        
        for img_paths in global_positions:
            src_path = img_paths[0]
            
            # Extract the filename from the src_path
            filename = os.path.basename(src_path)  # Extracts '31_51.jpeg' from the src_path
            
            # Remove the '.jpeg' extension and split by the underscore to get the row and col
            tile_id = filename.replace('.webp', '').split('_')  # Results in ['31', '51']
            
            # Convert the row and col to integers (optional but ensures they are correct types)
            tile_col = int(tile_id[0])
            tile_row = int(tile_id[1])

            # Print for debugging
            print(f"src_path: {src_path}, tile_col: {tile_col}, tile_row: {tile_row}")
            
            # Save the tile directly to the destination folder using the same tile_id naming convention
            tile_path = os.path.join(dir_path, '{}_{}.webp'.format(tile_col, tile_row))

            # Call the save_image function to save the image
            save_image(src_path, tile_path)

        return  # Exit the function since blending is not needed

    while processed_cntr < len(blend_list):

        workers = list()
        
        # Create worker processes
        for i in range(nworkers):
            # print(f"Starting worker {i}")
            worker = Process(target=blending.blend_worker, args=(aug_blend_queue, exception_queue))
            worker.start()
            workers.append(worker)

        # Queue tasks for workers
        for i in range(processed_cntr, len(blend_list)):
            blend_item = blend_list[i]

            # Ensure the queue doesn't overload the workers
            while aug_blend_queue.qsize() >= nworkers:
                print("Waiting for workers to process...")
                time.sleep(0.5)

            if pause_lock.acquire():  # Ensure we acquire the lock to add task
                aug_blend_queue.put((blend_item, dir_path, is_alpha_blending, scale))
                processed_cntr += 1
                print("processed_cntr: ",processed_cntr)
                pause_lock.release()
            else:
                break

        # Send termination signals to workers (one for each worker)
        for _ in range(nworkers):
            aug_blend_queue.put(None)

        # Wait for workers to finish
        for worker in workers:
            worker.join()

        # Check if any exceptions occurred
        if not exception_queue.empty():
            failed_task = exception_queue.get()
            logger.error("Blend and save failed for {}".format(failed_task[1]))
            raise BlendImgError("Blend and save failed")

        print("Batch processing complete.")
    
    print("All blending tasks complete.")

# def blend_q_processor(logger, pause_lock, blend_list, dir_path, is_alpha_blending, scale, nworkers):
#     print("inside bqp")
#     aug_blend_queue = Queue()
#     exception_queue = Queue()

#     processed_cntr = 0
#     while processed_cntr < len(blend_list):

#         workers = list()
#         for i in range(nworkers):
#             print("in i")
#             worker = Process(target=blending.blend_worker, args=(aug_blend_queue, exception_queue))
#             worker.start()
#             workers.append(worker)

#         for i in range(processed_cntr, len(blend_list)):
#             blend_item = blend_list[i]

#             while aug_blend_queue.qsize() > nworkers:
#                 print("in aug_blend")    
#                 time.sleep(0.5)

#             if pause_lock.acquire():
#                 aug_blend_queue.put((blend_item, dir_path, is_alpha_blending, scale))
#                 processed_cntr += 1
#                 pause_lock.release()
#             else:
#                 break

#         aug_blend_queue.put(None)
#         for worker in workers:
#             worker.join()
#         aug_blend_queue.get()
#         if not exception_queue.empty():
#             failed_task = exception_queue.get()
#             logger.error("Blend and save failed for {}".format(failed_task[1]))
#             exception_queue = Queue()
#             raise BlendImgError("Blend and save failed")


def generate_deepzoom_pyramid(global_positions, wsi_shape, tgt_path, logger, clog_file, tgt_tile_shape, overlap,
                              image_shape, pause_lock, nworkers, image_format='webp',
                              tmp_folder_prefix="__pyr_", pyramid_name="tiles"):
    next_overlap = 1
    logger.info("Creating c logger")
    print("c log file  is =========" , clog_file )
    blending.create_clogger(clog_file)
    print("after createing c log")
    logger.info("Created c logger")
    print("Created c logger")

    logger.info("Creating blend mask")
    blending.create_mask(image_shape[0], image_shape[1])
    logger.info("Created blend mask")
    print("Created blend mask")
    try:
        os.makedirs(tgt_path)
    except os.error:
        pass
    logger.info("Pyramid directory created")
    print("Pyramid directory created")

    with pause_lock:
        blend_list, wsi_pix_size = blending.compute_blending_grid(src_list=global_positions,
                                                                tgt_tile_size=tgt_tile_shape,
                                                                tgt_overlap=overlap)
    print(wsi_pix_size)
    logger.info("Creating dzi file")
    print("Creating dzi file")
    # Create the dzi file
    dzi_info = dict()
    dzi_info["Image"] = dict()
    dzi_info["Image"]["xmlns"] = "http://schemas.microsoft.com/deepzoom/2008"
    dzi_info["Image"]["Format"] = "webp"
    dzi_info["Image"]["Overlap"] = str(int(overlap))
    dzi_info["Image"]["TileSize"] = str(int(tgt_tile_shape[0]))
    dzi_info["Image"]["Size"] = dict()
    dzi_info["Image"]["Size"]["Height"] = str(int(wsi_pix_size[0]))
    dzi_info["Image"]["Size"]["Width"] = str(int(wsi_pix_size[1]))

    dzi_path = os.path.join(tgt_path, pyramid_name + '.dzi')
    with open(dzi_path, 'w') as fp:
        json.dump(dzi_info, fp, indent=4)

    base_level = int(np.ceil(np.log2(max(int(wsi_pix_size[0]), int(wsi_pix_size[1])))))
    next_level = base_level - 1
    print("dzi created")
    # --------------------------------------------------------------------------------------------
    # Create the image pyramid in a folder called pyramid_name_files
    # ---------------------------------------------------------------------------------------------
    pyr_files_tgt_dir = os.path.join(tgt_path, pyramid_name + "_files")
    if os.path.exists(pyr_files_tgt_dir):
        shutil.rmtree(pyr_files_tgt_dir)
    os.makedirs(pyr_files_tgt_dir)

    pyramid_base_dir = os.path.join(pyr_files_tgt_dir, str(base_level))

    try:
        os.makedirs(pyramid_base_dir)
    except os.error:
        pass
    print("going inside blend_q_processor")
    is_alpha_blending = True
    blend_q_processor(logger, pause_lock, blend_list, pyramid_base_dir, is_alpha_blending, 1.0, nworkers, overlap,global_positions)
    print("going out blend_q_processor")
    next_layer_src = list()
    for blend_item in blend_list:
        # print("inside blenditem loop")
        tile_id, tile_pos, tile_size = blend_item[0:3]
        tile_path = os.path.join(pyramid_base_dir, '{}_{}.webp'.format(tile_id[1], tile_id[0]))
        next_layer_src.append([tile_path, tile_pos, tile_size])
    logger.info("Base pyramid directory created")
    print("Base pyramid directory created")
    # level_list = [pyramid_base_dir]

    while True:

        level_path = os.path.join(pyr_files_tgt_dir, str(next_level))
        next_level = next_level - 1
        logger.debug("Creating pyramid directory " + level_path)
        print("Creating pyramid directory ",level_path)
        try:
            os.makedirs(level_path)
        except os.error:
            pass

        blend_list, wsi_size = blending.compute_blending_grid(next_layer_src,
                                                              tgt_tile_size=(tgt_tile_shape[0]*2, tgt_tile_shape[1]*2),
                                                              tgt_overlap=next_overlap*2)
        is_alpha_blending = False
        blend_q_processor(logger, pause_lock, blend_list, level_path, is_alpha_blending, 0.5, nworkers,next_overlap,global_positions)

        if len(blend_list) == 1 and np.all(np.array(blend_list[0][2], dtype=int) < 4):
            break

        next_layer_src = list()
        for blend_item in blend_list:

            tile_id, tile_pos, tile_size = blend_item[0:3]
            tile_path = os.path.join(level_path, '{}_{}.webp'.format(tile_id[1], tile_id[0]))

            tile_size_dwn = np.asarray(tile_size, dtype=int)/2
            tile_pos_dwn = np.asarray(tile_pos, dtype=int)/2
            next_layer_src.append([tile_path, tile_pos_dwn, tile_size_dwn])

    blending.remove_clogger()
    logger.info("C logger removed")
