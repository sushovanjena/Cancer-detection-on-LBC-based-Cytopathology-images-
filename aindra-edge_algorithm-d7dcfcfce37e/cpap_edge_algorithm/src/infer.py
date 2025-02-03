import logging
logger = logging.getLogger("astra_log")

import os
import time
import numpy as np

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import tensorflow_io as tfio
import tensorflow_hub as hub
import tensorflow_addons as tfa

logger.debug("Enabling XLA JIT optimization")
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True)

from . import tf_mem_manager as tf_mem_mngr
mem_manager = tf_mem_mngr.MemoryManager(500)


def read_images(paths, idxs, batch_size=1):

    def parse_webp_imgs(file_name, image_cords):
        file_name = tf.io.read_file(file_name)
        image = tfio.image.decode_webp(file_name)
        channels = tf.unstack(image, axis=-1)
        image_decoded = tf.stack([channels[0], channels[1], channels[2]], axis=-1)

        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)

        image = tf.expand_dims(image, 0)
        image = tf.image.resize_with_pad(image, 482, 482, antialias=True)
        image = tf.squeeze(image, [0])
        # For normalization see https://github.com/google-research/big_transfer/issues/16

        image.set_shape((482, 482, 3))
        return image, image_cords

    dataset = tf.data.Dataset.from_tensor_slices((paths, idxs))
    dataset = dataset.map(parse_webp_imgs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=1)
    return dataset


def extract_features(src_tile_dir, next_pyr_dir_path, model_path, logger, lock):
    features, indices = list(), list()

    while not os.path.exists(next_pyr_dir_path):
        time.sleep(1)
    logger.debug("Found {} exists, continuing".format(next_pyr_dir_path))

    logger.debug("Loading model from {}".format(model_path))
    with lock:
        mem_manager.acquire_mem()
        time.sleep(10)
        logger.info("Buffer memory acquired'")
        bit_module = hub.KerasLayer(model_path)
        mem_manager.release_mem()

    nimgs_processed = 0
    tiles = os.listdir(src_tile_dir)
    logger.debug("Found {} to process".format(len(tiles)))

    if len(tiles) > 0:
        tile_paths, tile_idxs = list(), list()
        for src_filename in tiles:
            if '.webp' in src_filename:
                tile_paths.append(os.path.join(src_tile_dir, src_filename))
                name, ext = src_filename.split(".")
                c, r = name.split("_")
                tile_idxs.append((int(r), int(c)))

        with lock:
            logger.debug("Reading image into dataset")
            processed_data = read_images(tile_paths, tile_idxs, batch_size=8)

        logger.debug("Extracting features from images")
        for images, indexs in processed_data:
            with lock:
                bit_features = bit_module(images)
            for i in range(bit_features.shape[0]):
                features.append(bit_features[i])
                indices.append(indexs[i])
                nimgs_processed += 1
                if nimgs_processed % 100 == 0:
                    logger.debug("Processed images till now {}".format(nimgs_processed))

    logger.info("Processed all images {}".format(len(features)))
    feature_matrix = np.zeros((len(features), 2048), dtype=np.float32)
    for i in range(len(features)):
        feature_matrix[i, :] = features[i]
    idx_matrix = np.asarray(indices)

    return feature_matrix, idx_matrix


def parse_feature(feature_path, idxs):

    features = tf.io.read_file(feature_path)
    features = tf.io.decode_raw(features, tf.float16)
    features = tf.reshape(features, [-1, 2048])

    idxs = tf.io.read_file(idxs)
    idxs = tf.io.decode_raw(idxs, tf.int32)
    idxs = tf.reshape(idxs, [-1, 2])

    features = tf.convert_to_tensor()

    return features, idxs


def _predict(feature, embed_model, classifier_model, logger):

    feature = tf.convert_to_tensor(feature)
    feature = tf.expand_dims(feature, 0)

    with tf.GradientTape() as tape:
        # Compute activations of the last conv-layer and make the tape watch it
        embedding = embed_model(feature)

        tape.watch(embedding)
        preds = classifier_model(embedding)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, embedding)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    last_conv_layer_output = embedding.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, i] *= pooled_grads[i]

    logits = tf.nn.softmax(preds)

    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    if logits[0][0] > logits[0][1]:
        heatmap *= 0.0

    return logits[0], heatmap


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def predict_mil_hm(feature, model_path, logger):
    logger.info("Loading model from:{}".format(model_path))
    base_model = tf.keras.models.load_model(model_path,
                                            custom_objects={'GroupNormalization': tfa.layers.GroupNormalization,
                                                            'sensitivity': sensitivity, 'specificity': specificity})

    classifier_model = tf.keras.models.Model(base_model.inputs, base_model.get_layer('class').output)

    logger.info("Model loaded")
    feature = tf.convert_to_tensor(feature, tf.float16)
    feature = tf.reshape(feature, [-1, 2048])
    feature = tf.expand_dims(feature, 0)
    feature = tf.image.resize_with_crop_or_pad(feature, 1, 2500)
    feature = tf.reshape(feature, [-1, 2048])
    feature = tf.expand_dims(feature, 0)
    preds = classifier_model(feature)
    logits = tf.nn.softmax(preds)

    with tf.GradientTape() as tape:
        embed_model = tf.keras.models.Model(base_model.inputs,
                                            [base_model.get_layer('class').output, base_model.get_layer('lc').output])

        preds, embedding = embed_model(feature)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
        grads = tape.gradient(top_class_channel, embedding)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, embedding), axis=-1)
    heatmap = np.maximum(heatmap, 0.0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.squeeze(axis=0)

    if logits[0][0] > logits[0][1]:
        heatmap *= 0.0

    return logits[0], heatmap


def predict(feature_path, model_path, logger):

    logger.info("Loading model from:{}".format(model_path))
    base_model = tf.keras.models.load_model(model_path,
                                            custom_objects={'GroupNormalization':tfa.layers.GroupNormalization,
                                                            'sensitivity':sensitivity, 'specificity':specificity})

    embed_model = tf.keras.models.Model(base_model.inputs, base_model.get_layer('embed').output)
    classifier_input = tf.keras.Input(shape=base_model.get_layer('embed').output.shape[1:])
    cx = classifier_input
    for layer_name in ["pool", "l0", "class"]:
        cx = base_model.get_layer(layer_name)(cx)
    classifier_model = tf.keras.models.Model(classifier_input, cx)
    logit, heat_map = _predict(feature_path, embed_model, classifier_model, logger)
    return logit, heat_map

    # if isinstance(feature_path, list):
    #     logger.debug("Got list of features as input")
    #     all_logits, all_heatmaps = list(), list()
    #     for idx in range(len(feature_path)):
    #         logger.debug("Performing Grad-CAM operation on {}".format(idx))
    #         logit, heat_map = _predict(feature_path[idx], embed_model, classifier_model, logger)
    #         all_logits.append(logit)
    #         all_heatmaps.append(heat_map)
    #     return np.asarray(all_logits), all_heatmaps
    # else:
    #     logger.debug("Performing Grad-CAM operation on single feature")
    #     logit, heat_map = _predict(feature_path, embed_model, classifier_model, logger)
    #     return logit, heat_map


