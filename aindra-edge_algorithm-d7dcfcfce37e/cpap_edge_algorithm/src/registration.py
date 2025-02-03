import os
import numpy as np
from PIL import Image
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


def fftshift(x, axes=[1, 2]):
    if axes is None:
        axes = tf.tuple(tf.range(tf.rank(x)))
        shift = [dim // 2 for dim in tf.shape[x]]
    elif isinstance(axes, int):
        shift = tf.shape(x)[axes] // 2
    else:
        shift = [tf.shape(x)[ax] // 2 for ax in axes]

    return tf.roll(x, shift, axes)


def ifftshift(x, axes=[1, 2]):
    if axes is None:
        axes = tf.tuple(tf.range(tf.rank(x)))
        shift = [-(dim // 2) for dim in tf.shape[x]]
    elif isinstance(axes, int):
        shift = -(tf.shape(x)[axes] // 2)
    else:
        shift = [-(tf.shape(x)[ax] // 2) for ax in axes]

    return tf.roll(x, shift, axes)


def pair_tx(img1, img2):
    img1 = tf.squeeze(img1, [-1])
    img2 = tf.squeeze(img2, [-1])

    img1_complex = tf.cast(img1, tf.complex64)
    img2_complex = tf.cast(img2, tf.complex64)

    spectra_1 = tf.fft2d(img1_complex)
    midpoints_height = tf.shape(spectra_1)[1] / 2
    midpoints_width = tf.shape(spectra_1)[2] / 2

    midpoints_height = tf.cast(midpoints_height, tf.int32)
    midpoints_width = tf.cast(midpoints_width, tf.int32)

    spectra_2 = tf.fft2d(img2_complex)

    mask = tf.ones(tf.shape(spectra_1))
    slice = mask[:, midpoints_height - 30:midpoints_height + 30, midpoints_width - 30:midpoints_width + 30]
    slice = tf.zeros(tf.shape(slice))
    paddings = [[0, 0], [midpoints_height - 30, midpoints_height - 30], [midpoints_width - 30, midpoints_width - 30]]
    slice = tf.pad(slice, paddings, mode='constant', constant_values=1)
    mask = mask * slice
    mask = tf.cast(mask, tf.complex64)
    fshift_spectra_1 = fftshift(spectra_1)
    fshift_spectra_2 = fftshift(spectra_2)
    fshift_spectra_1 = fshift_spectra_1 * mask
    fshift_spectra_2 = fshift_spectra_2 * mask
    spectra_1 = ifftshift(fshift_spectra_1)
    spectra_2 = ifftshift(fshift_spectra_2)
    spectra_2 = tf.conj(spectra_2)

    ccf = spectra_1 * spectra_2

    ifft_pc = tf.ifft2d(ccf)
    ifft_pc = tf.abs(ifft_pc)

    dim = tf.reduce_prod(tf.shape(ifft_pc)[1:])
    vec_ifft = tf.reshape(ifft_pc, [-1, dim])
    max_pos_vec = tf.argmax(vec_ifft, axis=1)
    corr = tf.reduce_max(vec_ifft, axis=1)
    max_pos_vec = tf.cast(max_pos_vec, tf.int32)

    height = tf.shape(img1)[1]
    width = tf.shape(img1)[2]

    r_shift = max_pos_vec / width
    c_shift = max_pos_vec % width

    r_shift = tf.cast(r_shift, tf.int32)
    c_shift = tf.cast(c_shift, tf.int32)


    midpoints_height_mask = tf.cast(tf.math.greater(r_shift, midpoints_height), tf.int32)
    midpoints_width_mask = tf.cast(tf.math.greater(c_shift, midpoints_width), tf.int32)

    row_shift = r_shift - tf.shape(spectra_1)[1] * midpoints_height_mask
    col_shift = c_shift - tf.shape(spectra_1)[2] * midpoints_width_mask

    var_1 = tf.nn.moments(img1, axes=[1, 2])[1]
    var_2 = tf.nn.moments(img2, axes=[1, 2])[1]
    net_var = var_1 + var_2

    row_shift = tf.cast(row_shift, tf.float32)
    col_shift = tf.cast(col_shift, tf.float32)
    # net_var = tf.cast(net_var, tf.int32)

    txs = tf.stack([row_shift, col_shift, corr], axis=-1)
    return txs


# FIXME : Hardcoded 400 pixels as overlap, need to automate this


def triplet_parse(triplets):
    center_path = tf.read_file(triplets[0])
    left_path = tf.read_file(triplets[1])
    top_path = tf.read_file(triplets[2])

    center_img = tf.image.decode_jpeg(center_path, channels=1)
    left_img = tf.image.decode_jpeg(left_path, channels=1)
    top_img = tf.image.decode_jpeg(top_path, channels=1)

    center_left_patch = center_img[-400:, :, :]
    left_right_patch = left_img[:400, :, :]

    center_top_patch = center_img[:, -400:, :]
    top_bottom_patch = top_img[:, :400, :]

    center_left_patch = tf.image.convert_image_dtype(center_left_patch, tf.float32)
    left_right_patch = tf.image.convert_image_dtype(left_right_patch, tf.float32)
    center_top_patch = tf.image.convert_image_dtype(center_top_patch, tf.float32)
    top_bottom_patch = tf.image.convert_image_dtype(top_bottom_patch, tf.float32)

    return center_left_patch, left_right_patch, center_top_patch, top_bottom_patch, triplets[0], triplets[1], triplets[
        2]


def horiz_pair_parse(pair):
    ref_image = tf.read_file(pair[0])
    rel_image = tf.read_file(pair[1])
    ref_image = tf.image.decode_jpeg(ref_image, channels=1)
    rel_image = tf.image.decode_jpeg(rel_image, channels=1)

    ref_left_patch = ref_image[-400:, :, :]
    rel_right_patch = rel_image[:400, :, :]
    ref_left_patch = tf.image.convert_image_dtype(ref_left_patch, tf.float32)

    rel_right_patch = tf.image.convert_image_dtype(rel_right_patch, tf.float32)

    return ref_left_patch, pair[0], rel_right_patch, pair[1]


def vert_pair_parse(pair):
    ref_image = tf.read_file(pair[0])
    rel_image = tf.read_file(pair[1])

    ref_image = tf.image.decode_jpeg(ref_image, channels=1)
    rel_image = tf.image.decode_jpeg(rel_image, channels=1)

    ref_top_patch = ref_image[:, -400:, :]
    rel_bottom_patch = rel_image[:, :400, :]

    ref_top_patch = tf.image.convert_image_dtype(ref_top_patch, tf.float32)
    rel_bottom_patch = tf.image.convert_image_dtype(rel_bottom_patch, tf.float32)

    return ref_top_patch, pair[0], rel_bottom_patch, pair[1]


def tx_est_graph(batch_size):
    tx_graph = tf.Graph()
    with tx_graph.as_default():
        init = tf.global_variables_initializer()
        tx_graph.add_to_collection("init", init)
        # Compute translation of a triplet. The triplet as an center image, top image and a left image
        list_triplets = tf.placeholder(tf.string, shape=[None, 3])
        ds_triplets = tf.data.Dataset.from_tensor_slices(list_triplets)
        ds_triplets = ds_triplets.map(triplet_parse, num_parallel_calls=8)
        ds_triplets = ds_triplets.batch(batch_size)
        ds_triplets = ds_triplets.prefetch(buffer_size=20)
        triplet_iterator = ds_triplets.make_initializable_iterator()
        center_left_patch, left_right_patch, center_top_patch, top_bottom_patch, path_center, \
        path_left, path_top = triplet_iterator.get_next()
        left_tx = pair_tx(center_left_patch, left_right_patch)
        top_tx = pair_tx(center_top_patch, top_bottom_patch)
        triplet_txs = [path_center, path_left, left_tx, path_top, top_tx]

        tx_graph.add_to_collection("triplet_params", triplet_iterator)
        tx_graph.add_to_collection("triplet_params", list_triplets)
        tx_graph.add_to_collection("triplet_params", triplet_txs)

        # Horizontal tx computation for pairs
        list_hz_pairs = tf.placeholder(tf.string, [None, 2])
        ds_hz_pairs = tf.data.Dataset.from_tensor_slices(list_hz_pairs)
        ds_hz_pairs = ds_hz_pairs.map(horiz_pair_parse, num_parallel_calls=8)
        ds_hz_pairs = ds_hz_pairs.batch(batch_size)
        ds_hz_pairs = ds_hz_pairs.prefetch(buffer_size=20)
        hz_iterator = ds_hz_pairs.make_initializable_iterator()
        image_hz_ref, path_hz_ref, image_hz_rel, path_hz_rel = hz_iterator.get_next()
        hz_tx = pair_tx(image_hz_ref, image_hz_rel)
        hzpair_tx = [path_hz_ref, path_hz_rel, hz_tx]

        tx_graph.add_to_collection("horiz_params", hz_iterator)
        tx_graph.add_to_collection("horiz_params", list_hz_pairs)
        tx_graph.add_to_collection("horiz_params", hzpair_tx)

        # Vertical tx computation for pairs
        list_vt_pairs = tf.placeholder(tf.string, [None, 2])
        ds_vt_pairs = tf.data.Dataset.from_tensor_slices(list_vt_pairs)
        ds_vt_pairs = ds_vt_pairs.map(vert_pair_parse, num_parallel_calls=8)
        ds_vt_pairs = ds_vt_pairs.batch(batch_size)
        ds_vt_pairs = ds_vt_pairs.prefetch(buffer_size=20)
        vt_iterator = ds_vt_pairs.make_initializable_iterator()
        image_vt_ref, path_vt_ref, image_vt_rel, path_vt_rel = vt_iterator.get_next()
        vt_tx = pair_tx(image_vt_ref, image_vt_rel)
        vtpair_tx = [path_vt_ref, path_vt_rel, vt_tx]

        tx_graph.add_to_collection("vert_params", vt_iterator)
        tx_graph.add_to_collection("vert_params", list_vt_pairs)
        tx_graph.add_to_collection("vert_params", vtpair_tx)

    return tx_graph


def compute_triplet_txs(triplets, tx_graph, sess, pause_lock):
    triplet_iterator, list_triplets, triplet_txs = tx_graph.get_collection("triplet_params")

    tx_list = list()
    sess.run(triplet_iterator.initializer, feed_dict={list_triplets: triplets})
    while True:
        with pause_lock:
            try:
                txs = sess.run(triplet_txs)
                for i in range(txs[0].shape[0]):
                    tx_rows, tx_cols, var = txs[2][i]

                    tx_list.append((txs[0][i], txs[1][i], tx_rows, tx_cols, var))

                    tx_rows, tx_cols, var = txs[4][i]

                    tx_list.append((txs[0][i], txs[3][i], tx_rows, tx_cols, var))

            except tf.errors.OutOfRangeError:
                break
    return tx_list


def compute_horizontal_txs(tx_pairs, tx_graph, sess, pause_lock):
    iterator, pair_tensor, txs_tensor = tx_graph.get_collection("horiz_params")
    tx_list = list()
    sess.run(iterator.initializer, feed_dict={pair_tensor: tx_pairs})
    while True:
        with pause_lock:
            try:
                txs = sess.run(txs_tensor)
                for i in range(txs[0].shape[0]):
                    tx_rows, tx_cols, var = txs[2][i]
                    tx_list.append((txs[0][i], txs[1][i], tx_rows, tx_cols, var))

            except tf.errors.OutOfRangeError:
                break
    return tx_list


def compute_vertical_txs(tx_pairs, tx_graph, sess, pause_lock):
    iterator, pair_tensor, txs_tensor = tx_graph.get_collection("vert_params")
    tx_list = list()
    sess.run(iterator.initializer, feed_dict={pair_tensor: tx_pairs})
    while True:
        with pause_lock:
            try:
                txs = sess.run(txs_tensor)
                for i in range(txs[0].shape[0]):
                    _, img_rows = Image.open(txs[1][i]).size
                    tx_rows, tx_cols, var = txs[2][i]
                    tx_list.append((txs[0][i], txs[1][i], tx_rows, tx_cols, var))
            except tf.errors.OutOfRangeError:
                break
    return tx_list


def estimate_global_txs(pair_txs, wsi_shape):
    col_tx_mat = np.zeros(shape=(wsi_shape[0], wsi_shape[1]), dtype=np.float)
    row_tx_mat = np.zeros(shape=(wsi_shape[0], wsi_shape[1]), dtype=np.float)
    last_tx_row = 200
    last_tx_col = 990
    for pair in pair_txs:
        r1, c1, r2, c2, tx_row, tx_col, est_var = pair
        if r1 == r2 and abs(c1 - c2) == 1:
            if tx_row > 250 or tx_row < 100:
                tx_row = last_tx_row
            else:
                last_tx_row = tx_row
            if tx_row == 0:
                tx_row = 844
            else:
                tx_row = 1024 - tx_row
            row_tx_mat[r1, c1] = tx_row

        if abs(r1 - r2) == 1 and c1 == c2:
            if tx_col < 900 or tx_col > 1050:
                tx_col = last_tx_col
            else:
                last_tx_col = tx_col
            if tx_col == 0:
                tx_col = 1010
            col_tx_mat[r1, c1] = tx_col

    col_pos_mat = np.cumsum(col_tx_mat, axis=0)
    row_pos_mat = np.cumsum(row_tx_mat, axis=1)

    for ctr in range(0, row_pos_mat.shape[0] - 1, 1):
        if ctr % 2 == 0:
            if row_pos_mat[ctr, -1] > row_pos_mat[ctr + 1, -1]:
                row_pos_mat[ctr + 1] = row_pos_mat[ctr + 1] + abs(row_pos_mat[ctr, -1] - row_pos_mat[ctr + 1, -1])
            else:
                row_pos_mat[ctr + 1] = row_pos_mat[ctr + 1] - abs(row_pos_mat[ctr, -1] - row_pos_mat[ctr + 1, -1])
        else:
            row_pos_mat[ctr + 1, 0] = row_pos_mat[ctr + 1, 0] + row_pos_mat[ctr, 0]

    pos_mat = np.stack((col_pos_mat, row_pos_mat), axis=-1)
    return pos_mat


def compute_global_positions(txs_list, wsi_shape):
    """
    Computes the global positions from pairwise translations.
    The idea is that the translations can be represented as
    P_{i} - P_{i-1} = tx{i, i-1}
    Where P_{i} is the global position  of i_th image
    and tx{i, i-1} is the computed translations between P_{i} and P_{i-1}
    Writing this in matrix form

    [ 0 0 0 0 0 .... -1 1 0 0 ...
      0 0 0 0 0 .... 0 -1 1 0 ...
    ...] * [P_{0} ..... P_{i-1} P_{i} ...]^T = [tx{1, 0} .... tx{i, i-1} .... ]^T

    This can be solved using least squares to get the global positions
    :param txs_list: list [[file1, file2, r1, c1, r2, c2, tx_row, tx_col, est_var]]
    :param wsi_shape: shape of the actual wsi
    :return: translations, a list with [image, row, col, image_size_nrow, image_size_ncol]
    """

    pair_txs = list()
    file_dict = dict()

    for line in txs_list:
        file1, file2, r1, c1, r2, c2, tx_row, tx_col, est_var = line

        pair_txs.append([r1, c1, r2, c2, tx_row, tx_col, est_var])

        file_dict['{}_{}'.format(c1, r1)] = file1
        file_dict['{}_{}'.format(c2, r2)] = file2

    pos_mat = estimate_global_txs(pair_txs, wsi_shape=wsi_shape)
    pos_mat = np.asarray(pos_mat, dtype=int)

    tx_list = list()

    for r in range(pos_mat.shape[0]):
        for c in range(pos_mat.shape[1]):
            if '{}_{}'.format(c, r) in file_dict:
                im_col, im_row = Image.open(file_dict['{}_{}'.format(c, r)]).size
                tx_list.append(
                    [file_dict['{}_{}'.format(c, r)], (pos_mat[r, c, 0], pos_mat[r, c, 1]), (im_row, im_col)])

    return tx_list


def create_tx_pairs(src_path, c_max):
    src_images = os.listdir(src_path)
    tx_triplets = list()
    horizontal_tx_pairs = list()
    vertical_tx_pairs = list()
    ctr = 0
    for src_image in src_images:

        name, ext = src_image.split(".")
        c, r = name.split("_")
        c, r = int(c), int(r)

        central_image = os.path.join(src_path, src_image)
        left_image = os.path.join(src_path, "{}_{}.{}".format(c - 1, r, ext))
        top_image = os.path.join(src_path, "{}_{}.{}".format(c_max - 1 - c, r - 1, ext))
        ctr += 1
        if os.path.exists(left_image) and os.path.exists(top_image):
            tx_triplets.append([central_image, left_image, top_image])
        elif os.path.exists(left_image):
            horizontal_tx_pairs.append([central_image, left_image])
        elif os.path.exists(top_image):
            vertical_tx_pairs.append([central_image, top_image])

    return tx_triplets, horizontal_tx_pairs, vertical_tx_pairs


def estimate_wsi_txs(txs_file):
    pair_txs = list()
    with open(txs_file, "a") as tx_file:
        for line in tx_file:
            path_1, path_2, txs = line.split(',')
            pair_txs.append([path_1, path_2, txs])
