IMG_SIZE = 1080

SMPL_MODEL = '/root/pmt/thirdparty/octopus/assets/neutral_smpl.pkl'

LABELS_FULL = {
    'Sunglasses': [170, 0, 51],
    'LeftArm': [51, 170, 221],
    'RightArm': [0, 255, 255],
    'LeftLeg': [85, 255, 170],
    'RightLeg': [170, 255, 85],
    'LeftShoe': [255, 255, 0],
    'RightShoe': [255, 170, 0],
}

LABELS_CLOTHING = {
    'Face': [0, 0, 255],
    'Arms': [51, 170, 221],
    'Legs': [85, 255, 170],
    'Shoes': [255, 255, 0]
}

import os
import argparse
import cv2
import sys
import dirt
import numpy as np
import tensorflow as tf
import keras.backend as K
import scipy.sparse as sp

from keras.models import Model
from keras.callbacks import LambdaCallback
from keras.layers import Input, Flatten, Dense, Lambda, Conv2D, MaxPool2D, Average, Concatenate, Add, Reshape
from keras.initializers import RandomNormal
from keras.engine.topology import Layer
from keras.layers import initializers, activations

from glob import glob
from tqdm import tqdm
from dirt import matrices
from scipy.sparse.linalg.eigen.arpack import eigsh
from tensorflow.python.framework import ops

if sys.version_info[0] == 3:
    import _pickle as pkl
else:
    import cPickle as pkl

import pyopenpose as op
# use this below if the above line doesn't work
# sys.path.append('/path/to/where/is/openpose/folder')

body_25_reg = None
face_reg = None
_cache = None

def format_pose(pose_keypoints, resolution):
    # Assumes first person is subject
    pose = np.array(pose_keypoints[0])
    pose[:, 2] /= np.expand_dims(np.mean(pose[:, 2][pose[:, 2] > 0.1]), -1)
    pose = pose * np.array([2. / resolution[1], -2. / resolution[0], 1.]) + np.array([-1., 1., 0.])
    pose[:, 0] *= 1. * resolution[1] / resolution[0]

    return pose

def format_face(face_keypoints, resolution):
    # Assumes first person is subject
    face = np.array(face_keypoints[0])
    face = face * np.array([2. / resolution[1], -2. / resolution[0], 1.]) + np.array([-1., 1., 0.])
    face[:, 0] *= 1. * resolution[1] / resolution[0]

    return face

def openpose_process(input_dir, openpose_model_dir):
    params = {}

    # Configure openpose params
    params['image_dir'] = input_dir
    params['model_folder'] = openpose_model_dir
    params['face'] = True

    op_wrapper = op.WrapperPython()
    op_wrapper.configure(params)
    op_wrapper.start()

    # Process images in input dir
    paths = op.get_images_on_directory(input_dir)
    joints_2d = []
    face_2d = []

    for path in paths:
        datum = op.Datum()
        img = cv2.imread(path)
        resolution = img.shape[:2]
        datum.cvInputData = img

        op_wrapper.emplaceAndPop(op.VectorDatum([datum]))
        #op_wrapper.emplaceAndPop([datum])

        joints_2d.append(format_pose(datum.poseKeypoints, resolution))
        face_2d.append(format_face(datum.faceKeypoints, resolution))

    return joints_2d, face_2d

def get_bodypart_vertex_ids():
    global _cache

    if _cache is None:
        with open(os.path.join(os.path.dirname(__file__), 'assets/bodyparts.pkl'), 'rb') as fp:
            _cache = pkl.load(fp,encoding='iso-8859-1')

    return _cache

def regularize_laplace():
    reg = np.ones(6890)
    v_ids = get_bodypart_vertex_ids()

    reg[v_ids['face']] = 8.
    reg[v_ids['hand_l']] = 5.
    reg[v_ids['hand_r']] = 5.
    reg[v_ids['fingers_l']] = 8.
    reg[v_ids['fingers_r']] = 8.
    reg[v_ids['foot_l']] = 5.
    reg[v_ids['foot_r']] = 5.
    reg[v_ids['toes_l']] = 8.
    reg[v_ids['toes_r']] = 8.
    reg[v_ids['ear_l']] = 10.
    reg[v_ids['ear_r']] = 10.

    return reg

def laplace_mse(_, ypred):
    w = regularize_laplace()
    return K.mean(w[np.newaxis, :, np.newaxis] * K.square(ypred), axis=-1)

def write_frame_data_oct(vertices):
    '''Writes the frame data (camera calib. and vertices) to a pkl file

    Assumes all images have been preprocessed to be 1080x1080
    '''
    data = {
            'width': IMG_SIZE,
            'height': IMG_SIZE,
            'camera_f': [IMG_SIZE, IMG_SIZE],
            'camera_c': [IMG_SIZE / 2., IMG_SIZE / 2.],
            'vertices': vertices,
        }

    return data

def read_segmentation_octopus(file):
    segm = np.array(file)[:, :, ::-1]

    segm[np.all(segm == LABELS_FULL['Sunglasses'], axis=2)] = LABELS_CLOTHING['Face']
    segm[np.all(segm == LABELS_FULL['LeftArm'], axis=2)] = LABELS_CLOTHING['Arms']
    segm[np.all(segm == LABELS_FULL['RightArm'], axis=2)] = LABELS_CLOTHING['Arms']
    segm[np.all(segm == LABELS_FULL['LeftLeg'], axis=2)] = LABELS_CLOTHING['Legs']
    segm[np.all(segm == LABELS_FULL['RightLeg'], axis=2)] = LABELS_CLOTHING['Legs']
    segm[np.all(segm == LABELS_FULL['LeftShoe'], axis=2)] = LABELS_CLOTHING['Shoes']
    segm[np.all(segm == LABELS_FULL['RightShoe'], axis=2)] = LABELS_CLOTHING['Shoes']

    return segm[:, :, ::-1] / 255.

def regularize_symmetry():
    reg = np.ones(6890)
    v_ids = get_bodypart_vertex_ids()

    reg[v_ids['face']] = 10.
    reg[v_ids['hand_l']] = 10.
    reg[v_ids['hand_r']] = 10.
    reg[v_ids['foot_l']] = 10.
    reg[v_ids['foot_r']] = 10.
    reg[v_ids['ear_l']] = 5.
    reg[v_ids['ear_r']] = 5.

    return reg

def symmetry_mse(_, ypred):
    w = regularize_symmetry()

    idx = np.load(os.path.join(os.path.dirname(__file__), 'assets/vert_sym_idxs.npy'))
    ypred_mirror = tf.gather(ypred, idx, axis=1) * np.array([-1., 1., 1.]).astype(np.float32).reshape(1, 1, 3)

    return K.mean(w[np.newaxis, :, np.newaxis] * K.square(ypred - ypred_mirror), axis=-1)

def NameLayer(name):
    return Lambda(lambda i: i, name=name)

def sparse_to_tensor(x, dtype=tf.float32):
    coo = x.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, tf.convert_to_tensor(coo.data, dtype=dtype), coo.shape)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_dot_adj_batch(adj, x):
    # adj: V x V
    # x: B x V x N
    return tf.map_fn(lambda xx: tf.sparse_tensor_dense_matmul(adj, xx), x)

def perspective_projection(f, c, w, h, near=0.1, far=10., name=None):
    """Constructs a perspective projection matrix.
    This function returns a perspective projection matrix, using the OpenGL convention that the camera
    looks along the negative-z axis in view/camera space, and the positive-z axis in clip space.
    Multiplying view-space homogeneous coordinates by this matrix maps them into clip space.

    Returns:
        a 4x4 `Tensor` containing the projection matrix
    """

    with ops.name_scope(name, 'PerspectiveProjection', [f, c, w, h, near, far]) as scope:
        f = 0.5 * (f[0] + f[1])
        pixel_center_offset = 0.5
        right = (w - (c[0] + pixel_center_offset)) * (near / f)
        left = -(c[0] + pixel_center_offset) * (near / f)
        top = (c[1] + pixel_center_offset) * (near / f)
        bottom = -(h - c[1] + pixel_center_offset) * (near / f)

        elements = [
            [2. * near / (right - left), 0., (right + left) / (right - left), 0.],
            [0., 2. * near / (top - bottom), (top + bottom) / (top - bottom), 0.],
            [0., 0., -(far + near) / (far - near), -2. * far * near / (far - near)],
            [0., 0., -1., 0.]
        ]

        return tf.transpose(tf.convert_to_tensor(elements, dtype=tf.float32))

def reprojection(fl, cc, w, h):

    def _r(ytrue, ypred):
        b_size = tf.shape(ypred)[0]
        projection_matrix = perspective_projection(fl, cc, w, h, .1, 10)
        projection_matrix = tf.tile(tf.expand_dims(projection_matrix, 0), (b_size, 1, 1))

        ypred_h = tf.concat([ypred, tf.ones_like(ypred[:, :, -1:])], axis=2)
        ypred_proj = tf.matmul(ypred_h, projection_matrix)
        ypred_proj /= tf.expand_dims(ypred_proj[:, :, -1], -1)

        return K.mean(K.square((ytrue[:, :, :2] - ypred_proj[:, :, :2]) * tf.expand_dims(ytrue[:, :, 2], -1)))

    return _r

def render_colored_batch(m_v, m_f, m_vc, width, height, camera_f, camera_c, bgcolor=np.zeros(3, dtype=np.float32),
                         num_channels=3, camera_t=np.zeros(3, dtype=np.float32),
                         camera_rt=np.zeros(3, dtype=np.float32), name=None):
    with ops.name_scope(name, "render_batch", [m_v]) as name:
        assert (num_channels == m_vc.shape[-1] == bgcolor.shape[0])

        projection_matrix = perspective_projection(camera_f, camera_c, width, height, .1, 10)

        view_matrix = matrices.compose(
            matrices.rodrigues(camera_rt.astype(np.float32)),
            matrices.translation(camera_t.astype(np.float32)),
        )

        bg = tf.tile(bgcolor.astype(np.float32)[np.newaxis, np.newaxis, np.newaxis, :],
                     (tf.shape(m_v)[0], height, width, 1))
        m_vc = tf.tile(tf.cast(m_vc, tf.float32)[np.newaxis, ...], (tf.shape(m_v)[0], 1, 1))

        m_v = tf.cast(m_v, tf.float32)
        m_v = tf.concat([m_v, tf.ones_like(m_v[:, :, -1:])], axis=2)
        m_v = tf.matmul(m_v, tf.tile(view_matrix[np.newaxis, ...], (tf.shape(m_v)[0], 1, 1)))
        m_v = tf.matmul(m_v, tf.tile(projection_matrix[np.newaxis, ...], (tf.shape(m_v)[0], 1, 1)))

        m_f = tf.tile(tf.cast(m_f, tf.int32)[np.newaxis, ...], (tf.shape(m_v)[0], 1, 1))

        return dirt.rasterise_batch(bg, m_v, m_vc, m_f, name=name)

def sparse_dense_matmul_batch(a, b):
    num_b = tf.shape(b)[0]
    shape = a.dense_shape

    indices = tf.reshape(a.indices, (num_b, -1, 3))
    values = tf.reshape(a.values, (num_b, -1))

    def matmul(arguments):
        i, bb=arguments
        sp = tf.SparseTensor(indices[i, :, 1:], values[i], shape[1:])
        return i, tf.sparse_tensor_dense_matmul(sp, bb)

    _, p = tf.map_fn(matmul, (tf.range(num_b), b))

    return p

def batch_laplacian(v, f):
    # v: B x N x 3
    # f: M x 3

    num_b = tf.shape(v)[0]
    num_v = tf.shape(v)[1]
    num_f = tf.shape(f)[0]

    v_a = f[:, 0]
    v_b = f[:, 1]
    v_c = f[:, 2]

    a = tf.gather(v, v_a, axis=1)
    b = tf.gather(v, v_b, axis=1)
    c = tf.gather(v, v_c, axis=1)

    ab = a - b
    bc = b - c
    ca = c - a

    cot_a = -1 * tf.reduce_sum(ab * ca, axis=2) / tf.sqrt(tf.reduce_sum(tf.cross(ab, ca) ** 2, axis=-1))
    cot_b = -1 * tf.reduce_sum(bc * ab, axis=2) / tf.sqrt(tf.reduce_sum(tf.cross(bc, ab) ** 2, axis=-1))
    cot_c = -1 * tf.reduce_sum(ca * bc, axis=2) / tf.sqrt(tf.reduce_sum(tf.cross(ca, bc) ** 2, axis=-1))

    I = tf.tile(tf.expand_dims(tf.concat((v_a, v_c, v_a, v_b, v_b, v_c), axis=0), 0), (num_b, 1))
    J = tf.tile(tf.expand_dims(tf.concat((v_c, v_a, v_b, v_a, v_c, v_b), axis=0), 0), (num_b, 1))

    W = 0.5 * tf.concat((cot_b, cot_b, cot_c, cot_c, cot_a, cot_a), axis=1)

    batch_dim = tf.tile(tf.expand_dims(tf.range(num_b), 1), (1, num_f * 6))

    indices = tf.reshape(tf.stack((batch_dim, J, I), axis=2), (num_b, 6, -1, 3))
    W = tf.reshape(W, (num_b, 6, -1))

    l_indices = [tf.cast(tf.reshape(indices[:, i], (-1, 3)), tf.int64) for i in range(6)]
    shape = tf.cast(tf.stack((num_b, num_v, num_v)), tf.int64)
    sp_L_raw = [tf.sparse_reorder(tf.SparseTensor(l_indices[i], tf.reshape(W[:, i], (-1,)), shape)) for i in range(6)]

    L = sp_L_raw[0]
    for i in range(1, 6):
        L = tf.sparse_add(L, sp_L_raw[i])

    dia_values = tf.sparse_reduce_sum(L, axis=-1) * -1

    I = tf.tile(tf.expand_dims(tf.range(num_v), 0), (num_b, 1))
    batch_dim = tf.tile(tf.expand_dims(tf.range(num_b), 1), (1, num_v))
    indices = tf.reshape(tf.stack((batch_dim, I, I), axis=2), (-1, 3))

    dia = tf.sparse_reorder(tf.SparseTensor(tf.cast(indices, tf.int64), tf.reshape(dia_values, (-1,)), shape))

    return tf.sparse_add(L, dia)

def compute_laplacian_diff(v0, v1, f):
    L0 = batch_laplacian(v0, f)
    L1 = batch_laplacian(v1, f)

    return sparse_dense_matmul_batch(L0, v0) - sparse_dense_matmul_batch(L1, v1)

def sparse_dense_matmul_batch_tile(a, b):
    return tf.map_fn(lambda x: tf.sparse_tensor_dense_matmul(a, x), b)

def joints_body25(v):
    global body_25_reg

    if body_25_reg is None:

        body_25_reg = sparse_to_tensor(
            pkl.load(open(os.path.join(os.path.dirname(__file__), 'assets/J_regressor.pkl'), 'rb'),encoding='iso-8859-1').T
        )

    return sparse_dense_matmul_batch_tile(body_25_reg, v)

def batch_global_rigid_transformation(Rs, Js, parent, rotate_base=False):
    """
    Computes absolute joint locations given pose.

    rotate_base: if True, rotates the global rotation by 90 deg in x axis.
    if False, this is the original SMPL coordinate.

    Args:
      Rs: N x 24 x 3 x 3 rotation vector of K joints
      Js: N x 24 x 3, joint locations before posing
      parent: 24 holding the parent id for each index

    Returns
      new_J : `Tensor`: N x 24 x 3 location of absolute joints
      A     : `Tensor`: N x 24 4 x 4 relative joint transformations for LBS.
    """
    with tf.name_scope("batch_forward_kinematics", values=[Rs, Js]):
        N = tf.shape(Rs)[0]
        if rotate_base:
            #print('Flipping the SMPL coordinate frame!!!!')
            rot_x = tf.constant(
                [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=Rs.dtype)
            rot_x = tf.reshape(tf.tile(rot_x, [N, 1]), [N, 3, 3])
            root_rotation = tf.matmul(Rs[:, 0, :, :], rot_x)
        else:
            root_rotation = Rs[:, 0, :, :]

        # Now Js is N x 24 x 3 x 1
        Js = tf.expand_dims(Js, -1)

        def make_A(R, t, name=None):
            # Rs is N x 3 x 3, ts is N x 3 x 1
            with tf.name_scope(name, "Make_A", [R, t]):
                R_homo = tf.pad(R, [[0, 0], [0, 1], [0, 0]])
                t_homo = tf.concat([t, tf.ones([N, 1, 1])], 1)
                return tf.concat([R_homo, t_homo], 2)

        A0 = make_A(root_rotation, Js[:, 0])
        results = [A0]
        for i in range(1, parent.shape[0]):
            j_here = Js[:, i] - Js[:, parent[i]]
            A_here = make_A(Rs[:, i], j_here)
            res_here = tf.matmul(
                results[parent[i]], A_here, name="propA%d" % i)
            results.append(res_here)

        # 10 x 24 x 4 x 4
        results = tf.stack(results, axis=1)

        new_J = results[:, :, :3, 3]

        # --- Compute relative A: Skinning is based on
        # how much the bone moved (not the final location of the bone)
        # but (final_bone - init_bone)
        # ---
        Js_w0 = tf.concat([Js, tf.zeros([N, 24, 1, 1])], 2)
        init_bone = tf.matmul(results, Js_w0)
        # Append empty 4 x 3:
        init_bone = tf.pad(init_bone, [[0, 0], [0, 0], [0, 0], [3, 0]])
        A = results - init_bone

        return new_J, A

def face_landmarks(v):
    global face_reg

    if face_reg is None:
        face_reg = sparse_to_tensor(
            pkl.load(open(os.path.join(os.path.dirname(__file__), 'assets/face_regressor.pkl'), 'rb'),encoding='iso-8859-1').T
        )

    return sparse_dense_matmul_batch_tile(face_reg, v)

def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r

def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    # print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0], dtype=adj.dtype) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0], dtype=adj.dtype)

    t_k = list()
    t_k.append(sp.eye(adj.shape[0], dtype=adj.dtype))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True, dtype=adj.dtype)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return t_k

def batch_skew(vec, batch_size=None):
    """
    vec is N x 3, batch_size is int

    returns N x 3 x 3. Skew_sym version of each matrix.
    """
    with tf.name_scope("batch_skew", values=[vec]):
        if batch_size is None:
            batch_size = vec.shape.as_list()[0]
        col_inds = tf.constant([1, 2, 3, 5, 6, 7])
        indices = tf.reshape(
            tf.reshape(tf.range(0, batch_size) * 9, [-1, 1]) + col_inds,
            [-1, 1])
        updates = tf.reshape(
            tf.stack(
                [
                    -vec[:, 2], vec[:, 1], vec[:, 2], -vec[:, 0], -vec[:, 1],
                    vec[:, 0]
                ],
                axis=1), [-1])
        out_shape = [batch_size * 9]
        res = tf.scatter_nd(indices, updates, out_shape)
        res = tf.reshape(res, [batch_size, 3, 3])

        return res

def batch_rodrigues(theta, name=None):
    """
    Theta is N x 3
    """
    with tf.name_scope(name, "batch_rodrigues", [theta]):
        batch_size = tf.shape(theta)[0]

        # angle = tf.norm(theta, axis=1)
        # r = tf.expand_dims(tf.div(theta, tf.expand_dims(angle + 1e-8, -1)), -1)
        # angle = tf.expand_dims(tf.norm(theta, axis=1) + 1e-8, -1)
        angle = tf.expand_dims(tf.norm(theta + 1e-8, axis=1), -1)
        r = tf.expand_dims(tf.div(theta, angle), -1)

        angle = tf.expand_dims(angle, -1)
        cos = tf.cos(angle)
        sin = tf.sin(angle)

        outer = tf.matmul(r, r, transpose_b=True, name="outer")

        eyes = tf.tile(tf.expand_dims(tf.eye(3), 0), [batch_size, 1, 1])
        R = cos * eyes + (1 - cos) * outer + sin * batch_skew(
            r, batch_size=batch_size)
        return R

class GraphConvolution(Layer):

    def __init__(self, output_dim, support, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
        self.output_dim = output_dim
        self.support = list(support)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernels = []
        self.bias = None

        super(GraphConvolution, self).__init__(**kwargs)

    def build(self, input_shape):
        for i in range(len(self.support)):
            self.kernels.append(self.add_weight(name='kernel',
                                                shape=(input_shape[2], self.output_dim),
                                                initializer=self.kernel_initializer,
                                                trainable=True))

        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        trainable=True)

        super(GraphConvolution, self).build(input_shape)

    def call(self, x):
        supports = list()
        for s, k in zip(self.support, self.kernels):
            pre_sup = K.dot(x, k)
            supports.append(sparse_dot_adj_batch(s, pre_sup))

        output = supports[0]
        for i in range(1, len(supports)):
            output += supports[i]

        if self.use_bias:
            output += self.bias

        if self.activation is not None:
            return self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.output_dim

class RenderLayer(Layer):

    def __init__(self, width, height, num_channels, vc, bgcolor, f, camera_f, camera_c,
                 camera_t=np.zeros(3), camera_rt=np.zeros(3), **kwargs):
        assert(num_channels == vc.shape[-1] == bgcolor.shape[0])

        self.width = width
        self.height = height
        self.num_channels = num_channels
        self.vc = np.array(vc).astype(np.float32)
        self.bgcolor = np.array(bgcolor).astype(np.float32)
        self.f = np.array(f).astype(np.int32)
        self.camera_f = np.array(camera_f).astype(np.float32)
        self.camera_c = np.array(camera_c).astype(np.float32)
        self.camera_t = np.array(camera_t).astype(np.float32)
        self.camera_rt = np.array(camera_rt).astype(np.float32)

        super(RenderLayer, self).__init__(**kwargs)

    def call(self, v):
        return render_colored_batch(m_v=v, m_f=self.f, m_vc=self.vc, width=self.width, height=self.height,
                                    camera_f=self.camera_f, camera_c=self.camera_c, num_channels=self.num_channels,
                                    camera_t=self.camera_t, camera_rt=self.camera_rt, bgcolor=self.bgcolor)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.height, self.width, self.num_channels

class SMPL(object):
    def __init__(self, pkl_path, theta_in_rodrigues=True, theta_is_perfect_rotmtx=True, dtype=tf.float32):
        """
        pkl_path is the path to a SMPL model
        """
        # -- Load SMPL params --
        with open(pkl_path, 'rb') as f:
            dd = pkl.load(f,encoding='iso-8859-1')
        # Mean template vertices
        self.v_template = tf.Variable(
            undo_chumpy(dd['v_template']),
            name='v_template',
            dtype=dtype,
            trainable=False)
        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template.shape[0].value, 3]
        self.num_betas = dd['shapedirs'].shape[-1]
        # Shape blend shape basis: 6980 x 3 x 10
        # reshaped to 6980*30 x 10, transposed to 10x6980*3
        shapedir = np.reshape(
            undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T
        self.shapedirs = tf.Variable(
            shapedir, name='shapedirs', dtype=dtype, trainable=False)

        # Regressor for joint locations given shape - 6890 x 24
        self.J_regressor = sparse_to_tensor(dd['J_regressor'], dtype=dtype)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
        num_pose_basis = dd['posedirs'].shape[-1]
        # 207 x 20670
        posedirs = np.reshape(
            undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
        self.posedirs = tf.Variable(
            posedirs, name='posedirs', dtype=dtype, trainable=False)

        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)

        # LBS weights
        self.weights = tf.Variable(
            undo_chumpy(dd['weights']),
            name='lbs_weights',
            dtype=dtype,
            trainable=False)

        # expect theta in rodrigues form
        self.theta_in_rodrigues = theta_in_rodrigues

        # if in matrix form, is it already rotmax?
        self.theta_is_perfect_rotmtx = theta_is_perfect_rotmtx

    def __call__(self, theta, beta, trans, v_personal, name=None):
        """
        Obtain SMPL with shape (beta) & pose (theta) inputs.
        Theta includes the global rotation.
        Args:
          beta: N x 10
          theta: N x 72 (with 3-D axis-angle rep)

        Updates:
        self.J_transformed: N x 24 x 3 joint location after shaping
                 & posing with beta and theta
        Returns:
          - joints: N x 19 or 14 x 3 joint locations depending on joint_type
        If get_skin is True, also returns
          - Verts: N x 6980 x 3
        """

        with tf.name_scope(name, "smpl_main", [beta, theta, trans, v_personal]):
            num_batch = tf.shape(beta)[0]

            # 1. Add shape blend shapes
            # (N x 10) x (10 x 6890*3) = N x 6890 x 3
            v_shaped_scaled = tf.reshape(
                tf.matmul(beta, self.shapedirs, name='shape_bs'),
                [-1, self.size[0], self.size[1]]) + self.v_template

            body_height = (v_shaped_scaled[:, 2802, 1] + v_shaped_scaled[:, 6262, 1]) - (v_shaped_scaled[:, 2237, 1] + v_shaped_scaled[:, 6728, 1])
            scale = tf.reshape(1.66 / body_height, (-1, 1, 1))

            self.v_shaped = scale * v_shaped_scaled
            self.v_shaped_personal = self.v_shaped + v_personal

            # 2. Infer shape-dependent joint locations.
            Jx = tf.transpose(tf.sparse_tensor_dense_matmul(self.J_regressor, tf.transpose(v_shaped_scaled[:, :, 0])))
            Jy = tf.transpose(tf.sparse_tensor_dense_matmul(self.J_regressor, tf.transpose(v_shaped_scaled[:, :, 1])))
            Jz = tf.transpose(tf.sparse_tensor_dense_matmul(self.J_regressor, tf.transpose(v_shaped_scaled[:, :, 2])))
            J = scale * tf.stack([Jx, Jy, Jz], axis=2)

            # 3. Add pose blend shapes
            # N x 24 x 3 x 3
            if self.theta_in_rodrigues:
                Rs = tf.reshape(
                    batch_rodrigues(tf.reshape(theta, [-1, 3])), [-1, 24, 3, 3])
            else:
                if self.theta_is_perfect_rotmtx:
                    Rs = theta
                else:
                    s, u, v = tf.svd(theta)
                    Rs = tf.matmul(u, tf.transpose(v, perm=[0, 1, 3, 2]))

            with tf.name_scope("lrotmin"):
                # Ignore global rotation.
                pose_feature = tf.reshape(Rs[:, 1:, :, :] - tf.eye(3), [-1, 207])

            # (N x 207) x (207, 20670) -> N x 6890 x 3
            self.v_posed = tf.reshape(
                tf.matmul(pose_feature, self.posedirs),
                [-1, self.size[0], self.size[1]]) + self.v_shaped_personal

            #4. Get the global joint location
            self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents)
            self.J_transformed += tf.expand_dims(trans, axis=1)

            # 5. Do skinning:
            # W is N x 6890 x 24
            W = tf.reshape(
                tf.tile(self.weights, [num_batch, 1]), [num_batch, -1, 24])
            # (N x 6890 x 24) x (N x 24 x 16)
            T = tf.reshape(
                tf.matmul(W, tf.reshape(A, [num_batch, 24, 16])),
                [num_batch, -1, 4, 4])
            v_posed_homo = tf.concat(
                [self.v_posed, tf.ones([num_batch, self.v_posed.shape[1], 1])], 2)
            v_homo = tf.matmul(T, tf.expand_dims(v_posed_homo, -1))

            verts = v_homo[:, :, :3, 0]
            verts_t = verts + tf.expand_dims(trans, axis=1)

            return verts_t

class SmplBody25FaceLayer(Layer):

    def __init__(self, model=SMPL_MODEL, theta_in_rodrigues=False, theta_is_perfect_rotmtx=True, **kwargs):
        self.smpl = SMPL(model, theta_in_rodrigues, theta_is_perfect_rotmtx)
        super(SmplBody25FaceLayer, self).__init__(**kwargs)

    def call(self, arguments):
        pose, betas, trans=arguments
        v_personal = tf.tile(tf.zeros((1, 6890, 3)), (tf.shape(betas)[0], 1, 1))

        v = self.smpl(pose, betas, trans, v_personal)

        return tf.concat((joints_body25(v), face_landmarks(v)), axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 95, 3

class SmplTPoseLayer(Layer):

    def __init__(self, model=SMPL_MODEL, theta_in_rodrigues=False, theta_is_perfect_rotmtx=True, **kwargs):
        self.smpl = SMPL(model, theta_in_rodrigues, theta_is_perfect_rotmtx)
        super(SmplTPoseLayer, self).__init__(**kwargs)

    def call(self, arguments):
        pose, betas, trans, v_personal=arguments
        verts = self.smpl(pose, betas, trans, v_personal)

        return [verts, self.smpl.v_shaped_personal, self.smpl.v_shaped]

    def compute_output_shape(self, input_shape):
        shape = input_shape[0][0], 6890, 3

        return [shape, shape, shape]

class Octopus(object):
    def __init__(self, num=8, img_size=1080):
        self.num = num
        self.img_size = img_size
        self.inputs = []
        self.poses = []
        self.ts = []

        images = [Input(shape=(self.img_size, self.img_size, 3), name='image_{}'.format(i)) for i in range(self.num)]
        Js = [Input(shape=(25, 3), name='J_2d_{}'.format(i)) for i in range(self.num)]

        self.inputs.extend(images)
        self.inputs.extend(Js)

        pose_raw = np.load(os.path.join(os.path.dirname(__file__), 'assets/mean_a_pose.npy'))
        pose_raw[:3] = 0.
        pose = tf.reshape(batch_rodrigues(pose_raw.reshape(-1, 3).astype(np.float32)), (-1, ))
        trans = np.array([0., 0.2, -2.3])

        batch_size = tf.shape(images[0])[0]

        conv2d_0 = Conv2D(8, (3, 3), strides=(2, 2), activation='relu', kernel_initializer='he_normal', trainable=False)
        maxpool_0 = MaxPool2D((2, 2))

        conv2d_1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', trainable=False)
        maxpool_1 = MaxPool2D((2, 2))

        conv2d_2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', trainable=False)
        maxpool_2 = MaxPool2D((2, 2))

        conv2d_3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', trainable=False)
        maxpool_3 = MaxPool2D((2, 2))

        conv2d_4 = Conv2D(128, (3, 3), trainable=False)
        maxpool_4 = MaxPool2D((2, 2))

        flat = Flatten()
        self.image_features = flat

        latent_code = Dense(20, name='latent_shape')

        pose_trans = tf.tile(tf.expand_dims(tf.concat((trans, pose), axis=0), 0), (batch_size, 1))
        posetrans_init = Input(tensor=pose_trans, name='posetrans_init')
        self.inputs.append(posetrans_init)

        J_flat = Flatten()
        concat_pose = Concatenate()

        latent_pose_from_I = Dense(200, name='latent_pose_from_I', activation='relu', trainable=False)
        latent_pose_from_J = Dense(200, name='latent_pose_from_J', activation='relu', trainable=False)
        latent_pose = Dense(100, name='latent_pose')
        posetrans_res = Dense(24 * 3 * 3 + 3, name='posetrans_res',
                              kernel_initializer=RandomNormal(stddev=0.01), trainable=False)
        posetrans = Add(name='posetrans')

        dense_layers = []

        for i, (J, image) in enumerate(zip(Js, images)):
            conv2d_0_i = conv2d_0(image)
            maxpool_0_i = maxpool_0(conv2d_0_i)

            conv2d_1_i = conv2d_1(maxpool_0_i)
            maxpool_1_i = maxpool_1(conv2d_1_i)

            conv2d_2_i = conv2d_2(maxpool_1_i)
            maxpool_2_i = maxpool_2(conv2d_2_i)

            conv2d_3_i = conv2d_3(maxpool_2_i)
            maxpool_3_i = maxpool_3(conv2d_3_i)

            conv2d_4_i = conv2d_4(maxpool_3_i)
            maxpool_4_i = maxpool_4(conv2d_4_i)

            # shape
            flat_i = flat(maxpool_4_i)

            latent_code_i = latent_code(flat_i)
            dense_layers.append(latent_code_i)

            # pose
            J_flat_i = J_flat(J)
            latent_pose_from_I_i = latent_pose_from_I(flat_i)
            latent_pose_from_J_i = latent_pose_from_J(J_flat_i)

            concat_pose_i = concat_pose([latent_pose_from_I_i, latent_pose_from_J_i])
            latent_pose_i = latent_pose(concat_pose_i)
            posetrans_res_i = posetrans_res(latent_pose_i)
            posetrans_i = posetrans([posetrans_res_i, posetrans_init])

            self.poses.append(
                Lambda(lambda x: tf.reshape(x[:, 3:], (-1, 24, 3, 3)), name='pose_{}'.format(i))(posetrans_i)
            )
            self.ts.append(
                Lambda(lambda x: x[:, :3], name='trans_{}'.format(i))(posetrans_i)
            )

        if self.num > 1:
            self.dense_merged = Average(name='merged_latent_shape')(dense_layers)
        else:
            self.dense_merged = NameLayer(name='merged_latent_shape')(dense_layers[0])

        # betas
        self.betas = Dense(10, name='betas', trainable=False)(self.dense_merged)

        with open(os.path.join(os.path.dirname(__file__), 'assets/smpl_sampling.pkl'), 'rb') as f:
            sampling = pkl.load(f,encoding='iso-8859-1')

        M = sampling['meshes']
        U = sampling['up']
        D = sampling['down']
        A = sampling['adjacency']

        self.faces = M[0]['f'].astype(np.int32)

        low_res = D[-1].shape[0]
        tf_U = [sparse_to_tensor(u) for u in U]
        tf_A = [map(sparse_to_tensor, chebyshev_polynomials(a, 3)) for a in A]

        shape_features_dense = Dense(low_res * 64, kernel_initializer=RandomNormal(stddev=0.003),
                                     name='shape_features_flat')(self.dense_merged)
        shape_features = Reshape((low_res, 64), name="shape_features")(shape_features_dense)

        conv_l3 = GraphConvolution(32, tf_A[3], activation='relu', name='conv_l3', trainable=False)(shape_features)
        unpool_l2 = Lambda(lambda v: sparse_dot_adj_batch(tf_U[2], v), name='unpool_l2')(conv_l3)
        conv_l2 = GraphConvolution(16, tf_A[2], activation='relu', name='conv_l2', trainable=False)(unpool_l2)
        unpool_l1 = Lambda(lambda v: sparse_dot_adj_batch(tf_U[1], v), name='unpool_l1')(conv_l2)
        conv_l1 = GraphConvolution(16, tf_A[1], activation='relu', name='conv_l1', trainable=False)(unpool_l1)
        unpool_l0 = Lambda(lambda v: sparse_dot_adj_batch(tf_U[0], v), name='unpool_l0')(conv_l1)
        conv_l0 = GraphConvolution(3, tf_A[0], activation='tanh', name='offsets_pre')(unpool_l0)

        self.offsets = Lambda(lambda x: x / 10., name='offsets')(conv_l0)

        smpl = SmplTPoseLayer(theta_in_rodrigues=False, theta_is_perfect_rotmtx=False)
        smpls = [NameLayer('smpl_{}'.format(i))(smpl([p, self.betas, t, self.offsets])) for i, (p, t) in
                 enumerate(zip(self.poses, self.ts))]

        self.vertices = [Lambda(lambda s: s[0], name='vertices_{}'.format(i))(smpl) for i, smpl in enumerate(smpls)]

        # we only need one instance per batch for laplace
        self.vertices_tposed = Lambda(lambda s: s[1], name='vertices_tposed')(smpls[0])
        vertices_naked = Lambda(lambda s: s[2], name='vertices_naked')(smpls[0])
        #function_laplacian=
        def laplacian_function(x,faces=self.faces):v0,v1=x;return compute_laplacian_diff(v0, v1, faces)
        self.laplacian = Lambda(laplacian_function, name='laplacian')([self.vertices_tposed, vertices_naked])
        self.symmetry = NameLayer('symmetry')(self.vertices_tposed)

        l = SmplBody25FaceLayer(theta_in_rodrigues=False, theta_is_perfect_rotmtx=False)
        kps = [NameLayer('kps_{}'.format(i))(l([p, self.betas, t]))
               for i, (p, t) in enumerate(zip(self.poses, self.ts))]

        self.Js = [Lambda(lambda jj: jj[:, :25], name='J_reproj_{}'.format(i))(j) for i, j in enumerate(kps)]
        self.face_kps = [Lambda(lambda jj: jj[:, 25:], name='face_reproj_{}'.format(i))(j) for i, j in enumerate(kps)]

        self.repr_loss = reprojection([self.img_size, self.img_size],
                                      [self.img_size / 2., self.img_size / 2.],
                                      self.img_size, self.img_size)

        renderer = RenderLayer(self.img_size, self.img_size, 1, np.ones((6890, 1)), np.zeros(1), self.faces,
                               [self.img_size, self.img_size], [self.img_size / 2., self.img_size / 2.],
                               name='render_layer')
        self.rendered = [NameLayer('rendered_{}'.format(i))(renderer(v)) for i, v in enumerate(self.vertices)]

        self.inference_model = Model(
            inputs=self.inputs,
            outputs=[self.vertices_tposed] + self.vertices + [self.betas, self.offsets] + self.poses + self.ts
        )

        self.opt_pose_model = Model(
            inputs=self.inputs,
            outputs=self.Js
        )

        opt_pose_loss = {'J_reproj_{}'.format(i): self.repr_loss for i in range(self.num)}
        self.opt_pose_model.compile(loss=opt_pose_loss, optimizer='adam')

        self.opt_shape_model = Model(
            inputs=self.inputs,
            outputs=self.Js + self.face_kps + self.rendered + [self.symmetry, self.laplacian]
        )

        opt_shape_loss = {
            'laplacian': laplace_mse,
            'symmetry': symmetry_mse,
        }
        opt_shape_weights = {
            'laplacian': 100. * self.num,
            'symmetry': 50. * self.num,
        }

        for i in range(self.num):
            opt_shape_loss['rendered_{}'.format(i)] = 'mse'
            opt_shape_weights['rendered_{}'.format(i)] = 1.

            opt_shape_loss['J_reproj_{}'.format(i)] = self.repr_loss
            opt_shape_weights['J_reproj_{}'.format(i)] = 50.

            opt_shape_loss['face_reproj_{}'.format(i)] = self.repr_loss
            opt_shape_weights['face_reproj_{}'.format(i)] = 10. * self.num

        self.opt_shape_model.compile(loss=opt_shape_loss, loss_weights=opt_shape_weights, optimizer='adam')

    def load(self, checkpoint_path):
        self.inference_model.load_weights(checkpoint_path, by_name=True)

    def opt_pose(self, segmentations, joints_2d, opt_steps):
        data = {}
        supervision = {}

        for i in range(self.num):
            data['image_{}'.format(i)] = np.tile(
                np.float32(segmentations[i].reshape((1, self.img_size, self.img_size, -1))),
                (opt_steps, 1, 1, 1)
            )
            data['J_2d_{}'.format(i)] = np.tile(
                np.float32(np.expand_dims(joints_2d[i], 0)),
                (opt_steps, 1, 1)
            )
            supervision['J_reproj_{}'.format(i)] = np.tile(
                np.float32(np.expand_dims(joints_2d[i], 0)),
                (opt_steps, 1, 1)
            )

        self.opt_pose_model.fit(
            data, supervision,
            batch_size=1, epochs=1, verbose=0,
        )

    def opt_shape(self, segmentations, joints_2d, face_kps, opt_steps):
        data = {}
        supervision = {
            'laplacian': np.zeros((opt_steps, 6890, 3)),
            'symmetry': np.zeros((opt_steps, 6890, 3)),
        }

        for i in range(self.num):
            data['image_{}'.format(i)] = np.tile(
                np.float32(segmentations[i].reshape((1, self.img_size, self.img_size, -1))),
                (opt_steps, 1, 1, 1)
            )
            data['J_2d_{}'.format(i)] = np.tile(
                np.float32(np.expand_dims(joints_2d[i], 0)),
                (opt_steps, 1, 1)
            )

            supervision['J_reproj_{}'.format(i)] = np.tile(
                np.float32(np.expand_dims(joints_2d[i], 0)),
                (opt_steps, 1, 1)
            )
            supervision['face_reproj_{}'.format(i)] = np.tile(
                np.float32(np.expand_dims(face_kps[i], 0)),
                (opt_steps, 1, 1)
            )
            supervision['rendered_{}'.format(i)] = np.tile(
                np.expand_dims(
                    np.any(np.float32(segmentations[i].reshape((1, self.img_size, self.img_size, -1)) > 0), axis=-1),
                    -1),
                (opt_steps, 1, 1, 1)
            )

        self.opt_shape_model.fit(
            data, supervision,
            batch_size=1, epochs=1, verbose=0,
        )

    def predict(self, segmentations, joints_2d):

        data = {}

        for i in range(self.num):
            data['image_{}'.format(i)] = np.float32(segmentations[i].reshape((1, self.img_size, self.img_size, -1)))
            data['J_2d_{}'.format(i)] = np.float32(np.expand_dims(joints_2d[i], 0))

        pred = self.inference_model.predict(data)

        res = {
            'vertices_tposed': pred[0][0],
            'vertices': np.array([p[0] for p in pred[1:self.num + 1]]),
            'faces': self.faces,
            'betas': pred[self.num + 1][0],
            'offsets': pred[self.num + 2][0],
            'poses': np.array(
                [cv2.Rodrigues(p0)[0] for p in pred[self.num + 3:2 * self.num + 3] for p0 in p[0]]
            ).reshape((self.num, -1)),
            'trans': np.array([t[0] for t in pred[2 * self.num + 3:]]),
        }

        return res

def main(weights, name, in_dir, segm_dir, pose_dir, out_dir, opt_pose_steps, opt_shape_steps, openpose_model_dir, frame_data_name):
    segm_files = sorted(glob(os.path.join(segm_dir, '*.png')) + glob(os.path.join(segm_dir, '*.jpg')))

    if pose_dir is None and openpose_model_dir is None:
        exit('No pose information available.')

    joints_2d, face_2d = [], []

    joints_2d, face_2d = openpose_process(in_dir, openpose_model_dir)

    K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))))

    model = Octopus(num=len(segm_files))
    model.load(weights)

    segmentations = [read_segmentation(f) for f in segm_files]

    if opt_pose_steps:
        #print('Optimizing for pose...')
        model.opt_pose(segmentations, joints_2d, opt_steps=opt_pose_steps)

    if opt_shape_steps:
        #print('Optimizing for shape...')
        model.opt_shape(segmentations, joints_2d, face_2d, opt_steps=opt_shape_steps)

    #print('Estimating shape...')
    pred = model.predict(segmentations, joints_2d)

    # Include texture coords in mesh
    vt = np.load('assets/basicModel_vt.npy')
    ft = np.load('assets/basicModel_ft.npy')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Output frame data if specified
    write_frame_data('{}/{}.pkl'.format(out_dir, frame_data_name), pred['vertices'])


if __name__ == '__main__':
    main('assets/octopus_weights.hdf5', 'sample', 'data/sample/frames', 'data/sample/segmentations', None, 'output', 5, 15, '/root/openpose/models', 'frame_data')
