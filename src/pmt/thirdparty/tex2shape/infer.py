from __future__ import print_function

import cv2
import sys
import os
import argparse
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm 

import chumpy as ch
from chumpy.ch import Ch
from chumpy.utils import col

from glob import glob

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten
from keras.layers import UpSampling2D, Concatenate, BatchNormalization
from keras.layers import LeakyReLU, MaxPool2D, Dropout

if sys.version_info[0] == 3:
    import _pickle as pkl
else:
    import cPickle as pkl

def lrotmin(p): 
    if isinstance(p, np.ndarray):
        p = p.ravel()[3:]
        return np.concatenate([(cv2.Rodrigues(np.array(pp))[0]-np.eye(3)).ravel() for pp in p.reshape((-1,3))]).ravel()        
    if p.ndim != 2 or p.shape[1] != 3:
        p = p.reshape((-1,3))
    p = p[1:]
    return ch.concatenate([(Rodrigues(pp)-ch.eye(3)).ravel() for pp in p]).ravel()

def posemap(s):
    if s == 'lrotmin':
        return lrotmin
    else:
        raise Exception('Unknown posemapping: %s' % (str(s),))

def backwards_compatibility_replacements(dd):

    # replacements
    if 'default_v' in dd:
        dd['v_template'] = dd['default_v']
        del dd['default_v']
    if 'template_v' in dd:
        dd['v_template'] = dd['template_v']
        del dd['template_v']
    if 'joint_regressor' in dd:
        dd['J_regressor'] = dd['joint_regressor']
        del dd['joint_regressor']
    if 'blendshapes' in dd:
        dd['posedirs'] = dd['blendshapes']
        del dd['blendshapes']
    if 'J' not in dd:
        dd['J'] = dd['joints']
        del dd['joints']

    # defaults
    if 'bs_style' not in dd:
        dd['bs_style'] = 'lbs'

def normalize(map):
    norm = np.linalg.norm(map, axis=-1)
    norm[norm == 0] = 1.

    return map / np.expand_dims(norm, -1)

def map_densepose_to_tex(img, iuv_img, tex_res):
    if map_densepose_to_tex.lut is None:
        map_densepose_to_tex.lut = np.load(os.path.join(os.path.dirname(__file__), './assets/dp_uv_lookup_256.npy'))

    iuv_raw = iuv_img[iuv_img[:, :, 0] > 0]
    data = img[iuv_img[:, :, 0] > 0]
    i = iuv_raw[:, 0] - 1

    if iuv_raw.dtype == np.uint8:
        u = iuv_raw[:, 1] / 255.
        v = iuv_raw[:, 2] / 255.
    else:
        u = iuv_raw[:, 1]
        v = iuv_raw[:, 2]

    u[u > 1] = 1.
    v[v > 1] = 1.

    uv_smpl = map_densepose_to_tex.lut[
        i.astype(np.int),
        np.round(v * 255.).astype(np.int),
        np.round(u * 255.).astype(np.int)
    ]

    tex = np.ones((tex_res, tex_res, img.shape[2])) * 0.5

    u_I = np.round(uv_smpl[:, 0] * (tex.shape[1] - 1)).astype(np.int32)
    v_I = np.round((1 - uv_smpl[:, 1]) * (tex.shape[0] - 1)).astype(np.int32)

    tex[v_I, u_I] = data

    return tex

def mesh_write(filename, v, f, vt=None, ft=None, vn=None, vc=None, texture=None):

    if vc is not None:
        filename += (('v {:f} {:f} {:f} {:f} {:f} {:f}\n' * len(v)).format(*np.hstack((v, vc)).reshape(-1)))
    else:
        filename += (('v {:f} {:f} {:f}\n' * len(v)).format(*v.reshape(-1)))

    if vn is not None:
        filename += (('vn {:f} {:f} {:f}\n' * len(vn)).format(*vn.reshape(-1)))

    if vt is not None:
        filename += (('vt {:f} {:f}\n' * len(vt)).format(*vt.reshape(-1)))

    if ft is not None:
        filename += (('f {:d}/{:d}/{:d} {:d}/{:d}/{:d} {:d}/{:d}/{:d}\n' * len(f)).format(*np.hstack((f.reshape(-1, 1), ft.reshape(-1, 1), f.reshape(-1, 1))).reshape(-1) + 1))
    else:
        filename += (('f {:d}//{:d} {:d}//{:d} {:d}//{:d}\n' * len(f)).format(*np.repeat(f.reshape(-1) + 1, 2)))
    
    return filename

class BaseModel(object):
    def __init__(self):
        self.model = None
        self.inputs = []
        self.outputs = []

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        #print("Saving model ({})...".format(checkpoint_path))
        self.model.save_weights(checkpoint_path)
        #print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        #print("Loading model checkpoint {} ...".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        #print("Model loaded")

    def summary(self):
        if self.model is None:
            raise Exception("You have to build the model first.")

        return self.model.summary()

    def predict(self, x):
        if self.model is None:
            raise Exception("You have to build the model first.")

        return self.model.predict(x)

    def __call__(self, inputs, **kwargs):
        if self.model is None:
            raise Exception("You have to build the model first.")

        return self.model(inputs, **kwargs)

    def build_model(self):
        raise NotImplementedError

class Rodrigues(ch.Ch):
    dterms = 'rt'
    
    def compute_r(self):
        return cv2.Rodrigues(self.rt.r)[0]
    
    def compute_dr_wrt(self, wrt):
        if wrt is self.rt:
            return cv2.Rodrigues(self.rt.r)[1].T

class sp_dot(ch.Ch):
    terms = 'a',
    dterms = 'b',

    def on_changed(self, which):
        if 'a' in which:
            a_csr = sp.csr_matrix(self.a)
            # To stay consistent with numpy, we must upgrade 1D arrays to 2D
            self.ar = sp.csr_matrix((a_csr.data, a_csr.indices, a_csr.indptr),
                                    shape=(max(np.sum(a_csr.shape[:-1]), 1), a_csr.shape[-1]))

        if 'b' in which:
            self.br = col(self.b.r) if len(self.b.r.shape) < 2 else self.b.r.reshape((self.b.r.shape[0], -1))

        if 'a' in which or 'b' in which:
            self.k = sp.kron(self.ar, sp.eye(self.br.shape[1], self.br.shape[1]))

    def compute_r(self):
        return self.a.dot(self.b.r)

    def compute(self):
        if self.br.ndim <= 1:
            return self.ar
        elif self.br.ndim <= 2:
            return self.k
        else:
            raise NotImplementedError

    def compute_dr_wrt(self, wrt):
        if wrt is self.b:
            return self.compute()

class Smpl(Ch):
    """
    Class to store SMPL object with slightly improved code and access to more matrices
    """
    terms = 'model',
    dterms = 'trans', 'betas', 'pose', 'v_personal'

    def __init__(self, *args, **kwargs):
        self.on_changed(self._dirty_vars)

    def on_changed(self, which):
        if 'model' in which:
            if not isinstance(self.model, dict):
                dd = pkl.load(open(self.model))
            else:
                dd = self.model

            backwards_compatibility_replacements(dd)

            # for s in ['v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs', 'betas', 'J']:
            for s in ['posedirs', 'shapedirs']:
                if (s in dd) and not hasattr(dd[s], 'dterms'):
                    dd[s] = ch.array(dd[s])

            self.f = dd['f']
            self.v_template = dd['v_template']
            if not hasattr(self, 'v_personal'):
                self.v_personal = ch.zeros_like(self.v_template)
            self.shapedirs = dd['shapedirs']
            self.J_regressor = dd['J_regressor']
            if 'J_regressor_prior' in dd:
                self.J_regressor_prior = dd['J_regressor_prior']
            self.bs_type = dd['bs_type']
            self.weights = dd['weights']
            if 'vert_sym_idxs' in dd:
                self.vert_sym_idxs = dd['vert_sym_idxs']
            if 'weights_prior' in dd:
                self.weights_prior = dd['weights_prior']
            self.kintree_table = dd['kintree_table']
            self.posedirs = dd['posedirs']

            if not hasattr(self, 'betas'):
                self.betas = ch.zeros(self.shapedirs.shape[-1])

            if not hasattr(self, 'trans'):
                self.trans = ch.zeros(3)

            if not hasattr(self, 'pose'):
                self.pose = ch.zeros(72)

            self._set_up()

    def _set_up(self):
        self.v_shaped = self.shapedirs.dot(self.betas) + self.v_template

        body_height = (self.v_shaped[2802, 1] + self.v_shaped[6262, 1]) - (
                self.v_shaped[2237, 1] + self.v_shaped[6728, 1])
        self.scale = 1.66 / float(body_height)

        self.v_shaped_personal = self.scale * self.v_shaped + self.v_personal

        if sp.issparse(self.J_regressor):
            self.J = self.scale * sp_dot(self.J_regressor, self.v_shaped)
        else:
            self.J = self.scale * ch.sum(self.J_regressor.T.reshape(-1, 1, 24) * self.v_shaped.reshape(-1, 3, 1), axis=0).T
        self.v_posevariation = self.posedirs.dot(posemap(self.bs_type)(self.pose))
        self.v_poseshaped = self.v_shaped_personal + self.v_posevariation

        self.A, A_global = self._global_rigid_transformation()
        self.Jtr = ch.vstack([g[:3, 3] for g in A_global])
        self.J_transformed = self.Jtr + self.trans.reshape((1, 3))

        self.V = self.A.dot(self.weights.T)

        rest_shape_h = ch.hstack((self.v_poseshaped, ch.ones((self.v_poseshaped.shape[0], 1))))
        self.v_posed = ch.sum(self.V.T * rest_shape_h.reshape(-1, 4, 1), axis=1)[:, :3]
        self.v = self.v_posed + self.trans

    def _global_rigid_transformation(self):
        results = {}
        pose = self.pose.reshape((-1, 3))
        parent = {i: self.kintree_table[0, i] for i in range(1, self.kintree_table.shape[1])}

        with_zeros = lambda x: ch.vstack((x, ch.array([[0.0, 0.0, 0.0, 1.0]])))
        pack = lambda x: ch.hstack([ch.zeros((4, 3)), x.reshape((4, 1))])

        results[0] = with_zeros(ch.hstack((Rodrigues(pose[0, :]), self.J[0, :].reshape((3, 1)))))

        for i in range(1, self.kintree_table.shape[1]):
            results[i] = results[parent[i]].dot(with_zeros(ch.hstack((
                Rodrigues(pose[i, :]),      # rotation around bone endpoint
                (self.J[i, :] - self.J[parent[i], :]).reshape((3, 1))     # bone
            ))))

        results = [results[i] for i in sorted(results.keys())]
        results_global = results

        # subtract rotated J position
        results2 = [results[i] - (pack(
            results[i].dot(ch.concatenate((self.J[i, :], [0]))))
        ) for i in range(len(results))]
        result = ch.dstack(results2)

        return result, results_global

    def compute_r(self):
        return self.v.r

    def compute_dr_wrt(self, wrt):
        if wrt is not self.trans and wrt is not self.betas and wrt is not self.pose and wrt is not self.v_personal:
            return None

        return self.v.dr_wrt(wrt)

class MeshFromMaps:

    def __init__(self):

        with open(os.path.join(os.path.dirname(__file__), './assets/hres.pkl'), 'rb') as f:
            self.hres = pkl.load(f, encoding='latin1')

        with open(os.path.join(os.path.dirname(__file__), './assets/neutral_smpl.pkl'), 'rb') as f:
            model_data = pkl.load(f, encoding='latin1')

        model_data = self._make_hres(model_data)

        self.vt = self.hres['vt']
        self.ft = self.hres['ft']
        self.smpl = Smpl(model_data)

        self._prepare()

    def _make_hres(self, dd):
        hv = self.hres['v']
        hf = self.hres['f']
        mapping = self.hres['mapping']
        num_betas = dd['shapedirs'].shape[-1]
        J_reg = dd['J_regressor'].asformat('csr')

        model = {
            'v_template': hv,
            'weights': np.hstack([
                np.expand_dims(
                    np.mean(
                        mapping.dot(np.repeat(np.expand_dims(dd['weights'][:, i], -1), 3)).reshape(-1, 3)
                        , axis=1),
                    axis=-1)
                for i in range(24)
            ]),
            'posedirs': mapping.dot(dd['posedirs'].reshape((-1, 207))).reshape(-1, 3, 207),
            'shapedirs': mapping.dot(dd['shapedirs'].reshape((-1, num_betas))).reshape(-1, 3, num_betas),
            'J_regressor': sp.csr_matrix((J_reg.data, J_reg.indices, J_reg.indptr), shape=(24, hv.shape[0])),
            'kintree_table': dd['kintree_table'],
            'bs_type': dd['bs_type'],
            'J': dd['J'],
            'f': hf,
        }

        return model

    def _prepare(self):
        vt_per_v = {i: [] for i in range(len(self.smpl))}

        for ff, fft in zip(self.smpl.f, self.ft):
            for vt0, v0 in zip(fft, ff):
                vt_per_v[v0].append(vt0)

        self.single_vt_keys = []
        self.single_vt_val = []
        self.dual_vt_keys = []
        self.dual_vt_val = []
        self.triple_vt_keys = []
        self.triple_vt_val = []
        self.quadruple_vt_keys = []
        self.quadruple_vt_val = []

        self.multi_vt = {}

        for v in vt_per_v.keys():
            vt_list = np.unique(vt_per_v[v])

            if len(vt_list) == 1:
                self.single_vt_keys.append(v)
                self.single_vt_val.append(vt_list[0])
            elif len(vt_list) == 2:
                self.dual_vt_keys.append(v)
                self.dual_vt_val.append(vt_list)
            elif len(vt_list) == 3:
                self.triple_vt_keys.append(v)
                self.triple_vt_val.append(vt_list)
            elif len(vt_list) == 4:
                self.quadruple_vt_keys.append(v)
                self.quadruple_vt_val.append(vt_list)
            else:
                self.multi_vt[v] = vt_list

    def _lookup(self, uv, map):
        ui = np.round(uv[:, 0] * (map.shape[1] - 1)).astype(np.int32)
        vi = np.round((1 - uv[:, 1]) * (map.shape[0] - 1)).astype(np.int32)

        return map[vi, ui]

    def get_mesh(self, n_map, d_map, betas=None, pose=None):
        n_map = normalize(n_map)

        normals = np.zeros_like(self.smpl)
        displacements = np.zeros_like(self.smpl)

        normals[self.single_vt_keys] = self._lookup(self.vt[self.single_vt_val], n_map)
        normals[self.dual_vt_keys] = np.mean(
            self._lookup(self.vt[np.array(self.dual_vt_val).ravel()], n_map).reshape((-1, 2, 3)),
            axis=1)
        normals[self.triple_vt_keys] = np.mean(
            self._lookup(self.vt[np.array(self.triple_vt_val).ravel()], n_map).reshape((-1, 3, 3)),
            axis=1)
        normals[self.quadruple_vt_keys] = np.mean(
            self._lookup(self.vt[np.array(self.quadruple_vt_val).ravel()], n_map).reshape((-1, 4, 3)),
            axis=1)

        for v in self.multi_vt:
            normals[v] = np.mean(self._lookup(self.vt[self.multi_vt[v]], n_map), axis=0)

        displacements[self.single_vt_keys] = self._lookup(self.vt[self.single_vt_val], d_map)
        displacements[self.dual_vt_keys] = np.mean(
            self._lookup(self.vt[np.array(self.dual_vt_val).ravel()], d_map).reshape((-1, 2, 3)),
            axis=1)
        displacements[self.triple_vt_keys] = np.mean(
            self._lookup(self.vt[np.array(self.triple_vt_val).ravel()], d_map).reshape((-1, 3, 3)),
            axis=1)
        displacements[self.quadruple_vt_keys] = np.mean(
            self._lookup(self.vt[np.array(self.quadruple_vt_val).ravel()], d_map).reshape((-1, 4, 3)), axis=1)

        for v in self.multi_vt:
            displacements[v] = np.mean(self._lookup(self.vt[self.multi_vt[v]], d_map), axis=0)

        if betas is not None:
            self.smpl.betas[:] = betas
        else:
            self.smpl.betas[:] = 0

        if pose is not None:
            self.smpl.pose[:] = pose
            normals_H = np.hstack((normals, np.zeros((normals.shape[0], 1))))
            normals = np.sum(self.smpl.V.T.r * normals_H.reshape((-1, 4, 1)), axis=1)[:, :3]
        else:
            self.smpl.pose[:] = 0

        self.smpl.v_personal[:] = displacements

        return {
            'v': self.smpl.r,
            'f': self.smpl.f,
            'vn': normals,
            'vt': self.vt,
            'ft': self.ft,
        }

class BetasModel(BaseModel):
    def __init__(self, input_shape=(1024, 1024, 3), output_dims=10,
                 kernel_size=3, bn=True):
        super(BetasModel, self).__init__()
        self.input_shape = input_shape
        self.output_dims = input_shape[2] if output_dims is None else output_dims
        self.kernel_size = (kernel_size, kernel_size)
        self.bn = bn
        self.build_model()

    def build_model(self):
        x = Input(shape=self.input_shape, name='image')

        self.inputs.append(x)

        filters = 8

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = MaxPool2D()(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Downsampling
        d1 = conv2d(x, filters, bn=False)
        d2 = conv2d(d1, filters * 2)
        d3 = conv2d(d2, filters * 4)
        d4 = conv2d(d3, filters * 8)
        d5 = conv2d(d4, filters * 8)
        d6 = conv2d(d5, filters * 8)
        d7 = conv2d(d6, filters * 8)

        x = Flatten()(d7)
        x = Dense(self.output_dims, name='betas')(x)

        self.outputs.append(x)

        self.model = Model(inputs=self.inputs, outputs=self.outputs)

class Tex2ShapeModel(BaseModel):
    def __init__(self, input_shape=(512, 512, 3), output_dims=6,
                 kernel_size=3, dropout_rate=0, bn=True, final_layer=None):
        super(Tex2ShapeModel, self).__init__()
        self.input_shape = input_shape
        self.output_dims = input_shape[2] if output_dims is None else output_dims
        self.kernel_size = (kernel_size, kernel_size)
        self.dropout_rate = dropout_rate
        self.bn = bn
        self.final_layer = final_layer
        self.build_model()

    def build_model(self):
        x = Input(shape=self.input_shape, name='image')

        self.inputs.append(x)

        x = self._unet_core(x)
        x = Conv2D(self.output_dims, self.kernel_size, padding='same', name='output')(x)

        if self.final_layer:
            x = self.final_layer(x)

        self.outputs.append(x)

        self.model = Model(inputs=self.inputs, outputs=self.outputs)

    def _unet_core(self, d0):

        filters = 64

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if self.dropout_rate:
                u = Dropout(self.dropout_rate)(u)
            if self.bn:
                u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Downsampling
        d1 = conv2d(d0, filters, bn=False)
        d2 = conv2d(d1, filters * 2)
        d3 = conv2d(d2, filters * 4)
        d4 = conv2d(d3, filters * 8)
        d5 = conv2d(d4, filters * 8)
        d6 = conv2d(d5, filters * 8)
        d7 = conv2d(d6, filters * 8)

        # Upsampling
        u1 = deconv2d(d7, d6, filters * 8)
        u2 = deconv2d(u1, d5, filters * 8)
        u3 = deconv2d(u2, d4, filters * 8)
        u4 = deconv2d(u3, d3, filters * 4)
        u5 = deconv2d(u4, d2, filters * 2)
        u6 = deconv2d(u5, d1, filters)

        u7 = UpSampling2D(size=2)(u6)

        return u7

map_densepose_to_tex.lut = None

def main(img_files, iuv_files, out_dir, weights_tex2shape, weights_betas):

    if os.path.isfile(img_files) != os.path.isfile(iuv_files):
        #print('Inconsistent input.')
        exit(1)

    tex2shape_model = Tex2ShapeModel()
    betas_model = BetasModel()

    tex2shape_model.load(weights_tex2shape)
    betas_model.load(weights_betas)

    mfm = MeshFromMaps()

    if os.path.isfile(img_files):
        img_files = [img_files]
        iuv_files = [iuv_files]
    else:
        img_files = sorted(glob(os.path.join(img_files, '*.png')) + glob(os.path.join(img_files, '*.jpg')))
        iuv_files = sorted(glob(os.path.join(iuv_files, '*.png')))

    for img_file, iuv_file in zip(img_files, iuv_files):

        img = cv2.imread(img_file) / 255.
        iuv_img = cv2.imread(iuv_file)
        unwrap = np.expand_dims(map_densepose_to_tex(img, iuv_img, 512), 0)

        name = os.path.splitext(os.path.basename(img_file))[0]

        #print('Processing {}...'.format(name))

        iuv_img = iuv_img * 1.
        iuv_img[:, :, 1:] /= 255.
        iuv_img = np.expand_dims(iuv_img, 0)

        #print('> Estimating normal and displacement maps...')
        pred = tex2shape_model.predict(unwrap)

        #print('> Estimating betas...')
        betas = betas_model.predict(iuv_img)

        if not os.path.exists('./output'):
            os.makedirs('output')

        #print('> Saving maps and betas...')
        #pkl.dump({
        #    'normal_map': normalize(pred[0, :, :, :3]),
        #    'displacement_map': pred[0, :, :, 3:] / 10.,
        #    'betas': betas[0],
        #}, open('{}/{}.pkl'.format(out_dir, name), 'wb'), protocol=2)

        #print('> Baking obj file for easy inspection...')
        m = mfm.get_mesh(pred[0, :, :, :3], pred[0, :, :, 3:] / 10, betas=betas[0])
        mesh_write('{}/{}.obj'.format(out_dir, name), v=m['v'], f=m['f'], vn=m['vn'], vt=m['vt'], ft=m['ft'])

        #print('Done.')


if __name__ == '__main__':
    main('data/images', 'data/densepose', 'output', 'assets/tex2shape_weights.hdf5', 'assets/betas_weights.hdf5')
