import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import os
import gco
import argparse
import sys
import platform
import numpy as np
import pickle as pkl

from chumpy import depends_on, Ch
from chumpy.utils import col

from skimage.measure import compare_ssim
from skimage import color

from tqdm import tqdm, trange
from glob import glob
from scipy import signal
from sklearn.mixture import GaussianMixture
from chumpy.utils import row, col

from opendr.camera import ProjectPoints
from opendr.renderer import DepthRenderer
from opendr.camera import ProjectPoints3D
from opendr.geometry import VertNormals
from opendr.geometry import Rodrigues
from opendr.renderer import BaseRenderer, ColoredRenderer, TexturedRenderer
from opendr.geometry import TriNormals
from opendr.renderer import draw_edge_visibility, draw_boundary_images, draw_boundaryid_image
from opendr.contexts._constants import *

if platform.system()=='Darwin':
    from .assets.ctx_mac import OsContext 
else:
    from .assets.ctx_mesa import OsContext 

#sys.path.append('path/to/opendr/opendr') # if dir above does not work with your OS

def compute_vpe_boundary_idxs(v, f, camera, fpe):
    # Figure out which edges are on pairs of differently visible triangles

    tn = TriNormals(v, f).r.reshape((-1,3))

    #ray = cv2.Rodrigues(camera.rt.r)[0].T[:,2]
    campos = -cv2.Rodrigues(camera.rt.r)[0].T.dot(camera.t.r)
    rays_to_verts = v.reshape((-1,3)) - row(campos)
    rays_to_faces = rays_to_verts[f[:,0]] + rays_to_verts[f[:,1]] + rays_to_verts[f[:,2]]
    faces_invisible = np.sum(rays_to_faces * tn, axis=1)
    dps = faces_invisible[fpe[:,0]] * faces_invisible[fpe[:,1]]
    silhouette_edges = np.asarray(np.nonzero(dps<=0)[0], np.uint32)
    return silhouette_edges, faces_invisible < 0

def draw_boundaryid_image(gl, v, f, vpe, fpe, camera):


    if False:
        visibility = draw_edge_visibility(gl, v, vpe, f, hidden_wireframe=True)
        return visibility
        
    if True:
    #try:
        gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        silhouette_edges, faces_facing_camera = compute_vpe_boundary_idxs(v, f, camera, fpe)
        lines_e = vpe[silhouette_edges]
        lines_v = v

        if len(lines_e)==0:
            return np.ones((gl.height, gl.width)).astype(np.int32) * 4294967295
        visibility = draw_edge_visibility(gl, lines_v, lines_e, f, hidden_wireframe=True)
        shape = visibility.shape
        visibility = visibility.ravel()
        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        visibility[visible] = silhouette_edges[visibility[visible]]
        result = visibility.reshape(shape)
        return result

        #return np.asarray(deepcopy(gl.getImage()), np.float64)
    #except:
    #    import pdb; pdb.set_trace()

def _setup_ortho(gl, l, r, b, t, near, far, view_matrix):
    gl.MatrixMode(GL_PROJECTION)
    gl.LoadIdentity()
    gl.Ortho(l, r, t, b, near, far)  # top and bottom switched for opencv coordinate system

    gl.MatrixMode(GL_MODELVIEW)
    gl.LoadIdentity()
    gl.Rotatef(180, 1, 0, 0)

    view_mtx = np.asarray(np.vstack((view_matrix, np.array([0, 0, 0, 1]))), np.float32, order='F')
    gl.MultMatrixf(view_mtx)

    gl.Enable(GL_DEPTH_TEST)
    gl.PolygonMode(GL_BACK, GL_FILL)
    gl.Disable(GL_LIGHTING)
    gl.Disable(GL_CULL_FACE)
    gl.PixelStorei(GL_PACK_ALIGNMENT, 1)
    gl.PixelStorei(GL_UNPACK_ALIGNMENT, 1)

    gl.UseProgram(0)

def read_segmentation(file_or_val):
    if isinstance(file_or_val, str):
        segm = cv2.resize(file_or_val, (1080, 1080))[:, :, ::-1]
    else:
        segm = file_or_val[:, :, :]

    segm[np.all(segm == LABELS_FULL['Sunglasses'], axis=2)] = LABELS_REDUCED['Face']
    segm[np.all(segm == LABELS_FULL['LeftArm'], axis=2)] = LABELS_REDUCED['Arms']
    segm[np.all(segm == LABELS_FULL['RightArm'], axis=2)] = LABELS_REDUCED['Arms']
    segm[np.all(segm == LABELS_FULL['LeftLeg'], axis=2)] = LABELS_REDUCED['Legs']
    segm[np.all(segm == LABELS_FULL['RightLeg'], axis=2)] = LABELS_REDUCED['Legs']
    segm[np.all(segm == LABELS_FULL['LeftShoe'], axis=2)] = LABELS_REDUCED['Shoes']
    segm[np.all(segm == LABELS_FULL['RightShoe'], axis=2)] = LABELS_REDUCED['Shoes']

    return segm


    terms = ['f', 'overdraw']
    dterms = ['ortho', 'v']

    @property
    def v(self):
        return self.ortho.v

    @v.setter
    def v(self, newval):
        self.ortho.v = newval

    @depends_on('f', 'ortho', 'overdraw')
    def barycentric_image(self):
        return super(OrthoBaseRenderer, self).barycentric_image

    @depends_on(terms+dterms)
    def boundaryid_image(self):
        self._call_on_changed()
        return draw_boundaryid_image(self.glb, self.v.r, self.f, self.vpe, self.fpe, self.ortho)

    @depends_on('f', 'ortho', 'overdraw')
    def visibility_image(self):
        return super(OrthoBaseRenderer, self).visibility_image

    @depends_on('f', 'ortho')
    def edge_visibility_image(self):
        self._call_on_changed()
        return draw_edge_visibility(self.glb, self.v.r, self.vpe, self.f)

def edges_seams(seams, tex_res, edge_idx):
    edges = np.zeros((0, 2), dtype=np.int32)

    for _, e0, _, e1 in seams:
        idx0 = np.array(edge_idx[e0][0]) * tex_res + np.array(edge_idx[e0][1])
        idx1 = np.array(edge_idx[e1][0]) * tex_res + np.array(edge_idx[e1][1])

        if len(idx0) and len(idx1):
            if idx0.shape[0] < idx1.shape[0]:
                idx0 = cv2.resize(idx0.reshape(-1, 1), (1, idx1.shape[0]), interpolation=cv2.INTER_NEAREST)
            elif idx0.shape[0] > idx1.shape[0]:
                idx1 = cv2.resize(idx1.reshape(-1, 1), (1, idx0.shape[0]), interpolation=cv2.INTER_NEAREST)

            edges_new = np.hstack((idx0.reshape(-1, 1), idx1.reshape(-1, 1)))
            edges = np.vstack((edges, edges_new))

    edges = np.sort(edges, axis=1)

    return edges[:, 0], edges[:, 1]
    
def to_ids(segm):
    ids = np.zeros(segm.shape[:2], dtype=np.uint8)
    i = 0

    for id in LABELS_REDUCED:
        ids[np.all(segm == LABELS_REDUCED[id], axis=2)] = i
        i += 1

    return ids
    
class Stitcher:

    def __init__(self, seams, tex_res, mask, edge_idx_file='thirdparty/pic2tex/assets/basicModel_edge_idx_1000_.pkl'):
        self.tex_res = tex_res
        self.seams = seams
        self.edge_idx = pkl.load(open(edge_idx_file, 'rb'))

        dr_v = signal.convolve2d(mask, [[-1, 1]])[:, 1:]
        dr_h = signal.convolve2d(mask, [[-1], [1]])[1:, :]

        self.where_v = mask - dr_v
        self.where_h = mask - dr_h

        idxs = np.arange(tex_res ** 2).reshape(tex_res, tex_res)
        v_edges_from = idxs[:-1, :][self.where_v[:-1, :] == 1].flatten()
        v_edges_to = idxs[1:, :][self.where_v[:-1, :] == 1].flatten()
        h_edges_from = idxs[:, :-1][self.where_h[:, :-1] == 1].flatten()
        h_edges_to = idxs[:, 1:][self.where_h[:, :-1] == 1].flatten()

        self.s_edges_from, self.s_edges_to = self._edges_seams()

        self.edges_from = np.r_[v_edges_from, h_edges_from, self.s_edges_from]
        self.edges_to = np.r_[v_edges_to, h_edges_to, self.s_edges_to]

    def stich(self, im0, im1, unaries0, unaries1, labels0, labels1, pairwise_mask, segmentation):

        gc = gco.GCO()
        gc.create_general_graph(self.tex_res ** 2, 2, True)
        gc.set_data_cost(np.dstack((unaries0, unaries1)).reshape(-1, 2))

        edges_w = self._rgb_grad(im0, im1, labels0, labels1, pairwise_mask, segmentation)

        gc.set_all_neighbors(self.edges_from, self.edges_to, edges_w)
        gc.set_smooth_cost((1 - np.eye(2)) * 65)
        gc.swap()

        labels = gc.get_labels()
        gc.destroy_graph()

        labels = labels.reshape(self.tex_res, self.tex_res).astype(np.float32)
        label_maps = np.zeros((2, self.tex_res, self.tex_res))

        for l in range(2):
            label_maps[l] = cv2.blur(np.float32(labels == l), (self.tex_res // 100, self.tex_res // 100))  # TODO

        norm_masks = np.sum(label_maps, axis=0)
        result = (np.atleast_3d(label_maps[0]) * im0 + np.atleast_3d(label_maps[1]) * im1)
        result[norm_masks != 0] /= np.atleast_3d(norm_masks)[norm_masks != 0]

        return result, labels

    def _edges_seams(self):
        edges = np.zeros((0, 2), dtype=np.int32)

        for _, e0, _, e1 in self.seams:
            idx0 = np.array(self.edge_idx[e0][0]) * self.tex_res + np.array(self.edge_idx[e0][1])
            idx1 = np.array(self.edge_idx[e1][0]) * self.tex_res + np.array(self.edge_idx[e1][1])

            if len(idx0) and len(idx1):
                if idx0.shape[0] < idx1.shape[0]:
                    idx0 = cv2.resize(idx0.reshape(-1, 1), (1, idx1.shape[0]), interpolation=cv2.INTER_NEAREST)
                elif idx0.shape[0] > idx1.shape[0]:
                    idx1 = cv2.resize(idx1.reshape(-1, 1), (1, idx0.shape[0]), interpolation=cv2.INTER_NEAREST)

                edges_new = np.hstack((idx0.reshape(-1, 1), idx1.reshape(-1, 1)))
                edges = np.vstack((edges, edges_new))

        edges = np.sort(edges, axis=1)

        return edges[:, 0], edges[:, 1]

    def _rgb_grad(self, im0, im1, labels0, labels1, pairwise_mask, segmentation):
        gray0 = color.rgb2gray(im0) * pairwise_mask
        gray1 = color.rgb2gray(im1) * pairwise_mask

        grad0 = np.abs(gray0.flatten()[self.edges_from] - gray1.flatten()[self.edges_to])
        grad1 = np.abs(gray1.flatten()[self.edges_from] - gray0.flatten()[self.edges_to])

        label_grad = np.logical_not(np.equal(labels0.flatten()[self.edges_from], labels1.flatten()[self.edges_to]))
        if segmentation is not None:
            seg_grad = np.equal(segmentation.flatten()[self.edges_from], segmentation.flatten()[self.edges_to])
        else:
            seg_grad = 1.

        return np.maximum(grad0, grad1) * np.float32(label_grad) * np.float32(seg_grad)

class VisibilityRenderer:
    def __init__(self, vt, ft, tex_res, f):
        ortho = OrthoProjectPoints(rt=np.zeros(3), t=np.zeros(3), near=-1, far=1, left=-0.5, right=0.5, bottom=-0.5,
                                   top=0.5, width=tex_res, height=tex_res)
        vt3d = np.dstack((vt[:, 0] - 0.5, 1 - vt[:, 1] - 0.5, np.zeros(vt.shape[0])))[0]
        vt3d = vt3d[ft].reshape(-1, 3)
        self.f = f
        self.rn = OrthoColoredRenderer(bgcolor=np.zeros(3), ortho=ortho, v=vt3d, f=np.arange(ft.size).reshape(-1, 3),
                                       num_channels=1)

    def render(self, vertex_visibility):
        vc = vertex_visibility.reshape(-1, 1)
        vc = np.hstack((vc, vc, vc))
        self.rn.set(vc=vc[self.f].reshape(-1, 3))

        return np.array(self.rn.r)

    def mask(self):
        self.rn.set(vc=np.ones_like(self.rn.v))
        return np.array(self.rn.r)

class OrthoBaseRenderer(BaseRenderer):
    terms = ['f', 'overdraw']
    dterms = ['ortho', 'v']

    @property
    def v(self):
        return self.ortho.v

    @v.setter
    def v(self, newval):
        self.ortho.v = newval

    @depends_on('f', 'ortho', 'overdraw')
    def barycentric_image(self):
        return super(OrthoBaseRenderer, self).barycentric_image

    @depends_on(terms+dterms)
    def boundaryid_image(self):
        self._call_on_changed()
        return draw_boundaryid_image(self.glb, self.v.r, self.f, self.vpe, self.fpe, self.ortho)

    @depends_on('f', 'ortho', 'overdraw')
    def visibility_image(self):
        return super(OrthoBaseRenderer, self).visibility_image

    @depends_on('f', 'ortho')
    def edge_visibility_image(self):
        self._call_on_changed()
        return draw_edge_visibility(self.glb, self.v.r, self.vpe, self.f)

class OrthoColoredRenderer(OrthoBaseRenderer, ColoredRenderer):
    terms = 'f', 'background_image', 'overdraw', 'num_channels'
    dterms = 'vc', 'ortho', 'bgcolor'

    def compute_r(self):
        return self.color_image

    def compute_dr_wrt(self, wrt):
        raise NotImplementedError

    def on_changed(self, which):
        if 'ortho' in which:
            w = self.ortho.width
            h = self.ortho.height
            self.glf = OsContext(np.int(w), np.int(h), typ=GL_FLOAT)
            _setup_ortho(self.glf, self.ortho.left.r, self.ortho.right.r, self.ortho.bottom.r, self.ortho.top.r,
                         self.ortho.near, self.ortho.far, self.ortho.view_mtx)
            self.glf.Viewport(0, 0, w, h)
            self.glb = OsContext(np.int(w), np.int(h), typ=GL_UNSIGNED_BYTE)
            self.glb.Viewport(0, 0, w, h)
            _setup_ortho(self.glb, self.ortho.left.r, self.ortho.right.r, self.ortho.bottom.r, self.ortho.top.r,
                         self.ortho.near, self.ortho.far, self.ortho.view_mtx)

        if not hasattr(self, 'num_channels'):
            self.num_channels = 3

        if not hasattr(self, 'bgcolor'):
            self.bgcolor = Ch(np.array([.5] * self.num_channels))
            which.add('bgcolor')

        if not hasattr(self, 'overdraw'):
            self.overdraw = True

        if 'bgcolor' in which:
            self.glf.ClearColor(self.bgcolor.r[0], self.bgcolor.r[1 % self.num_channels],
                                self.bgcolor.r[2 % self.num_channels], 1.)

    @depends_on('f', 'ortho', 'vc')
    def boundarycolor_image(self):
        return self.draw_boundarycolor_image(with_vertex_colors=True)

    @depends_on('f', 'ortho')
    def boundary_images(self):
        self._call_on_changed()
        return draw_boundary_images(self.glb, self.v.r, self.f, self.vpe, self.fpe, self.ortho)

    @depends_on(terms+dterms)
    def color_image(self):
        return super(OrthoColoredRenderer, self).color_image

    @property
    def shape(self):
        return (self.ortho.height, self.ortho.width, 3)

class OrthoTexturedRenderer(OrthoColoredRenderer, TexturedRenderer):
    terms = 'f', 'ft', 'background_image', 'overdraw'
    dterms = 'vc', 'ortho', 'bgcolor', 'texture_image', 'vt'

    def compute_dr_wrt(self, wrt):
        raise NotImplementedError

    def on_changed(self, which):
        OrthoColoredRenderer.on_changed(self, which)

        # have to redo if ortho changes, b/c ortho triggers new context
        if 'texture_image' in which or 'ortho' in which:
            gl = self.glf
            texture_data = np.array(self.texture_image * 255., dtype='uint8', order='C')
            tmp = np.zeros(1, dtype=np.uint32)

            self.release_textures()
            gl.GenTextures(1, tmp)

            self.textureID = tmp[0]
            gl.BindTexture(GL_TEXTURE_2D, self.textureID)

            gl.TexImage2Dub(GL_TEXTURE_2D, 0, GL_RGB, texture_data.shape[1], texture_data.shape[0], 0, GL_RGB,
                            texture_data.ravel())
            gl.GenerateMipmap(GL_TEXTURE_2D)

    def release_textures(self):
        if hasattr(self, 'textureID'):
            arr = np.asarray(np.array([self.textureID]), np.uint32)
            self.glf.DeleteTextures(arr)

    def texture_mapping_on(self, gl, with_vertex_colors):
        gl.Enable(GL_TEXTURE_2D)
        gl.BindTexture(GL_TEXTURE_2D, self.textureID)
        gl.TexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        gl.TexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        gl.TexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE if with_vertex_colors else GL_REPLACE)
        gl.EnableClientState(GL_TEXTURE_COORD_ARRAY)

    @depends_on(dterms+terms)
    def boundaryid_image(self):
        return super(OrthoTexturedRenderer, self).boundaryid_image

    @depends_on(terms+dterms)
    def color_image(self):
        self.glf.BindTexture(GL_TEXTURE_2D, self.textureID)
        return super(OrthoTexturedRenderer, self).color_image

    @depends_on(terms+dterms)
    def boundarycolor_image(self):
        self.glf.BindTexture(GL_TEXTURE_2D, self.textureID)
        return super(OrthoTexturedRenderer, self).boundarycolor_image

    @property
    def shape(self):
        return (self.ortho.height, self.ortho.width, 3)

    @depends_on('vt', 'ft')
    def mesh_tex_coords(self):
        ftidxs = self.ft.ravel()
        data = np.asarray(self.vt.r[ftidxs].astype(np.float32)[:, 0:2], np.float32, order='C')
        data[:, 1] = 1.0 - 1.0 * data[:, 1]
        return data

class VisibilityChecker:
    def __init__(self, w, h, f):
        self.w = w
        self.h = h
        self.f = f
        self.rn_d = DepthRenderer(frustum={'near': 0.1, 'far': 10., 'width': w, 'height': h}, f=f)

    def vertex_visibility(self, camera, mask=None):
        cam3d = ProjectPoints3D(**{k: getattr(camera, k) for k in camera.dterms if hasattr(camera, k)})

        in_viewport = np.logical_and(
            np.logical_and(np.round(camera.r[:, 0]) >= 0, np.round(camera.r[:, 0]) < self.w),
            np.logical_and(np.round(camera.r[:, 1]) >= 0, np.round(camera.r[:, 1]) < self.h),
        )

        if not hasattr(self.rn_d, 'camera') or not np.all(self.rn_d.camera.r == camera.r):
            self.rn_d.set(camera=camera)
        depth = self.rn_d.r

        proj = cam3d.r[in_viewport]
        d = proj[:, 2]
        idx = np.round(proj[:, [1, 0]].T).astype(np.int).tolist()

        visible = np.zeros(cam3d.shape[0], dtype=np.bool)
        visible[in_viewport] = np.abs(d - depth[tuple(idx)]) < 0.01

        if mask is not None:
            mask = cv2.erode(mask, np.ones((5, 5)))
            visible[in_viewport] = np.logical_and(visible[in_viewport], mask[tuple(idx)])

        return visible

    def face_visibility(self, camera, mask=None):
        v_vis = self.vertex_visibility(camera, mask)

        return np.min(v_vis[self.f], axis=1)

    def vertex_visibility_angle(self, camera):
        n = VertNormals(camera.v, self.f)
        v_cam = camera.v.r.dot(cv2.Rodrigues(camera.rt.r)[0]) + camera.t.r
        n_cam = n.r.dot(cv2.Rodrigues(camera.rt.r)[0])

        return np.sum(v_cam / (np.linalg.norm(v_cam, axis=1).reshape(-1, 1)) * -1 * n_cam, axis=1)

    def face_visibility_angle(self, camera):
        v_cam = camera.v.r.dot(cv2.Rodrigues(camera.rt.r)[0]) + camera.t.r
        f_cam = v_cam[self.f]
        f_norm = np.cross(f_cam[:, 0] - f_cam[:, 1], f_cam[:, 0] - f_cam[:, 2], axisa=1, axisb=1)
        f_norm /= np.linalg.norm(f_norm, axis=1).reshape(-1, 1)
        center = np.mean(f_cam, axis=1)

        return np.sum(center / (np.linalg.norm(center, axis=1).reshape(-1, 1)) * -1 * f_norm, axis=1)

class Texture:

    def __init__(self, tex_res, seams, mask, segm_template, gmm):
        self.tex_res = tex_res
        self.mask = mask
        self.face_mask = cv2.imread('thirdparty/pic2tex/assets/tex_face_mask_1000.png', flags=cv2.IMREAD_GRAYSCALE) / 255.
        self.face_mask = cv2.resize(self.face_mask, (tex_res, tex_res), interpolation=cv2.INTER_NEAREST)

        self.stitcher = Stitcher(seams, tex_res, self.mask)

        self.segm_template = None
        self.segm_template_id = None
        self.gmms = None

        self.segm_template = segm_template
        self.segm_template_id = to_ids(self.segm_template)
        self.gmms = gmm

        self.tex_agg = None
        self.vis_agg = None
        self.gmm_agg = None

    def add_iso(self, tex_current, vis, current_label, silh_err=0., inpaint=True):

        if self.tex_agg is None:
            self.vis_agg = vis
            self.tex_agg = tex_current
            self.silh_err_agg = np.ones_like(vis) * silh_err
            self.labels_agg = np.ones_like(vis) * current_label
            self.gmm_agg = np.zeros((self.tex_res, self.tex_res))
            self.init_face = np.mean(self.tex_agg, axis=2)

            if inpaint:
                return self.inpaint_segments(self.tex_agg, self.vis_agg), self.labels_agg
            else:
                return self.tex_agg, self.labels_agg

        pairwise_mask = np.logical_or(self.vis_agg < 1, vis < 1)

        _, ssim = compare_ssim(self.init_face * self.face_mask, np.mean(tex_current, axis=2) * self.face_mask,
                               full=True, data_range=1)
        ssim = (1 - ssim) / 2.
        ssim[ssim < 0] = 0

        gmm = np.zeros((self.tex_res, self.tex_res))
        self.gmm_agg = np.zeros((self.tex_res, self.tex_res))

        tex_agg_hsv = cv2.cvtColor(np.uint8(self.tex_agg * 255), cv2.COLOR_RGB2HSV) / 255.
        tex_current_hsv = cv2.cvtColor(np.uint8(tex_current * 255), cv2.COLOR_RGB2HSV) / 255.

        if self.segm_template is not None and self.gmms is not None:
            for i, color_id in enumerate(LABELS_REDUCED):
                if color_id != 'Unseen' and color_id != 'BG':
                    where = np.all(self.segm_template == LABELS_REDUCED[color_id], axis=2)
                    w = 10. if color_id in ['Arms', 'Legs'] else 1.

                    if np.max(where):
                        c = self.gmms[color_id].n_components

                        data = tex_current_hsv[where]
                        diff = data.reshape(-1, 1, 3) - self.gmms[color_id].means_
                        mahal = np.sqrt(np.sum((np.sum(diff.reshape(-1, c, 1, 3) * self.gmms[color_id].covariances_, axis=3) * diff), axis=2))
                        gmm[where] = np.min(mahal, axis=1) * w

                        data = tex_agg_hsv[where]
                        diff = data.reshape(-1, 1, 3) - self.gmms[color_id].means_
                        mahal = np.sqrt(np.sum((np.sum(diff.reshape(-1, c, 1, 3) * self.gmms[color_id].covariances_, axis=3) * diff), axis=2))
                        self.gmm_agg[where] = np.min(mahal, axis=1) * w

        unaries_agg = 2. * self.vis_agg + 20 * self.gmm_agg + 0.7 * self.silh_err_agg
        unaries_current = 2. * vis + 20 * gmm + 10 * ssim + 0.7 * silh_err

        labels_current = np.ones_like(self.labels_agg) * current_label

        self.tex_agg, update = self.stitcher.stich(self.tex_agg, tex_current, unaries_agg, unaries_current,
                                                   self.labels_agg, labels_current, pairwise_mask, self.segm_template_id)
        self.vis_agg[update == 1] = vis[update == 1]
        self.silh_err_agg[update == 1] = silh_err
        self.labels_agg[update == 1] = current_label

        if inpaint:
            return self._grow_tex(self.inpaint_segments(self.tex_agg, self.vis_agg)), self.labels_agg * self.mask
        else:
            return self.tex_agg, self.labels_agg * self.mask

    def inpaint_segments(self, tex, vis):

        if self.segm_template_id is not None:
            visible = vis < 0.95

            tmp = np.array(tex)
            for i, l in enumerate(LABELS_REDUCED):
                if l != 'Unseen' and l != 'BG':
                    seen = np.float32(np.logical_and(visible, self.segm_template_id == i))
                    area = 1 - cv2.erode(seen, np.ones((3, 3), dtype=np.uint8), iterations=2)

                    if np.max(seen):
                        part = cv2.inpaint(np.uint8(tex * 255), np.uint8(area * 255), 3, cv2.INPAINT_TELEA) / 255.
                        where = np.logical_and(np.logical_not(visible), self.segm_template_id == i)
                        tmp[where] = part[where]

            return tmp

        return self.inpaint(tex, vis)

    def inpaint(self, tex, vis):

        visible = np.float32(vis < 0.7)
        visible[self.mask < 1] = 0

        area = cv2.dilate(1 - visible, np.ones((3, 3), dtype=np.uint8), iterations=2)
        tex = cv2.inpaint(np.uint8(tex * 255), np.uint8(area * 255), 3, cv2.INPAINT_TELEA) / 255.

        return tex

    def _grow_tex(self, tex):
        kernel_size = np.int(self.vis_agg.shape[1] * 0.005)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        inpaint_area = cv2.dilate(1 - self.mask, np.ones((3, 3), dtype=np.uint8), iterations=3)

        tex_inpaint = cv2.inpaint(np.uint8(tex * 255), np.uint8(inpaint_area * 255), 3, cv2.INPAINT_TELEA)
        return (tex_inpaint * np.atleast_3d(cv2.dilate(self.mask, kernel))) / 255.

class OrthoProjectPoints(Ch):
    terms = 'near', 'far', 'width', 'height'
    dterms = 'v', 'rt', 't', 'left', 'right', 'bottom', 'top'

    def compute_r(self):
        return self.r_and_derivatives.r

    def compute_dr_wrt(self, wrt):
        if wrt not in [self.v, self.rt, self.t, self.left, self.right, self.bottom, self.top]:
            return None

        return self.r_and_derivatives.dr_wrt(wrt)

    def unproject_points(self, uvd, camera_space=False):
        tmp = np.hstack((
            col(2. * uvd[:, 0] / self.width - 1 + (self.right + self.left) / (self.right - self.left)).r * (self.right - self.left).r / 2.,
            col(2. * uvd[:, 1] / self.height - 1 + (self.bottom + self.top) / (self.bottom - self.top)).r * (self.bottom - self.top).r / 2.,
            np.ones((uvd.shape[0], 1))
        ))

        if camera_space:
            return tmp
        tmp -= self.t.r  # translate

        return tmp.dot(Rodrigues(self.rt).r.T)  # rotate

    @depends_on('t', 'rt')
    def view_mtx(self):
        R = cv2.Rodrigues(self.rt.r)[0]
        return np.hstack((R, col(self.t.r)))

    @property
    def r_and_derivatives(self):
        tmp = self.v.dot(Rodrigues(self.rt)) + self.t

        return ch.hstack((
            col(2. / (self.right - self.left) * tmp[:, 0] - (self.right + self.left) / (self.right - self.left) + 1.) * self.width / 2.,
            col(2. / (self.bottom - self.top) * tmp[:, 1] - (self.bottom + self.top) / (self.bottom - self.top) + 1.) * self.height / 2.,
        ))

class Isomapper():
    def __init__(self, vt, ft, tex_res, bgcolor=np.zeros(3)):
        vt3d = np.dstack((vt[:, 0] - 0.5, 1 - vt[:, 1] - 0.5, np.zeros(vt.shape[0])))[0]
        ortho = OrthoProjectPoints(rt=np.zeros(3), t=np.zeros(3), near=-1, far=1, left=-0.5, right=0.5, bottom=-0.5,
                                   top=0.5, width=tex_res, height=tex_res)
        self.tex_res = tex_res
        self.f = ft
        self.rn_tex = OrthoTexturedRenderer(v=vt3d, f=ft, ortho=ortho, vc=np.ones_like(vt3d), bgcolor=bgcolor)
        self.rn_vis = OrthoColoredRenderer(v=vt3d, f=ft, ortho=ortho, vc=np.ones_like(vt3d), bgcolor=np.zeros(3),
                                           num_channels=1)
        self.bgcolor = bgcolor
        self.iso_mask = np.array(self.rn_vis.r)

    def render(self, frame, proj_v, f, visible_faces=None, inpaint=True, inpaint_segments=None):
        h, w, _ = np.atleast_3d(frame).shape
        v2d = proj_v.r
        v2d_as_vt = np.dstack((v2d[:, 0] / w, 1 - v2d[:, 1] / h))[0]

        self.rn_tex.set(texture_image=frame, vt=v2d_as_vt, ft=f)
        tex = np.array(self.rn_tex.r)

        if visible_faces is not None:
            self.rn_vis.set(f=self.f[visible_faces])
            if inpaint:
                visible = cv2.erode(self.rn_vis.r, np.ones((self.tex_res // 100, self.tex_res // 100)))

                if inpaint_segments is None:
                    tex = np.atleast_3d(self.iso_mask) * cv2.inpaint(np.uint8(tex * 255), np.uint8((1 - visible) * 255),
                                                                     3, cv2.INPAINT_TELEA) / 255.
                else:
                    tmp = np.zeros_like(tex)
                    for i in range(np.max(inpaint_segments) + 1):
                        seen = np.logical_and(visible, inpaint_segments == i)
                        part = cv2.inpaint(np.uint8(tex * 255), np.uint8((1 - seen) * 255), 3, cv2.INPAINT_TELEA) / 255.
                        tmp[inpaint_segments == i] = part[inpaint_segments == i]

                    tex = np.atleast_3d(self.iso_mask) * tmp
            else:
                mask = np.atleast_3d(self.rn_vis.r)
                tex = mask * tex + (1 - mask) * self.bgcolor

        return tex

class TextureData:
    def __init__(self, tex_res, f, vt, ft, visibility):
        self.tex_res = tex_res
        self.visibility = visibility
        self.f = f

        self.isomapper = Isomapper(vt, ft, tex_res)
        self.iso_nearest = Isomapper(vt, ft, tex_res, bgcolor=np.array(LABELS_REDUCED['Unseen']) / 255.)
        self.mask = self.isomapper.iso_mask

        self.visibility_rn = VisibilityRenderer(vt, ft, tex_res, f)

    def get_data(self, frame, camera, silh, segm):
        f_vis = self.visibility.face_visibility(camera, silh)
        vis = self.visibility_rn.render(self._vis_angle(camera, silh))

        iso = self.isomapper.render(frame, camera, self.f, visible_faces=f_vis)
        iso_segm = self.iso_nearest.render(segm, camera, self.f, visible_faces=f_vis, inpaint=False)

        return vis, iso, iso_segm

    def _vis_angle(self, camera, silh):
        v_vis = self.visibility.vertex_visibility(camera, silh)
        v_angle = self.visibility.vertex_visibility_angle(camera)
        v_angle[np.logical_not(v_vis)] = 0

        return 1 - v_angle ** 2

LABELS_FULL = {
    'BG': [0, 0, 0],
    'Hat': [128, 0, 0],
    'Hair': [255, 0, 0],
    'Glove': [0, 85, 0],
    'Sunglasses': [170, 0, 51],
    'UpperClothes': [255, 85, 0],
    'Dress': [0, 0, 85],
    'Coat': [0, 119, 221],
    'Socks': [85, 85, 0],
    'Pants': [0, 85, 85],
    'Torso-skin': [85, 51, 0],
    'Scarf': [52, 86, 128],
    'Skirt': [0, 128, 0],
    'Face': [0, 0, 255],
    'LeftArm': [51, 170, 221],
    'RightArm': [0, 255, 255],
    'LeftLeg': [85, 255, 170],
    'RightLeg': [170, 255, 85],
    'LeftShoe': [255, 255, 0],
    'RightShoe': [255, 170, 0],
}

LABELS_REDUCED = {
    'BG': [0, 0, 0],
    'Hat': [128, 0, 0],
    'Hair': [255, 0, 0],
    'Glove': [0, 85, 0],
    'Face': [0, 0, 255],
    'UpperClothes': [255, 85, 0],
    'Dress': [0, 0, 85],
    'Coat': [0, 119, 221],
    'Socks': [85, 85, 0],
    'Pants': [0, 85, 85],
    'Torso-skin': [85, 51, 0],
    'Scarf': [52, 86, 128],
    'Skirt': [0, 128, 0],
    'Arms': [51, 170, 221],
    'Legs': [85, 255, 170],
    'Shoes': [255, 255, 0],

    'Unseen': [125, 125, 125],
}

LABELS_MIXTURES = {
    'BG': 1,
    'Hat': 3,
    'Hair': 3,
    'Glove': 2,
    'Face': 5,
    'UpperClothes': 4,
    'Dress': 4,
    'Coat': 4,
    'Socks': 2,
    'Pants': 4,
    'Torso-skin': 2,
    'Scarf': 3,
    'Skirt': 4,
    'Arms': 1,
    'Legs': 1,
    'Shoes': 3,

    'Unseen': 1,
}

LABEL_COMP = np.ones(len(LABELS_REDUCED)) - np.eye(len(LABELS_REDUCED))
LABEL_COMP[0, 0] = 1.

def main(data_file, frame_dir, segm_dir, out_file, num_iter):
    # Step 1: Make unwraps

    data = pkl.load(open(data_file, 'rb'))

    segm_files = np.array(sorted(glob(os.path.join(segm_dir, '*.png')) + glob(os.path.join(segm_dir, '*.jpg'))))
    
    if len(segm_files) < 3:
        if len(segm_files) < 2:
            segm_files = np.array((segm_files[0], segm_files[0], segm_files[0]))
        else:
            segm_files = np.array((segm_files[0], segm_files[1], segm_files[1]))

    frame_files = np.array(sorted(glob(os.path.join(frame_dir, '*.png')) + glob(os.path.join(frame_dir, '*.jpg'))))

    if len(frame_files) < 3:
        if len(frame_files) < 2:
            frame_files = np.array((frame_files[0], frame_files[0], frame_files[0]))
        else:
            frame_files = np.array((frame_files[0], frame_files[1], frame_files[1]))

    vt = np.load('assets/basicModel_vt.npy')
    ft = np.load('assets/basicModel_ft.npy')
    f = np.load('assets/basicModel_f.npy')

    verts = data['vertices']

    if len(verts) < 3:
        if len(verts) < 2:
            verts = np.array((data['vertices'][0], data['vertices'][0], data['vertices'][0]), dtype=object)
        else:
            verts = np.array((data['vertices'][0], data['vertices'][1], data['vertices'][1]), dtype=object)

    camera_c = data['camera_c']
    camera_f = data['camera_f']
    width = data['width']
    height = data['height']

    camera = ProjectPoints(t=np.zeros(3), rt=np.array([-np.pi, 0., 0.]), c=camera_c, f=camera_f, k=np.zeros(5))

    visibility = VisibilityChecker(width, height, f)

    texture = TextureData(1000, f, vt, ft, visibility)

    isos, vises, iso_segms = [], [], []

    #print('Unwrapping inputs...')
    for i, (v, frame_file, segm_file) in enumerate(zip(verts, frame_files, segm_files)):
        frame = cv2.resize(cv2.imread(frame_file), (1080, 1080)) / 255.
        segm = read_segmentation(segm_file) / 255.
        mask = np.float32(np.any(segm > 0, axis=-1))

        camera.set(v=v)

        vis, iso, iso_segm = texture.get_data(frame, camera, mask, segm)

        vises.append(vis)
        isos.append(iso)
        iso_segms.append(np.uint8(iso_segm * 255))

    # Step 2: Segm vote gmm

    iso_mask = cv2.imread('assets/tex_mask_1000.png', flags=cv2.IMREAD_GRAYSCALE) / 255.
    iso_mask = cv2.resize(iso_mask, (1000, 1000), interpolation=cv2.INTER_NEAREST)

    voting = np.zeros((1000, 1000, len(LABELS_REDUCED)))

    gmms = {}
    gmm_pixels = {}

    for color_id in LABELS_REDUCED:
        gmms[color_id] = GaussianMixture(LABELS_MIXTURES[color_id])
        gmm_pixels[color_id] = []

    for frame, segm, vis in zip(isos, iso_segms, vises):
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) / 255.
        tex_segm = read_segmentation(segm)
        tex_weights = 1 - vis
        tex_weights = np.sqrt(tex_weights)

        for i, color_id in enumerate(LABELS_REDUCED):
            if color_id != 'Unseen' and color_id != 'BG':
                where = np.all(tex_segm == LABELS_REDUCED[color_id], axis=2)
                voting[where, i] += tex_weights[where]
                gmm_pixels[color_id].extend(frame[where].tolist())

    #print('Fitting GMMs...')
    for color_id in LABELS_REDUCED:
        if gmm_pixels[color_id]:
            gmms[color_id].fit(np.array(gmm_pixels[color_id]))

    for i, color_id in enumerate(LABELS_REDUCED):
        if color_id == 'Unseen' or color_id == 'BG':
            voting[:, i] = -10

    voting[iso_mask == 0] = 0
    voting[iso_mask == 0, 0] = 1

    unaries = np.ascontiguousarray((1 - voting / len(isos)) * 10)
    pairwise = np.ascontiguousarray(LABEL_COMP)

    seams = np.load('assets/basicModel_seams.npy')
    edge_idx = pkl.load(open('assets/basicModel_edge_idx_1000_.pkl', 'rb'))

    dr_v = signal.convolve2d(iso_mask, [[-1, 1]])[:, 1:]
    dr_h = signal.convolve2d(iso_mask, [[-1], [1]])[1:, :]

    where_v = iso_mask - dr_v
    where_h = iso_mask - dr_h

    idxs = np.arange(1000 ** 2).reshape(1000, 1000)
    v_edges_from = idxs[:-1, :][where_v[:-1, :] == 1].flatten()
    v_edges_to = idxs[1:, :][where_v[:-1, :] == 1].flatten()
    h_edges_from = idxs[:, :-1][where_h[:, :-1] == 1].flatten()
    h_edges_to = idxs[:, 1:][where_h[:, :-1] == 1].flatten()

    s_edges_from, s_edges_to = edges_seams(seams, 1000, edge_idx)

    edges_from = np.r_[v_edges_from, h_edges_from, s_edges_from]
    edges_to = np.r_[v_edges_to, h_edges_to, s_edges_to]
    edges_w = np.r_[np.ones_like(v_edges_from), np.ones_like(h_edges_from), np.ones_like(s_edges_from)]

    gc = gco.GCO()
    gc.create_general_graph(1000 ** 2, pairwise.shape[0], True)
    gc.set_data_cost(unaries.reshape(1000 ** 2, pairwise.shape[0]))

    gc.set_all_neighbors(edges_from, edges_to, edges_w)
    gc.set_smooth_cost(pairwise)
    gc.swap(-1)

    labels = gc.get_labels().reshape(1000, 1000)
    gc.destroy_graph()

    segm_colors = np.zeros((1000, 1000, 3), dtype=np.uint8)

    for i, color_id in enumerate(LABELS_REDUCED):
        segm_colors[labels == i] = LABELS_REDUCED[color_id]

    # Step 3: Stitch texture

    seams = np.load('assets/basicModel_seams.npy')
    mask = cv2.imread('assets/tex_mask_1000.png', flags=cv2.IMREAD_GRAYSCALE) / 255.

    segm_template = read_segmentation(segm_colors)

    num_labels = len(isos)
    texture = Texture(1000, seams, mask, segm_template, gmms)

    texture_agg = isos[0]
    visibility_agg = np.array(vises[0])

    tex, _ = texture.add_iso(texture_agg, visibility_agg, np.zeros_like(visibility_agg), inpaint=False)

    #print('Aggregating texture...')
    for i in trange(num_iter):
        rl = np.random.choice(num_labels)
        texture_agg, labels = texture.add_iso(isos[rl], vises[rl], rl, inpaint=i == (num_iter-1))

    #print('saving {}...'.format(os.path.basename(out_file)))

    if not os.path.exists('./output'):
        os.makedirs('output')

    cv2.imwrite(out_file, np.uint8(255 * texture_agg))

if __name__ == '__main__':
    main('data/frame_data.pkl', 'data/frames', 'data/segmentations', 'output/sample_texture.jpg', 15)
