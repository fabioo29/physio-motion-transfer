
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

_DETECTRON_OPS_LIB = 'libcaffe2_detectron_ops_gpu.so'
_CMAKE_INSTALL_PREFIX = '/usr/local'
HIGHEST_BACKBONE_LVL = 5
LOWEST_BACKBONE_LVL = 2

import argparse
import cv2  
# import glob 
import copy
import logging
import os
import sys
import six
import time
import importlib
import pprint
import contextlib
import re
import scipy.sparse
import collections
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

# sys.path.append('/root/pmt/thirdparty/densepose/np') # path to where detectron

# bash this with model missing in 'filename' and add the path to sys
# find / -type f -iname "filename*"
# sys.path.append('path/to/where/cafffe2/is') 

from caffe2.python import workspace
from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import dyndep
from caffe2.python import scope
from caffe2.python import cnn
from caffe2.python import muji
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags

from pycocotools import mask as COCOmask
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from .assets.config import assert_and_infer_cfg
from .assets.config import cfg
from .assets.config import merge_cfg_from_file
from .assets.config import load_cfg

import cython_bbox as cython_bbox
import cython_nms as cython_nms

from collections import defaultdict
from collections import OrderedDict

from six import string_types
from six.moves import cPickle as pickle
from six.moves import urllib

from glob import glob
from scipy.io import loadmat
from matplotlib.patches import Polygon

box_utils_bbox_overlaps = cython_bbox.bbox_overlaps
bbox_overlaps = cython_bbox.bbox_overlaps

logger = logging.getLogger(__name__)

FpnLevelInfo = collections.namedtuple(
    'FpnLevelInfo',
    ['blobs', 'dims', 'spatial_scales']
)

def _progress_bar(count, total):
    """Report download progress.
    Credit:
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write(
        '  [{}] {}% of {:.1f}MB file  \r'.
        format(bar, percents, total / 1024 / 1024)
    )
    sys.stdout.flush()
    if count >= total:
        sys.stdout.write('\n')

def download_url(
    url, dst_file_path, chunk_size=8192, progress_hook=_progress_bar
):
    """Download url and write it to dst_file_path.
    Credit:
    https://stackoverflow.com/questions/2028517/python-urllib2-progress-hook
    """
    response = urllib.request.urlopen(url)
    if six.PY2:
        total_size = response.info().getheader('Content-Length').strip()
    else:
        total_size = response.info().get('Content-Length').strip()
    total_size = int(total_size)
    bytes_so_far = 0

    with open(dst_file_path, 'wb') as f:
        while 1:
            chunk = response.read(chunk_size)
            bytes_so_far += len(chunk)
            if not chunk:
                break
            if progress_hook:
                progress_hook(bytes_so_far, total_size)
            f.write(chunk)

    return bytes_so_far

def get_class_string(class_index, score, dataset):
    class_text = dataset.classes[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0')

def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines

def colormap(rgb=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list

def keypoint_utils_get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    # Keypoints are not available in the COCO json for the test split, so we
    # provide them here.
    keypoints = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    keypoint_flip_map = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }
    return keypoints, keypoint_flip_map

def convert_from_cls_format(cls_boxes, cls_segms, cls_keyps):
    """Convert from the class boxes/segms/keyps format generated by the testing
    code.
    """
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, keyps, classes

def vis_utils_vis_one_image(
        im, boxes, segms=None, keypoints=None, body_uv=None, thresh=0.9,
        kp_thresh=2, dpi=200, box_alpha=0.0, dataset=None, show_class=False,
        ext='pdf'):
    """Visual debugging of detections."""

    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return

    dataset_keypoints, _ = keypoint_utils_get_keypoints()

    if segms is not None and len(segms) > 0:
        masks = mask_util.decode(segms)

    color_list = colormap(rgb=True) / 255

    kp_lines = kp_connections(dataset_keypoints)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    IUV_fields = body_uv[1]
    #
    All_Coords = np.zeros(im.shape)
    All_inds = np.zeros([im.shape[0],im.shape[1]])
    K = 26
    ##
    inds = np.argsort(boxes[:,4])
    ##
    for i, ind in enumerate(inds):
        entry = boxes[ind,:]
        if entry[4] > 0.65:
            entry=entry[0:4].astype(int)
            ####
            output = IUV_fields[ind]
            ####
            All_Coords_Old = All_Coords[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2],:]
            All_Coords_Old[All_Coords_Old==0]=output.transpose([1,2,0])[All_Coords_Old==0]
            All_Coords[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2],:]= All_Coords_Old
            ###
            CurrentMask = (output[0,:,:]>0).astype(np.float32)
            All_inds_old = All_inds[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2]]
            All_inds_old[All_inds_old==0] = CurrentMask[All_inds_old==0]*i
            All_inds[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2]] = All_inds_old
    #
    All_Coords[:,:,1:3] = 255. * All_Coords[:,:,1:3]
    All_Coords[All_Coords>255] = 255.
    All_Coords = All_Coords.astype(np.uint8)

    return All_Coords

def envu_get_detectron_ops_lib():
    """Retrieve Detectron ops library."""
    # Candidate prefixes for detectron ops lib path
    prefixes = [_CMAKE_INSTALL_PREFIX, sys.prefix, sys.exec_prefix] + sys.path
    # Candidate subdirs for detectron ops lib
    subdirs = ['lib', 'torch/lib']
    # Try to find detectron ops lib
    for prefix in prefixes:
        for subdir in subdirs:
            ops_path = os.path.join(prefix, subdir, _DETECTRON_OPS_LIB)
            if os.path.exists(ops_path):
                #print('Found Detectron ops lib: {}'.format(ops_path))
                return ops_path
    raise Exception('Detectron ops lib not found')

def c2_utils_import_detectron_ops():
    """Import Detectron ops."""
    detectron_ops_lib = envu_get_detectron_ops_lib()
    dyndep.InitOpsLibrary(detectron_ops_lib)

def dummy_datasets_get_coco_dataset():
    """A dummy COCO dataset that includes only the 'classes' field."""
    ds = AttrDict()
    classes = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def im_detect_body_uv(model, im_scale, boxes):
    """Compute body uv predictions."""
    M = cfg.BODY_UV_RCNN.HEATMAP_SIZE
    P = cfg.BODY_UV_RCNN.NUM_PATCHES
    if boxes.shape[0] == 0:
        pred_body_uvs = np.zeros((0, P, M, M), np.float32)
        return pred_body_uvs

    inputs = {'body_uv_rois': _get_rois_blob(boxes, im_scale)}

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'body_uv_rois')

    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v)
    workspace.RunNet(model.body_uv_net.Proto().name)

    AnnIndex = workspace.FetchBlob(core.ScopedName('AnnIndex')).squeeze()
    Index_UV = workspace.FetchBlob(core.ScopedName('Index_UV')).squeeze()
    U_uv = workspace.FetchBlob(core.ScopedName('U_estimated')).squeeze()
    V_uv = workspace.FetchBlob(core.ScopedName('V_estimated')).squeeze()

    # In case of 1
    if AnnIndex.ndim == 3:
        AnnIndex = np.expand_dims(AnnIndex, axis=0)
    if Index_UV.ndim == 3:
        Index_UV = np.expand_dims(Index_UV, axis=0)
    if U_uv.ndim == 3:
        U_uv = np.expand_dims(U_uv, axis=0)
    if V_uv.ndim == 3:
        V_uv = np.expand_dims(V_uv, axis=0)

    K = cfg.BODY_UV_RCNN.NUM_PATCHES + 1
    outputs = []

    for ind, entry in enumerate(boxes):
        # Compute ref box width and height
        bx = int(max(entry[2] - entry[0], 1))
        by = int(max(entry[3] - entry[1], 1))

        # preds[ind] axes are CHW; bring p axes to WHC
        CurAnnIndex = np.swapaxes(AnnIndex[ind], 0, 2)
        CurIndex_UV = np.swapaxes(Index_UV[ind], 0, 2)
        CurU_uv = np.swapaxes(U_uv[ind], 0, 2)
        CurV_uv = np.swapaxes(V_uv[ind], 0, 2)

        # Resize p from (HEATMAP_SIZE, HEATMAP_SIZE, c) to (int(bx), int(by), c)
        CurAnnIndex = cv2.resize(CurAnnIndex, (by, bx))
        CurIndex_UV = cv2.resize(CurIndex_UV, (by, bx))
        CurU_uv = cv2.resize(CurU_uv, (by, bx))
        CurV_uv = cv2.resize(CurV_uv, (by, bx))

        # Bring Cur_Preds axes back to CHW
        CurAnnIndex = np.swapaxes(CurAnnIndex, 0, 2)
        CurIndex_UV = np.swapaxes(CurIndex_UV, 0, 2)
        CurU_uv = np.swapaxes(CurU_uv, 0, 2)
        CurV_uv = np.swapaxes(CurV_uv, 0, 2)

        # Removed squeeze calls due to singleton dimension issues
        CurAnnIndex = np.argmax(CurAnnIndex, axis=0)
        CurIndex_UV = np.argmax(CurIndex_UV, axis=0)
        CurIndex_UV = CurIndex_UV * (CurAnnIndex>0).astype(np.float32)

        output = np.zeros([3, int(by), int(bx)], dtype=np.float32)
        output[0] = CurIndex_UV

        for part_id in range(1, K):
            CurrentU = CurU_uv[part_id]
            CurrentV = CurV_uv[part_id]
            output[1, CurIndex_UV==part_id] = CurrentU[CurIndex_UV==part_id]
            output[2, CurIndex_UV==part_id] = CurrentV[CurIndex_UV==part_id]
        outputs.append(output)

    num_classes = cfg.MODEL.NUM_CLASSES
    cls_bodys = [[] for _ in range(num_classes)]
    person_idx = keypoint_utils_get_person_class_index()
    cls_bodys[person_idx] = outputs

    return cls_bodys

def compute_oks(src_keypoints, src_roi, dst_keypoints, dst_roi):
    """Compute OKS for predicted keypoints wrt gt_keypoints.
    src_keypoints: 4xK
    src_roi: 4x1
    dst_keypoints: Nx4xK
    dst_roi: Nx4
    """

    sigmas = np.array([
        .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87,
        .87, .89, .89]) / 10.0
    vars = (sigmas * 2)**2

    # area
    src_area = (src_roi[2] - src_roi[0] + 1) * (src_roi[3] - src_roi[1] + 1)

    # measure the per-keypoint distance if keypoints visible
    dx = dst_keypoints[:, 0, :] - src_keypoints[0, :]
    dy = dst_keypoints[:, 1, :] - src_keypoints[1, :]

    e = (dx**2 + dy**2) / vars / (src_area + np.spacing(1)) / 2
    e = np.sum(np.exp(-e), axis=1) / e.shape[1]

    return e

def keypoint_utils_nms_oks(kp_predictions, rois, thresh):
    """Nms based on kp predictions."""
    scores = np.mean(kp_predictions[:, 2, :], axis=1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = compute_oks(
            kp_predictions[i], rois[i], kp_predictions[order[1:]],
            rois[order[1:]])
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def scores_to_probs(scores):
    """Transforms CxHxW of scores to probabilities spatially."""
    channels = scores.shape[0]
    for c in range(channels):
        temp = scores[c, :, :]
        max_score = temp.max()
        temp = np.exp(temp - max_score) / np.sum(np.exp(temp - max_score))
        scores[c, :, :] = temp
    return scores

def keypoint_utils_heatmaps_to_keypoints(maps, rois):
    """Extract predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    """
    # This function converts a discrete image coordinate in a HEATMAP_SIZE x
    # HEATMAP_SIZE image to a continuous keypoint coordinate. We maintain
    # consistency with keypoints_to_heatmap_labels by using the conversion from
    # Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
    # continuous coordinate.
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = np.maximum(widths, 1)
    heights = np.maximum(heights, 1)
    widths_ceil = np.ceil(widths)
    heights_ceil = np.ceil(heights)

    # NCHW to NHWC for use with OpenCV
    maps = np.transpose(maps, [0, 2, 3, 1])
    min_size = cfg.KRCNN.INFERENCE_MIN_SIZE
    xy_preds = np.zeros(
        (len(rois), 4, cfg.KRCNN.NUM_KEYPOINTS), dtype=np.float32)
    for i in range(len(rois)):
        if min_size > 0:
            roi_map_width = int(np.maximum(widths_ceil[i], min_size))
            roi_map_height = int(np.maximum(heights_ceil[i], min_size))
        else:
            roi_map_width = widths_ceil[i]
            roi_map_height = heights_ceil[i]
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height
        roi_map = cv2.resize(
            maps[i], (roi_map_width, roi_map_height),
            interpolation=cv2.INTER_CUBIC)
        # Bring back to CHW
        roi_map = np.transpose(roi_map, [2, 0, 1])
        roi_map_probs = scores_to_probs(roi_map.copy())
        w = roi_map.shape[2]
        for k in range(cfg.KRCNN.NUM_KEYPOINTS):
            pos = roi_map[k, :, :].argmax()
            x_int = pos % w
            y_int = (pos - x_int) // w
            assert (roi_map_probs[k, y_int, x_int] ==
                    roi_map_probs[k, :, :].max())
            x = (x_int + 0.5) * width_correction
            y = (y_int + 0.5) * height_correction
            xy_preds[i, 0, k] = x + offset_x[i]
            xy_preds[i, 1, k] = y + offset_y[i]
            xy_preds[i, 2, k] = roi_map[k, y_int, x_int]
            xy_preds[i, 3, k] = roi_map_probs[k, y_int, x_int]

    return xy_preds

def keypoint_utils_get_person_class_index():
    """Index of the person class in COCO."""
    return 1

def keypoint_results(cls_boxes, pred_heatmaps, ref_boxes):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_keyps = [[] for _ in range(num_classes)]
    person_idx = keypoint_utils_get_person_class_index()
    xy_preds = keypoint_utils_heatmaps_to_keypoints(pred_heatmaps, ref_boxes)

    # NMS OKS
    if cfg.KRCNN.NMS_OKS:
        keep = keypoint_utils_nms_oks(xy_preds, ref_boxes, 0.3)
        xy_preds = xy_preds[keep, :, :]
        ref_boxes = ref_boxes[keep, :]
        pred_heatmaps = pred_heatmaps[keep, :, :, :]
        cls_boxes[person_idx] = cls_boxes[person_idx][keep, :]

    kps = [xy_preds[i] for i in range(xy_preds.shape[0])]
    cls_keyps[person_idx] = kps
    return cls_keyps

def im_detect_keypoints(model, im_scale, boxes):
    """Infer instance keypoint poses. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scales (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)

    Returns:
        pred_heatmaps (ndarray): R x J x M x M array of keypoint location
            logits (softmax inputs) for each of the J keypoint types output
            by the network (must be processed by keypoint_results to convert
            into point predictions in the original image coordinate space)
    """
    M = cfg.KRCNN.HEATMAP_SIZE
    if boxes.shape[0] == 0:
        pred_heatmaps = np.zeros((0, cfg.KRCNN.NUM_KEYPOINTS, M, M), np.float32)
        return pred_heatmaps

    inputs = {'keypoint_rois': _get_rois_blob(boxes, im_scale)}

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'keypoint_rois')

    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v)
    workspace.RunNet(model.keypoint_net.Proto().name)

    pred_heatmaps = workspace.FetchBlob(core.ScopedName('kps_score')).squeeze()

    # In case of 1
    if pred_heatmaps.ndim == 3:
        pred_heatmaps = np.expand_dims(pred_heatmaps, axis=0)

    return pred_heatmaps

def combine_heatmaps_size_dep(hms_ts, ds_ts, us_ts, boxes, heur_f):
    """Combines heatmaps while taking object sizes into account."""
    assert len(hms_ts) == len(ds_ts) and len(ds_ts) == len(us_ts), \
        'All sets of hms must be tagged with downscaling and upscaling flags'

    # Classify objects into small+medium and large based on their box areas
    areas = box_utils_boxes_area(boxes)
    sm_objs = areas < cfg.TEST.KPS_AUG.AREA_TH
    l_objs = areas >= cfg.TEST.KPS_AUG.AREA_TH

    # Combine heatmaps computed under different transformations for each object
    hms_c = np.zeros_like(hms_ts[0])

    for i in range(hms_c.shape[0]):
        hms_to_combine = []
        for hms_t, ds_t, us_t in zip(hms_ts, ds_ts, us_ts):
            # Discard downscaling predictions for small and medium objects
            if sm_objs[i] and ds_t:
                continue
            # Discard upscaling predictions for large objects
            if l_objs[i] and us_t:
                continue
            hms_to_combine.append(hms_t[i])
        hms_c[i] = heur_f(hms_to_combine)

    return hms_c

def im_detect_keypoints_aspect_ratio(
    model, im, aspect_ratio, boxes, hflip=False
):
    """Detects keypoints at the given width-relative aspect ratio."""

    # Perform keypoint detectionon the transformed image
    im_ar = image_utils_aspect_ratio_rel(im, aspect_ratio)
    boxes_ar = box_utils_aspect_ratio(boxes, aspect_ratio)

    if hflip:
        heatmaps_ar = im_detect_keypoints_hflip(
            model, im_ar, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes_ar
        )
    else:
        im_scale = im_conv_body_only(
            model, im_ar, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE
        )
        heatmaps_ar = im_detect_keypoints(model, im_scale, boxes_ar)

    return heatmaps_ar

def im_detect_keypoints_scale(
    model, im, target_scale, target_max_size, boxes, hflip=False
):
    """Computes keypoint predictions at the given scale."""
    if hflip:
        heatmaps_scl = im_detect_keypoints_hflip(
            model, im, target_scale, target_max_size, boxes
        )
    else:
        im_scale = im_conv_body_only(model, im, target_scale, target_max_size)
        heatmaps_scl = im_detect_keypoints(model, im_scale, boxes)
    return heatmaps_scl

def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    # Keypoints are not available in the COCO json for the test split, so we
    # provide them here.
    keypoints = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    keypoint_flip_map = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }
    return keypoints, keypoint_flip_map

def keypoint_utils_flip_heatmaps(heatmaps):
    """Flip heatmaps horizontally."""
    keypoints, flip_map = get_keypoints()
    heatmaps_flipped = heatmaps.copy()
    for lkp, rkp in flip_map.items():
        lid = keypoints.index(lkp)
        rid = keypoints.index(rkp)
        heatmaps_flipped[:, rid, :, :] = heatmaps[:, lid, :, :]
        heatmaps_flipped[:, lid, :, :] = heatmaps[:, rid, :, :]
    heatmaps_flipped = heatmaps_flipped[:, :, :, ::-1]
    return heatmaps_flipped

def im_detect_keypoints_hflip(model, im, target_scale, target_max_size, boxes):
    """Computes keypoint predictions on the horizontally flipped image.
    Function signature is the same as for im_detect_keypoints_aug.
    """
    # Compute keypoints for the flipped image
    im_hf = im[:, ::-1, :]
    boxes_hf = box_utils_flip_boxes(boxes, im.shape[1])

    im_scale = im_conv_body_only(model, im_hf, target_scale, target_max_size)
    heatmaps_hf = im_detect_keypoints(model, im_scale, boxes_hf)

    # Invert the predicted keypoints
    heatmaps_inv = keypoint_utils_flip_heatmaps(heatmaps_hf)

    return heatmaps_inv

def im_detect_keypoints_aug(model, im, boxes):
    """Computes keypoint predictions with test-time augmentations.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): BGR image to test
        boxes (ndarray): R x 4 array of bounding boxes

    Returns:
        heatmaps (ndarray): R x J x M x M array of keypoint location logits
    """

    # Collect heatmaps predicted under different transformations
    heatmaps_ts = []
    # Tag predictions computed under downscaling and upscaling transformations
    ds_ts = []
    us_ts = []

    def add_heatmaps_t(heatmaps_t, ds_t=False, us_t=False):
        heatmaps_ts.append(heatmaps_t)
        ds_ts.append(ds_t)
        us_ts.append(us_t)

    # Compute the heatmaps for the original image (identity transform)
    im_scale = im_conv_body_only(model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)
    heatmaps_i = im_detect_keypoints(model, im_scale, boxes)
    add_heatmaps_t(heatmaps_i)

    # Perform keypoints detection on the horizontally flipped image
    if cfg.TEST.KPS_AUG.H_FLIP:
        heatmaps_hf = im_detect_keypoints_hflip(
            model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes
        )
        add_heatmaps_t(heatmaps_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.KPS_AUG.SCALES:
        ds_scl = scale < cfg.TEST.SCALE
        us_scl = scale > cfg.TEST.SCALE
        heatmaps_scl = im_detect_keypoints_scale(
            model, im, scale, cfg.TEST.KPS_AUG.MAX_SIZE, boxes
        )
        add_heatmaps_t(heatmaps_scl, ds_scl, us_scl)

        if cfg.TEST.KPS_AUG.SCALE_H_FLIP:
            heatmaps_scl_hf = im_detect_keypoints_scale(
                model, im, scale, cfg.TEST.KPS_AUG.MAX_SIZE, boxes, hflip=True
            )
            add_heatmaps_t(heatmaps_scl_hf, ds_scl, us_scl)

    # Compute keypoints at different aspect ratios
    for aspect_ratio in cfg.TEST.KPS_AUG.ASPECT_RATIOS:
        heatmaps_ar = im_detect_keypoints_aspect_ratio(
            model, im, aspect_ratio, boxes
        )
        add_heatmaps_t(heatmaps_ar)

        if cfg.TEST.KPS_AUG.ASPECT_RATIO_H_FLIP:
            heatmaps_ar_hf = im_detect_keypoints_aspect_ratio(
                model, im, aspect_ratio, boxes, hflip=True
            )
            add_heatmaps_t(heatmaps_ar_hf)

    # Select the heuristic function for combining the heatmaps
    if cfg.TEST.KPS_AUG.HEUR == 'HM_AVG':
        np_f = np.mean
    elif cfg.TEST.KPS_AUG.HEUR == 'HM_MAX':
        np_f = np.amax
    else:
        raise NotImplementedError(
            'Heuristic {} not supported'.format(cfg.TEST.KPS_AUG.HEUR)
        )

    def heur_f(hms_ts):
        return np_f(hms_ts, axis=0)

    # Combine the heatmaps
    if cfg.TEST.KPS_AUG.SCALE_SIZE_DEP:
        heatmaps_c = combine_heatmaps_size_dep(
            heatmaps_ts, ds_ts, us_ts, boxes, heur_f
        )
    else:
        heatmaps_c = heur_f(heatmaps_ts)

    return heatmaps_c

def box_utils_expand_boxes(boxes, scale):
    """Expand an array of boxes by a given scale."""
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp

def segm_results(cls_boxes, masks, ref_boxes, im_h, im_w):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_segms = [[] for _ in range(num_classes)]
    mask_ind = 0
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    M = cfg.MRCNN.RESOLUTION
    scale = (M + 2.0) / M
    ref_boxes = box_utils_expand_boxes(ref_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)

    # skip j = 0, because it's the background class
    for j in range(1, num_classes):
        segms = []
        for _ in range(cls_boxes[j].shape[0]):
            if cfg.MRCNN.CLS_SPECIFIC_MASK:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, j, :, :]
            else:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, 0, :, :]

            ref_box = ref_boxes[mask_ind, :]
            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > cfg.MRCNN.THRESH_BINARIZE, dtype=np.uint8)
            im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_w)
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_h)

            im_mask[y_0:y_1, x_0:x_1] = mask[
                (y_0 - ref_box[1]):(y_1 - ref_box[1]),
                (x_0 - ref_box[0]):(x_1 - ref_box[0])
            ]

            # Get RLE encoding used by the COCO evaluation API
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F')
            )[0]
            segms.append(rle)

            mask_ind += 1

        cls_segms[j] = segms

    assert mask_ind == masks.shape[0]
    return cls_segms

def im_detect_mask(model, im_scale, boxes):
    """Infer instance segmentation masks. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scales (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)

    Returns:
        pred_masks (ndarray): R x K x M x M array of class specific soft masks
            output by the network (must be processed by segm_results to convert
            into hard masks in the original image coordinate space)
    """
    M = cfg.MRCNN.RESOLUTION
    if boxes.shape[0] == 0:
        pred_masks = np.zeros((0, M, M), np.float32)
        return pred_masks

    inputs = {'mask_rois': _get_rois_blob(boxes, im_scale)}
    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'mask_rois')

    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v)
    workspace.RunNet(model.mask_net.Proto().name)

    # Fetch masks
    pred_masks = workspace.FetchBlob(
        core.ScopedName('mask_fcn_probs')
    ).squeeze()

    if cfg.MRCNN.CLS_SPECIFIC_MASK:
        pred_masks = pred_masks.reshape([-1, cfg.MODEL.NUM_CLASSES, M, M])
    else:
        pred_masks = pred_masks.reshape([-1, 1, M, M])

    return pred_masks

def im_detect_mask_aspect_ratio(model, im, aspect_ratio, boxes, hflip=False):
    """Computes mask detections at the given width-relative aspect ratio."""

    # Perform mask detection on the transformed image
    im_ar = image_utils_aspect_ratio_rel(im, aspect_ratio)
    boxes_ar = box_utils_aspect_ratio(boxes, aspect_ratio)

    if hflip:
        masks_ar = im_detect_mask_hflip(
            model, im_ar, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes_ar
        )
    else:
        im_scale = im_conv_body_only(
            model, im_ar, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE
        )
        masks_ar = im_detect_mask(model, im_scale, boxes_ar)

    return masks_ar

def im_detect_mask_scale(
    model, im, target_scale, target_max_size, boxes, hflip=False
):
    """Computes masks at the given scale."""
    if hflip:
        masks_scl = im_detect_mask_hflip(
            model, im, target_scale, target_max_size, boxes
        )
    else:
        im_scale = im_conv_body_only(model, im, target_scale, target_max_size)
        masks_scl = im_detect_mask(model, im_scale, boxes)
    return masks_scl

def im_detect_mask_hflip(model, im, target_scale, target_max_size, boxes):
    """Performs mask detection on the horizontally flipped image.
    Function signature is the same as for im_detect_mask_aug.
    """
    # Compute the masks for the flipped image
    im_hf = im[:, ::-1, :]
    boxes_hf = box_utils_flip_boxes(boxes, im.shape[1])

    im_scale = im_conv_body_only(model, im_hf, target_scale, target_max_size)
    masks_hf = im_detect_mask(model, im_scale, boxes_hf)

    # Invert the predicted soft masks
    masks_inv = masks_hf[:, :, :, ::-1]

    return masks_inv

def im_conv_body_only(model, im, target_scale, target_max_size):
    """Runs `model.conv_body_net` on the given image `im`."""
    im_blob, im_scale, _im_info = blob_utils_get_image_blob(
        im, target_scale, target_max_size
    )
    workspace.FeedBlob(core.ScopedName('data'), im_blob)
    workspace.RunNet(model.conv_body_net.Proto().name)
    return im_scale

def im_detect_mask_aug(model, im, boxes):
    """Performs mask detection with test-time augmentations.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): BGR image to test
        boxes (ndarray): R x 4 array of bounding boxes

    Returns:
        masks (ndarray): R x K x M x M array of class specific soft masks
    """
    assert not cfg.TEST.MASK_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'

    # Collect masks computed under different transformations
    masks_ts = []

    # Compute masks for the original image (identity transform)
    im_scale_i = im_conv_body_only(model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)
    masks_i = im_detect_mask(model, im_scale_i, boxes)
    masks_ts.append(masks_i)

    # Perform mask detection on the horizontally flipped image
    if cfg.TEST.MASK_AUG.H_FLIP:
        masks_hf = im_detect_mask_hflip(
            model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes
        )
        masks_ts.append(masks_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.MASK_AUG.SCALES:
        max_size = cfg.TEST.MASK_AUG.MAX_SIZE
        masks_scl = im_detect_mask_scale(model, im, scale, max_size, boxes)
        masks_ts.append(masks_scl)

        if cfg.TEST.MASK_AUG.SCALE_H_FLIP:
            masks_scl_hf = im_detect_mask_scale(
                model, im, scale, max_size, boxes, hflip=True
            )
            masks_ts.append(masks_scl_hf)

    # Compute masks at different aspect ratios
    for aspect_ratio in cfg.TEST.MASK_AUG.ASPECT_RATIOS:
        masks_ar = im_detect_mask_aspect_ratio(model, im, aspect_ratio, boxes)
        masks_ts.append(masks_ar)

        if cfg.TEST.MASK_AUG.ASPECT_RATIO_H_FLIP:
            masks_ar_hf = im_detect_mask_aspect_ratio(
                model, im, aspect_ratio, boxes, hflip=True
            )
            masks_ts.append(masks_ar_hf)

    # Combine the predicted soft masks
    if cfg.TEST.MASK_AUG.HEUR == 'SOFT_AVG':
        masks_c = np.mean(masks_ts, axis=0)
    elif cfg.TEST.MASK_AUG.HEUR == 'SOFT_MAX':
        masks_c = np.amax(masks_ts, axis=0)
    elif cfg.TEST.MASK_AUG.HEUR == 'LOGIT_AVG':

        def logit(y):
            return -1.0 * np.log((1.0 - y) / np.maximum(y, 1e-20))

        logit_masks = [logit(y) for y in masks_ts]
        logit_masks = np.mean(logit_masks, axis=0)
        masks_c = 1.0 / (1.0 + np.exp(-logit_masks))
    else:
        raise NotImplementedError(
            'Heuristic {} not supported'.format(cfg.TEST.MASK_AUG.HEUR)
        )

    return masks_c

def box_utils_box_voting(top_dets, all_dets, thresh, scoring_method='ID', beta=1.0):
    """Apply bounding-box voting to refine `top_dets` by voting with `all_dets`.
    See: https://arxiv.org/abs/1505.01749. Optional score averaging (not in the
    referenced  paper) can be applied by setting `scoring_method` appropriately.
    """
    # top_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    # all_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    top_dets_out = top_dets.copy()
    top_boxes = top_dets[:, :4]
    all_boxes = all_dets[:, :4]
    all_scores = all_dets[:, 4]
    top_to_all_overlaps = bbox_overlaps(top_boxes, all_boxes)
    for k in range(top_dets_out.shape[0]):
        inds_to_vote = np.where(top_to_all_overlaps[k] >= thresh)[0]
        boxes_to_vote = all_boxes[inds_to_vote, :]
        ws = all_scores[inds_to_vote]
        top_dets_out[k, :4] = np.average(boxes_to_vote, axis=0, weights=ws)
        if scoring_method == 'ID':
            # Identity, nothing to do
            pass
        elif scoring_method == 'TEMP_AVG':
            # Average probabilities (considered as P(detected class) vs.
            # P(not the detected class)) after smoothing with a temperature
            # hyperparameter.
            P = np.vstack((ws, 1.0 - ws))
            P_max = np.max(P, axis=0)
            X = np.log(P / P_max)
            X_exp = np.exp(X / beta)
            P_temp = X_exp / np.sum(X_exp, axis=0)
            P_avg = P_temp[0].mean()
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'AVG':
            # Combine new probs from overlapping boxes
            top_dets_out[k, 4] = ws.mean()
        elif scoring_method == 'IOU_AVG':
            P = ws
            ws = top_to_all_overlaps[k, inds_to_vote]
            P_avg = np.average(P, weights=ws)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'GENERALIZED_AVG':
            P_avg = np.mean(ws**beta)**(1.0 / beta)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'QUASI_SUM':
            top_dets_out[k, 4] = ws.sum() / float(len(ws))**beta
        else:
            raise NotImplementedError(
                'Unknown scoring method {}'.format(scoring_method)
            )

    return top_dets_out

def box_utils_soft_nms(
    dets, sigma=0.5, overlap_thresh=0.3, score_thresh=0.001, method='linear'
):
    """Apply the soft NMS algorithm from https://arxiv.org/abs/1704.04503."""
    if dets.shape[0] == 0:
        return dets, []

    methods = {'hard': 0, 'linear': 1, 'gaussian': 2}
    assert method in methods, 'Unknown soft_nms method: {}'.format(method)

    dets, keep = cython_nms.soft_nms(
        np.ascontiguousarray(dets, dtype=np.float32),
        np.float32(sigma),
        np.float32(overlap_thresh),
        np.float32(score_thresh),
        np.uint8(methods[method])
    )
    return dets, keep

def box_results_with_nms_and_limit(scores, boxes):
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4:(j + 1) * 4]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(
            np.float32, copy=False
        )
        if cfg.TEST.SOFT_NMS.ENABLED:
            nms_dets, _ = box_utils_soft_nms(
                dets_j,
                sigma=cfg.TEST.SOFT_NMS.SIGMA,
                overlap_thresh=cfg.TEST.NMS,
                score_thresh=0.0001,
                method=cfg.TEST.SOFT_NMS.METHOD
            )
        else:
            keep = box_utils_nms(dets_j, cfg.TEST.NMS)
            nms_dets = dets_j[keep, :]
        # Refine the post-NMS boxes using bounding-box voting
        if cfg.TEST.BBOX_VOTE.ENABLED:
            nms_dets = box_utils_box_voting(
                nms_dets,
                dets_j,
                cfg.TEST.BBOX_VOTE.VOTE_TH,
                scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
            )
        cls_boxes[j] = nms_dets

    # Limit to max_per_image detections **over all classes**
    if cfg.TEST.DETECTIONS_PER_IM > 0:
        image_scores = np.hstack(
            [cls_boxes[j][:, -1] for j in range(1, num_classes)]
        )
        if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
            image_thresh = np.sort(image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
            for j in range(1, num_classes):
                keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes

def _add_multilevel_rois_for_test(blobs, name):
    """Distributes a set of RoIs across FPN pyramid levels by creating new level
    specific RoI blobs.

    Arguments:
        blobs (dict): dictionary of blobs
        name (str): a key in 'blobs' identifying the source RoI blob

    Returns:
        [by ref] blobs (dict): new keys named by `name + 'fpn' + level`
            are added to dict each with a value that's an R_level x 5 ndarray of
            RoIs (see _get_rois_blob for format)
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    lvls = fpn_map_rois_to_fpn_levels(blobs[name][:, 1:5], lvl_min, lvl_max)
    fpn_add_multilevel_roi_blobs(
        blobs, name, blobs[name], lvls, lvl_min, lvl_max
    )

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (ndarray): image pyramid levels used by each projected RoI
    """
    rois = im_rois.astype(np.float, copy=False) * scales
    levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)
    return rois, levels

def _get_rois_blob(im_rois, im_scale):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid with columns
            [level, x1, y1, x2, y2]
    """
    rois, levels = _project_im_rois(im_rois, im_scale)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _get_blobs(im, rois, target_scale, target_max_size):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale, blobs['im_info'] = \
        blob_utils_get_image_blob(im, target_scale, target_max_size)
    if rois is not None:
        blobs['rois'] = _get_rois_blob(rois, im_scale)
    return blobs, im_scale

def im_detect_bbox(model, im, target_scale, target_max_size, boxes=None):
    """Bounding box object detection for an image with given box proposals.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals in 0-indexed
            [x1, y1, x2, y2] format, or None if using RPN

    Returns:
        scores (ndarray): R x K array of object class scores for K classes
            (K includes background as object category 0)
        boxes (ndarray): R x 4*K array of predicted bounding boxes
        im_scales (list): list of image scales used in the input blob (as
            returned by _get_blobs and for use with im_detect_mask, etc.)
    """
    inputs, im_scale = _get_blobs(im, boxes, target_scale, target_max_size)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(inputs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(
            hashes, return_index=True, return_inverse=True
        )
        inputs['rois'] = inputs['rois'][index, :]
        boxes = boxes[index, :]

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS and not cfg.MODEL.FASTER_RCNN:
        _add_multilevel_rois_for_test(inputs, 'rois')

    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v)
    workspace.RunNet(model.net.Proto().name)

    # Read out blobs
    if cfg.MODEL.FASTER_RCNN:
        rois = workspace.FetchBlob(core.ScopedName('rois'))
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scale

    # Softmax class probabilities
    scores = workspace.FetchBlob(core.ScopedName('cls_prob')).squeeze()
    # In case there is 1 proposal
    scores = scores.reshape([-1, scores.shape[-1]])

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = workspace.FetchBlob(core.ScopedName('bbox_pred')).squeeze()
        # In case there is 1 proposal
        box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            # Remove predictions for bg class (compat with MSRA code)
            box_deltas = box_deltas[:, -4:]
        pred_boxes = box_utils_bbox_transform(
            boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS
        )
        pred_boxes = box_utils_clip_tiled_boxes(pred_boxes, im.shape)
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            pred_boxes = np.tile(pred_boxes, (1, scores.shape[1]))
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes, im_scale

def box_utils_aspect_ratio(boxes, aspect_ratio):
    """Perform width-relative aspect ratio transformation."""
    boxes_ar = boxes.copy()
    boxes_ar[:, 0::4] = aspect_ratio * boxes[:, 0::4]
    boxes_ar[:, 2::4] = aspect_ratio * boxes[:, 2::4]
    return boxes_ar

def image_utils_aspect_ratio_rel(im, aspect_ratio):
    """Performs width-relative aspect ratio transformation."""
    im_h, im_w = im.shape[:2]
    im_ar_w = int(round(aspect_ratio * im_w))
    im_ar = cv2.resize(im, dsize=(im_ar_w, im_h))
    return im_ar

def im_detect_bbox_aspect_ratio(
    model, im, aspect_ratio, box_proposals=None, hflip=False
):
    """Computes bbox detections at the given width-relative aspect ratio.
    Returns predictions in the original image space.
    """
    # Compute predictions on the transformed image
    im_ar = image_utils_aspect_ratio_rel(im, aspect_ratio)

    if not cfg.MODEL.FASTER_RCNN:
        box_proposals_ar = box_utils_aspect_ratio(box_proposals, aspect_ratio)
    else:
        box_proposals_ar = None

    if hflip:
        scores_ar, boxes_ar, _ = im_detect_bbox_hflip(
            model,
            im_ar,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            box_proposals=box_proposals_ar
        )
    else:
        scores_ar, boxes_ar, _ = im_detect_bbox(
            model,
            im_ar,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            boxes=box_proposals_ar
        )

    # Invert the detected boxes
    boxes_inv = box_utils_aspect_ratio(boxes_ar, 1.0 / aspect_ratio)

    return scores_ar, boxes_inv

def im_detect_bbox_scale(
    model, im, target_scale, target_max_size, box_proposals=None, hflip=False
):
    """Computes bbox detections at the given scale.
    Returns predictions in the original image space.
    """
    if hflip:
        scores_scl, boxes_scl, _ = im_detect_bbox_hflip(
            model, im, target_scale, target_max_size, box_proposals=box_proposals
        )
    else:
        scores_scl, boxes_scl, _ = im_detect_bbox(
            model, im, target_scale, target_max_size, boxes=box_proposals
        )
    return scores_scl, boxes_scl

def box_utils_flip_boxes(boxes, im_width):
    """Flip boxes horizontally."""
    boxes_flipped = boxes.copy()
    boxes_flipped[:, 0::4] = im_width - boxes[:, 2::4] - 1
    boxes_flipped[:, 2::4] = im_width - boxes[:, 0::4] - 1
    return boxes_flipped

def im_detect_bbox_hflip(
    model, im, target_scale, target_max_size, box_proposals=None
):
    """Performs bbox detection on the horizontally flipped image.
    Function signature is the same as for im_detect_bbox.
    """
    # Compute predictions on the flipped image
    im_hf = im[:, ::-1, :]
    im_width = im.shape[1]

    if not cfg.MODEL.FASTER_RCNN:
        box_proposals_hf = box_utils_flip_boxes(box_proposals, im_width)
    else:
        box_proposals_hf = None

    scores_hf, boxes_hf, im_scale = im_detect_bbox(
        model, im_hf, target_scale, target_max_size, boxes=box_proposals_hf
    )

    # Invert the detections computed on the flipped image
    boxes_inv = box_utils_flip_boxes(boxes_hf, im_width)

    return scores_hf, boxes_inv, im_scale

def im_detect_bbox_aug(model, im, box_proposals=None):
    """Performs bbox detection with test-time augmentations.
    Function signature is the same as for im_detect_bbox.
    """
    assert not cfg.TEST.BBOX_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'
    assert not cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION' or \
        cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION', \
        'Coord heuristic must be union whenever score heuristic is union'
    assert not cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION' or \
        cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', \
        'Score heuristic must be union whenever coord heuristic is union'
    assert not cfg.MODEL.FASTER_RCNN or \
        cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', \
        'Union heuristic must be used to combine Faster RCNN predictions'

    # Collect detections computed under different transformations
    scores_ts = []
    boxes_ts = []

    def add_preds_t(scores_t, boxes_t):
        scores_ts.append(scores_t)
        boxes_ts.append(boxes_t)

    # Perform detection on the horizontally flipped image
    if cfg.TEST.BBOX_AUG.H_FLIP:
        scores_hf, boxes_hf, _ = im_detect_bbox_hflip(
            model,
            im,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            box_proposals=box_proposals
        )
        add_preds_t(scores_hf, boxes_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.BBOX_AUG.SCALES:
        max_size = cfg.TEST.BBOX_AUG.MAX_SIZE
        scores_scl, boxes_scl = im_detect_bbox_scale(
            model, im, scale, max_size, box_proposals
        )
        add_preds_t(scores_scl, boxes_scl)

        if cfg.TEST.BBOX_AUG.SCALE_H_FLIP:
            scores_scl_hf, boxes_scl_hf = im_detect_bbox_scale(
                model, im, scale, max_size, box_proposals, hflip=True
            )
            add_preds_t(scores_scl_hf, boxes_scl_hf)

    # Perform detection at different aspect ratios
    for aspect_ratio in cfg.TEST.BBOX_AUG.ASPECT_RATIOS:
        scores_ar, boxes_ar = im_detect_bbox_aspect_ratio(
            model, im, aspect_ratio, box_proposals
        )
        add_preds_t(scores_ar, boxes_ar)

        if cfg.TEST.BBOX_AUG.ASPECT_RATIO_H_FLIP:
            scores_ar_hf, boxes_ar_hf = im_detect_bbox_aspect_ratio(
                model, im, aspect_ratio, box_proposals, hflip=True
            )
            add_preds_t(scores_ar_hf, boxes_ar_hf)

    # Compute detections for the original image (identity transform) last to
    # ensure that the Caffe2 workspace is populated with blobs corresponding
    # to the original image on return (postcondition of im_detect_bbox)
    scores_i, boxes_i, im_scale_i = im_detect_bbox(
        model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=box_proposals
    )
    add_preds_t(scores_i, boxes_i)

    # Combine the predicted scores
    if cfg.TEST.BBOX_AUG.SCORE_HEUR == 'ID':
        scores_c = scores_i
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'AVG':
        scores_c = np.mean(scores_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION':
        scores_c = np.vstack(scores_ts)
    else:
        raise NotImplementedError(
            'Score heur {} not supported'.format(cfg.TEST.BBOX_AUG.SCORE_HEUR)
        )

    # Combine the predicted boxes
    if cfg.TEST.BBOX_AUG.COORD_HEUR == 'ID':
        boxes_c = boxes_i
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'AVG':
        boxes_c = np.mean(boxes_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION':
        boxes_c = np.vstack(boxes_ts)
    else:
        raise NotImplementedError(
            'Coord heur {} not supported'.format(cfg.TEST.BBOX_AUG.COORD_HEUR)
        )

    return scores_c, boxes_c, im_scale_i

def im_list_to_blob(ims):
    """Convert a list of images into a network input. Assumes images were
    prepared using prep_im_for_blob or equivalent: i.e.
      - BGR channel order
      - pixel means subtracted
      - resized to the desired input size
      - float32 numpy ndarray format
    Output is a 4D HCHW tensor of the images concatenated along axis 0 with
    shape.
    """
    if not isinstance(ims, list):
        ims = [ims]
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    # Pad the image so they can be divisible by a stride
    if cfg.FPN.FPN_ON:
        stride = float(cfg.FPN.COARSEST_STRIDE)
        max_shape[0] = int(np.ceil(max_shape[0] / stride) * stride)
        max_shape[1] = int(np.ceil(max_shape[1] / stride) * stride)

    num_images = len(ims)
    blob = np.zeros(
        (num_images, max_shape[0], max_shape[1], 3), dtype=np.float32
    )
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Prepare an image for use as a network input blob. Specially:
      - Subtract per-channel pixel mean
      - Convert to float32
      - Rescale to each of the specified target size (capped at max_size)
    Returns a list of transformed images, one for each target size. Also returns
    the scale factors that were used to compute each returned image.
    """
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than max_size
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(
        im,
        None,
        None,
        fx=im_scale,
        fy=im_scale,
        interpolation=cv2.INTER_LINEAR
    )
    return im, im_scale

def blob_utils_get_image_blob(im, target_scale, target_max_size):
    """Convert an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale (float): image scale (target size) / (original size)
        im_info (ndarray)
    """
    processed_im, im_scale = prep_im_for_blob(
        im, cfg.PIXEL_MEANS, target_scale, target_max_size
    )
    blob = im_list_to_blob(processed_im)
    # NOTE: this height and width may be larger than actual scaled input image
    # due to the FPN.COARSEST_STRIDE related padding in im_list_to_blob. We are
    # maintaining this behavior for now to make existing results exactly
    # reproducible (in practice using the true input image height and width
    # yields nearly the same results, but they are sometimes slightly different
    # because predictions near the edge of the image will be pruned more
    # aggressively).
    height, width = blob.shape[2], blob.shape[3]
    im_info = np.hstack((height, width, im_scale))[np.newaxis, :]
    return blob, im_scale, im_info.astype(np.float32)

def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1)
        )
    )
    return anchors

def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _generate_anchors(base_size, scales, aspect_ratios):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )
    return anchors

def generate_anchors(
    stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """
    return _generate_anchors(
        stride,
        np.array(sizes, dtype=np.float) / stride,
        np.array(aspect_ratios, dtype=np.float)
    )

def _create_cell_anchors():
    """
    Generate all types of anchors for all fpn levels/scales/aspect ratios.
    This function is called only once at the beginning of inference.
    """
    k_max, k_min = cfg.FPN.RPN_MAX_LEVEL, cfg.FPN.RPN_MIN_LEVEL
    scales_per_octave = cfg.RETINANET.SCALES_PER_OCTAVE
    aspect_ratios = cfg.RETINANET.ASPECT_RATIOS
    anchor_scale = cfg.RETINANET.ANCHOR_SCALE
    A = scales_per_octave * len(aspect_ratios)
    anchors = {}
    for lvl in range(k_min, k_max + 1):
        # create cell anchors array
        stride = 2. ** lvl
        cell_anchors = np.zeros((A, 4))
        a = 0
        for octave in range(scales_per_octave):
            octave_scale = 2 ** (octave / float(scales_per_octave))
            for aspect in aspect_ratios:
                anchor_sizes = (stride * octave_scale * anchor_scale, )
                anchor_aspect_ratios = (aspect, )
                cell_anchors[a, :] = generate_anchors(
                    stride=stride, sizes=anchor_sizes,
                    aspect_ratios=anchor_aspect_ratios)
                a += 1
        anchors[lvl] = cell_anchors
    return anchors

def test_retinanet_im_detect_bbox(model, im, timers=None):
    """Generate RetinaNet detections on a single image."""
    if timers is None:
        timers = defaultdict(Timer)
    # Although anchors are input independent and could be precomputed,
    # recomputing them per image only brings a small overhead
    anchors = _create_cell_anchors()
    timers['im_detect_bbox'].tic()
    k_max, k_min = cfg.FPN.RPN_MAX_LEVEL, cfg.FPN.RPN_MIN_LEVEL
    A = cfg.RETINANET.SCALES_PER_OCTAVE * len(cfg.RETINANET.ASPECT_RATIOS)
    inputs = {}
    inputs['data'], im_scale, inputs['im_info'] = \
        blob_utils_get_image_blob(im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)
    cls_probs, box_preds = [], []
    for lvl in range(k_min, k_max + 1):
        suffix = 'fpn{}'.format(lvl)
        cls_probs.append(core.ScopedName('retnet_cls_prob_{}'.format(suffix)))
        box_preds.append(core.ScopedName('retnet_bbox_pred_{}'.format(suffix)))
    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v.astype(np.float32, copy=False))

    workspace.RunNet(model.net.Proto().name)
    cls_probs = workspace.FetchBlobs(cls_probs)
    box_preds = workspace.FetchBlobs(box_preds)

    # here the boxes_all are [x0, y0, x1, y1, score]
    boxes_all = defaultdict(list)

    cnt = 0
    for lvl in range(k_min, k_max + 1):
        # create cell anchors array
        stride = 2. ** lvl
        cell_anchors = anchors[lvl]

        # fetch per level probability
        cls_prob = cls_probs[cnt]
        box_pred = box_preds[cnt]
        cls_prob = cls_prob.reshape((
            cls_prob.shape[0], A, int(cls_prob.shape[1] / A),
            cls_prob.shape[2], cls_prob.shape[3]))
        box_pred = box_pred.reshape((
            box_pred.shape[0], A, 4, box_pred.shape[2], box_pred.shape[3]))
        cnt += 1

        if cfg.RETINANET.SOFTMAX:
            cls_prob = cls_prob[:, :, 1::, :, :]

        cls_prob_ravel = cls_prob.ravel()
        # In some cases [especially for very small img sizes], it's possible that
        # candidate_ind is empty if we impose threshold 0.05 at all levels. This
        # will lead to errors since no detections are found for this image. Hence,
        # for lvl 7 which has small spatial resolution, we take the threshold 0.0
        th = cfg.RETINANET.INFERENCE_TH if lvl < k_max else 0.0
        candidate_inds = np.where(cls_prob_ravel > th)[0]
        if (len(candidate_inds) == 0):
            continue

        pre_nms_topn = min(cfg.RETINANET.PRE_NMS_TOP_N, len(candidate_inds))
        inds = np.argpartition(
            cls_prob_ravel[candidate_inds], -pre_nms_topn)[-pre_nms_topn:]
        inds = candidate_inds[inds]

        inds_5d = np.array(np.unravel_index(inds, cls_prob.shape)).transpose()
        classes = inds_5d[:, 2]
        anchor_ids, y, x = inds_5d[:, 1], inds_5d[:, 3], inds_5d[:, 4]
        scores = cls_prob[:, anchor_ids, classes, y, x]

        boxes = np.column_stack((x, y, x, y)).astype(dtype=np.float32)
        boxes *= stride
        boxes += cell_anchors[anchor_ids, :]

        if not cfg.RETINANET.CLASS_SPECIFIC_BBOX:
            box_deltas = box_pred[0, anchor_ids, :, y, x]
        else:
            box_cls_inds = classes * 4
            box_deltas = np.vstack(
                [box_pred[0, ind:ind + 4, yi, xi]
                 for ind, yi, xi in zip(box_cls_inds, y, x)]
            )
        pred_boxes = (
            box_utils_bbox_transform(boxes, box_deltas)
            if cfg.TEST.BBOX_REG else boxes)
        pred_boxes /= im_scale
        pred_boxes = box_utils_clip_tiled_boxes(pred_boxes, im.shape)
        box_scores = np.zeros((pred_boxes.shape[0], 5))
        box_scores[:, 0:4] = pred_boxes
        box_scores[:, 4] = scores

        for cls in range(1, cfg.MODEL.NUM_CLASSES):
            inds = np.where(classes == cls - 1)[0]
            if len(inds) > 0:
                boxes_all[cls].extend(box_scores[inds, :])
    timers['im_detect_bbox'].toc()

    # Combine predictions across all levels and retain the top scoring by class
    timers['misc_bbox'].tic()
    detections = []
    for cls, boxes in boxes_all.items():
        cls_dets = np.vstack(boxes).astype(dtype=np.float32)
        # do class specific nms here
        keep = box_utils_nms(cls_dets, cfg.TEST.NMS)
        cls_dets = cls_dets[keep, :]
        out = np.zeros((len(keep), 6))
        out[:, 0:5] = cls_dets
        out[:, 5].fill(cls)
        detections.append(out)

    # detections (N, 6) format:
    #   detections[:, :4] - boxes
    #   detections[:, 4] - scores
    #   detections[:, 5] - classes
    detections = np.vstack(detections)
    # sort all again
    inds = np.argsort(-detections[:, 4])
    detections = detections[inds[0:cfg.TEST.DETECTIONS_PER_IM], :]

    # Convert the detections to image cls_ format (see core/test_engine.py)
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_boxes = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    for c in range(1, num_classes):
        inds = np.where(detections[:, 5] == c)[0]
        cls_boxes[c] = detections[inds, :5]
    timers['misc_bbox'].toc()

    return cls_boxes

def infer_engine_im_detect_all(model, im, box_proposals, timers=None):
    if timers is None:
        timers = defaultdict(Timer)

    # Handle RetinaNet testing separately for now
    if cfg.RETINANET.RETINANET_ON:
        cls_boxes = test_retinanet_im_detect_bbox(model, im, timers)
        return cls_boxes, None, None

    timers['im_detect_bbox'].tic()
    if cfg.TEST.BBOX_AUG.ENABLED:
        scores, boxes, im_scale = im_detect_bbox_aug(model, im, box_proposals)
    else:
        scores, boxes, im_scale = im_detect_bbox(
            model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=box_proposals
        )
    timers['im_detect_bbox'].toc()

    # score and boxes are from the whole image after score thresholding and nms
    # (they are not separated by class)
    # cls_boxes boxes and scores are separated by class and in the format used
    # for evaluating results
    timers['misc_bbox'].tic()
    scores, boxes, cls_boxes = box_results_with_nms_and_limit(scores, boxes)
    timers['misc_bbox'].toc()

    if cfg.MODEL.MASK_ON and boxes.shape[0] > 0:
        timers['im_detect_mask'].tic()
        if cfg.TEST.MASK_AUG.ENABLED:
            masks = im_detect_mask_aug(model, im, boxes)
        else:
            masks = im_detect_mask(model, im_scale, boxes)
        timers['im_detect_mask'].toc()

        timers['misc_mask'].tic()
        cls_segms = segm_results(
            cls_boxes, masks, boxes, im.shape[0], im.shape[1]
        )
        timers['misc_mask'].toc()
    else:
        cls_segms = None

    if cfg.MODEL.KEYPOINTS_ON and boxes.shape[0] > 0:
        timers['im_detect_keypoints'].tic()
        if cfg.TEST.KPS_AUG.ENABLED:
            heatmaps = im_detect_keypoints_aug(model, im, boxes)
        else:
            heatmaps = im_detect_keypoints(model, im_scale, boxes)
        timers['im_detect_keypoints'].toc()

        timers['misc_keypoints'].tic()
        cls_keyps = keypoint_results(cls_boxes, heatmaps, boxes)
        timers['misc_keypoints'].toc()
    else:
        cls_keyps = None

    if cfg.MODEL.BODY_UV_ON and boxes.shape[0] > 0:
        timers['im_detect_body_uv'].tic()
        cls_bodys = im_detect_body_uv(model, im_scale, boxes)
        timers['im_detect_body_uv'].toc()
        
    else:
        cls_bodys = None

    return cls_boxes, cls_segms, cls_keyps, cls_bodys

def model_builder_add_inference_inputs(model):
    """Create network input blobs used for inference."""

    def create_input_blobs_for_net(net_def):
        for op in net_def.op:
            for blob_in in op.input:
                if not workspace.HasBlob(blob_in):
                    workspace.CreateBlob(blob_in)

    create_input_blobs_for_net(model.net.Proto())
    if cfg.MODEL.MASK_ON:
        create_input_blobs_for_net(model.mask_net.Proto())
    if cfg.MODEL.KEYPOINTS_ON:
        create_input_blobs_for_net(model.keypoint_net.Proto())
    if cfg.MODEL.BODY_UV_ON:
        create_input_blobs_for_net(model.body_uv_net.Proto())

def add_single_gpu_param_update_ops(model, gpu_id):
    # Learning rate of 0 is a dummy value to be set properly at the
    # start of training
    lr = model.param_init_net.ConstantFill(
        [], 'lr', shape=[1], value=0.0
    )
    one = model.param_init_net.ConstantFill(
        [], 'one', shape=[1], value=1.0
    )
    wd = model.param_init_net.ConstantFill(
        [], 'wd', shape=[1], value=cfg.SOLVER.WEIGHT_DECAY
    )
    # weight decay of GroupNorm's parameters
    wd_gn = model.param_init_net.ConstantFill(
        [], 'wd_gn', shape=[1], value=cfg.SOLVER.WEIGHT_DECAY_GN
    )
    for param in model.TrainableParams(gpu_id=gpu_id):
        logger.debug('param ' + str(param) + ' will be updated')
        param_grad = model.param_to_grad[param]
        # Initialize momentum vector
        param_momentum = model.param_init_net.ConstantFill(
            [param], param + '_momentum', value=0.0
        )
        if param in model.biases:
            # Special treatment for biases (mainly to match historical impl.
            # details):
            # (1) Do not apply weight decay
            # (2) Use a 2x higher learning rate
            model.Scale(param_grad, param_grad, scale=2.0)
        elif param in model.gn_params:
            # Special treatment for GroupNorm's parameters
            model.WeightedSum([param_grad, one, param, wd_gn], param_grad)
        elif cfg.SOLVER.WEIGHT_DECAY > 0:
            # Apply weight decay to non-bias weights
            model.WeightedSum([param_grad, one, param, wd], param_grad)
        # Update param_grad and param_momentum in place
        model.net.MomentumSGDUpdate(
            [param_grad, param_momentum, lr, param],
            [param_grad, param_momentum, param],
            momentum=cfg.SOLVER.MOMENTUM
        )

def _add_allreduce_graph(model):
    """Construct the graph that performs Allreduce on the gradients."""
    # Need to all-reduce the per-GPU gradients if training with more than 1 GPU
    all_params = model.TrainableParams()
    assert len(all_params) % cfg.NUM_GPUS == 0
    # The model parameters are replicated on each GPU, get the number
    # distinct parameter blobs (i.e., the number of parameter blobs on
    # each GPU)
    params_per_gpu = int(len(all_params) / cfg.NUM_GPUS)
    with c2_utils_CudaScope(0):
        # Iterate over distinct parameter blobs
        for i in range(params_per_gpu):
            # Gradients from all GPUs for this parameter blob
            gradients = [
                model.param_to_grad[p] for p in all_params[i::params_per_gpu]
            ]
            if len(gradients) > 0:
                if cfg.USE_NCCL:
                    model.net.NCCLAllreduce(gradients, gradients)
                else:
                    muji.Allreduce(model.net, gradients, reduced_affix='')

def _build_forward_graph(model, single_gpu_build_func):
    """Construct the forward graph on each GPU."""
    all_loss_gradients = {}  # Will include loss gradients from all GPUs
    # Build the model on each GPU with correct name and device scoping
    for gpu_id in range(cfg.NUM_GPUS):
        with c2_utils_NamedCudaScope(gpu_id):
            all_loss_gradients.update(single_gpu_build_func(model))
    return all_loss_gradients

def optim_build_data_parallel_model(model, single_gpu_build_func):
    """Build a data parallel model given a function that builds the model on a
    single GPU.
    """
    if model.only_build_forward_pass:
        single_gpu_build_func(model)
    elif model.train:
        all_loss_gradients = _build_forward_graph(model, single_gpu_build_func)
        # Add backward pass on all GPUs
        model.AddGradientOperators(all_loss_gradients)
        if cfg.NUM_GPUS > 1:
            _add_allreduce_graph(model)
        for gpu_id in range(cfg.NUM_GPUS):
            # After allreduce, all GPUs perform SGD updates on their identical
            # params and gradients in parallel
            with c2_utils_NamedCudaScope(gpu_id):
                add_single_gpu_param_update_ops(model, gpu_id)
    else:
        # Test-time network operates on single GPU
        # Test-time parallelism is implemented through multiprocessing
        with c2_utils_NamedCudaScope(model.target_gpu_id):
            single_gpu_build_func(model)

def body_uv_rcnn_heads_add_body_uv_losses(model, pref=''):

    ## Reshape for GT blobs.
    model.net.Reshape( ['body_uv_X_points'], ['X_points_reshaped'+pref, 'X_points_shape'+pref],  shape=( -1 ,1 ) )
    model.net.Reshape( ['body_uv_Y_points'], ['Y_points_reshaped'+pref, 'Y_points_shape'+pref],  shape=( -1 ,1 ) )
    model.net.Reshape( ['body_uv_I_points'], ['I_points_reshaped'+pref, 'I_points_shape'+pref],  shape=( -1 ,1 ) )
    model.net.Reshape( ['body_uv_Ind_points'], ['Ind_points_reshaped'+pref, 'Ind_points_shape'+pref],  shape=( -1 ,1 ) )
    ## Concat Ind,x,y to get Coordinates blob.
    model.net.Concat( ['Ind_points_reshaped'+pref,'X_points_reshaped'+pref, \
                       'Y_points_reshaped'+pref],['Coordinates'+pref,'Coordinate_Shapes'+pref ], axis = 1 )
    ##
    ### Now reshape UV blobs, such that they are 1x1x(196*NumSamples)xNUM_PATCHES 
    ## U blob to
    ##
    model.net.Reshape(['body_uv_U_points'], \
                      ['U_points_reshaped'+pref, 'U_points_old_shape'+pref],\
                      shape=(-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1,196))
    model.net.Transpose(['U_points_reshaped'+pref] ,['U_points_reshaped_transpose'+pref],axes=(0,2,1) )
    model.net.Reshape(['U_points_reshaped_transpose'+pref], \
                      ['U_points'+pref, 'U_points_old_shape2'+pref], \
                      shape=(1,1,-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1))
    ## V blob
    ##
    model.net.Reshape(['body_uv_V_points'], \
                      ['V_points_reshaped'+pref, 'V_points_old_shape'+pref],\
                      shape=(-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1,196))
    model.net.Transpose(['V_points_reshaped'+pref] ,['V_points_reshaped_transpose'+pref],axes=(0,2,1) )
    model.net.Reshape(['V_points_reshaped_transpose'+pref], \
                      ['V_points'+pref, 'V_points_old_shape2'+pref], \
                      shape=(1,1,-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1))
    ###
    ## UV weights blob
    ##
    model.net.Reshape(['body_uv_point_weights'], \
                      ['Uv_point_weights_reshaped'+pref, 'Uv_point_weights_old_shape'+pref],\
                      shape=(-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1,196))
    model.net.Transpose(['Uv_point_weights_reshaped'+pref] ,['Uv_point_weights_reshaped_transpose'+pref],axes=(0,2,1) )
    model.net.Reshape(['Uv_point_weights_reshaped_transpose'+pref], \
                      ['Uv_point_weights'+pref, 'Uv_point_weights_old_shape2'+pref], \
                      shape=(1,1,-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1))

    #####################
    ###  Pool IUV for points via bilinear interpolation.
    model.PoolPointsInterp(['U_estimated','Coordinates'+pref], ['interp_U'+pref])
    model.PoolPointsInterp(['V_estimated','Coordinates'+pref], ['interp_V'+pref])
    model.PoolPointsInterp(['Index_UV'+pref,'Coordinates'+pref], ['interp_Index_UV'+pref])

    ## Reshape interpolated UV coordinates to apply the loss.
    
    model.net.Reshape(['interp_U'+pref], \
                      ['interp_U_reshaped'+pref, 'interp_U_shape'+pref],\
                      shape=(1, 1, -1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1))
    
    model.net.Reshape(['interp_V'+pref], \
                      ['interp_V_reshaped'+pref, 'interp_V_shape'+pref],\
                      shape=(1, 1, -1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1))
    ###

    ### Do the actual labels here !!!!
    model.net.Reshape( ['body_uv_ann_labels'],    \
                      ['body_uv_ann_labels_reshaped'   +pref, 'body_uv_ann_labels_old_shape'+pref], \
                      shape=(-1, cfg.BODY_UV_RCNN.HEATMAP_SIZE , cfg.BODY_UV_RCNN.HEATMAP_SIZE))
    
    model.net.Reshape( ['body_uv_ann_weights'],   \
                      ['body_uv_ann_weights_reshaped'   +pref, 'body_uv_ann_weights_old_shape'+pref], \
                      shape=( -1 , cfg.BODY_UV_RCNN.HEATMAP_SIZE , cfg.BODY_UV_RCNN.HEATMAP_SIZE))
    ###
    model.net.Cast( ['I_points_reshaped'+pref], ['I_points_reshaped_int'+pref], to=core.DataType.INT32)
    ### Now add the actual losses 
    ## The mask segmentation loss (dense)
    probs_seg_AnnIndex, loss_seg_AnnIndex = model.net.SpatialSoftmaxWithLoss( \
                          ['AnnIndex'+pref, 'body_uv_ann_labels_reshaped'+pref,'body_uv_ann_weights_reshaped'+pref],\
                          ['probs_seg_AnnIndex'+pref,'loss_seg_AnnIndex'+pref], \
                           scale=cfg.BODY_UV_RCNN.INDEX_WEIGHTS / cfg.NUM_GPUS)
    ## Point Patch Index Loss.
    probs_IndexUVPoints, loss_IndexUVPoints = model.net.SoftmaxWithLoss(\
                          ['interp_Index_UV'+pref,'I_points_reshaped_int'+pref],\
                          ['probs_IndexUVPoints'+pref,'loss_IndexUVPoints'+pref], \
                          scale=cfg.BODY_UV_RCNN.PART_WEIGHTS / cfg.NUM_GPUS, spatial=0)
    ## U and V point losses.
    loss_Upoints = model.net.SmoothL1Loss( \
                          ['interp_U_reshaped'+pref, 'U_points'+pref, \
                               'Uv_point_weights'+pref, 'Uv_point_weights'+pref], \
                          'loss_Upoints'+pref, \
                            scale=cfg.BODY_UV_RCNN.POINT_REGRESSION_WEIGHTS  / cfg.NUM_GPUS)
    
    loss_Vpoints = model.net.SmoothL1Loss( \
                          ['interp_V_reshaped'+pref, 'V_points'+pref, \
                               'Uv_point_weights'+pref, 'Uv_point_weights'+pref], \
                          'loss_Vpoints'+pref, scale=cfg.BODY_UV_RCNN.POINT_REGRESSION_WEIGHTS / cfg.NUM_GPUS)
    ## Add the losses.
    loss_gradients = blob_utils_get_loss_gradients(model, \
                       [ loss_Upoints, loss_Vpoints, loss_seg_AnnIndex, loss_IndexUVPoints])
    model.losses = list(set(model.losses + \
                       ['loss_Upoints'+pref , 'loss_Vpoints'+pref , \
                        'loss_seg_AnnIndex'+pref ,'loss_IndexUVPoints'+pref]))

    return loss_gradients

def body_uv_rcnn_heads_add_body_uv_outputs(model, blob_in, dim, pref=''):
    ####
    model.ConvTranspose(blob_in, 'AnnIndex_lowres'+pref, dim, 15,cfg.BODY_UV_RCNN.DECONV_KERNEL, pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1), stride=2, weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}), bias_init=('ConstantFill', {'value': 0.}))    
    ####
    model.ConvTranspose(blob_in, 'Index_UV_lowres'+pref, dim, cfg.BODY_UV_RCNN.NUM_PATCHES+1,cfg.BODY_UV_RCNN.DECONV_KERNEL, pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1), stride=2, weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}), bias_init=('ConstantFill', {'value': 0.}))    
    ####
    model.ConvTranspose(
        blob_in, 'U_lowres'+pref, dim, (cfg.BODY_UV_RCNN.NUM_PATCHES+1),
        cfg.BODY_UV_RCNN.DECONV_KERNEL,
        pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1),
        stride=2,
        weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
        bias_init=('ConstantFill', {'value': 0.}))
    #####
    model.ConvTranspose(
            blob_in, 'V_lowres'+pref, dim, cfg.BODY_UV_RCNN.NUM_PATCHES+1,
            cfg.BODY_UV_RCNN.DECONV_KERNEL,
            pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1),
            stride=2,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.}))
    ####
    blob_Ann_Index = model.BilinearInterpolation('AnnIndex_lowres'+pref, 'AnnIndex'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    blob_Index = model.BilinearInterpolation('Index_UV_lowres'+pref, 'Index_UV'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    blob_U = model.BilinearInterpolation('U_lowres'+pref, 'U_estimated'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    blob_V = model.BilinearInterpolation('V_lowres'+pref, 'V_estimated'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    ###
    return blob_U,blob_V,blob_Index,blob_Ann_Index

def _add_roi_body_uv_head(
    model, add_roi_body_uv_head_func, blob_in, dim_in, spatial_scale_in
):
    """Add a body UV prediction head to the model."""
    # Capture model graph before adding the mask head
    bbox_net = copy.deepcopy(model.net.Proto())
    # Add the body UV head
    blob_body_uv_head, dim_body_uv_head = add_roi_body_uv_head_func(
        model, blob_in, dim_in, spatial_scale_in
    )
    # Add the body UV output
    blobs_body_uv = body_uv_rcnn_heads_add_body_uv_outputs(
        model, blob_body_uv_head, dim_body_uv_head
    )

    if not model.train:  # == inference
        # Inference uses a cascade of box predictions, then body uv predictions
        # This requires separate nets for box and body uv prediction.
        # So we extract the keypoint prediction net, store it as its own
        # network, then restore model.net to be the bbox-only network
        model.body_uv_net, body_uv_blob_out = c2_utils_SuffixNet(
            'body_uv_net', model.net, len(bbox_net.op), blobs_body_uv
        )
        model.net._net = bbox_net
        loss_gradients = None
    else:
        loss_gradients = body_uv_rcnn_heads_add_body_uv_losses(model)
    return loss_gradients

def keypoint_rcnn_heads_add_keypoint_losses(model):
    """Add Mask R-CNN keypoint specific losses."""
    # Reshape input from (N, K, H, W) to (NK, HW)
    model.net.Reshape(
        ['kps_score'], ['kps_score_reshaped', '_kps_score_old_shape'],
        shape=(-1, cfg.KRCNN.HEATMAP_SIZE * cfg.KRCNN.HEATMAP_SIZE)
    )
    # Softmax across **space** (woahh....space!)
    # Note: this is not what is commonly called "spatial softmax"
    # (i.e., softmax applied along the channel dimension at each spatial
    # location); This is softmax applied over a set of spatial locations (i.e.,
    # each spatial location is a "class").
    kps_prob, loss_kps = model.net.SoftmaxWithLoss(
        ['kps_score_reshaped', 'keypoint_locations_int32', 'keypoint_weights'],
        ['kps_prob', 'loss_kps'],
        scale=cfg.KRCNN.LOSS_WEIGHT / cfg.NUM_GPUS,
        spatial=0
    )
    if not cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS:
        # Discussion: the softmax loss above will average the loss by the sum of
        # keypoint_weights, i.e. the total number of visible keypoints. Since
        # the number of visible keypoints can vary significantly between
        # minibatches, this has the effect of up-weighting the importance of
        # minibatches with few visible keypoints. (Imagine the extreme case of
        # only one visible keypoint versus N: in the case of N, each one
        # contributes 1/N to the gradient compared to the single keypoint
        # determining the gradient direction). Instead, we can normalize the
        # loss by the total number of keypoints, if it were the case that all
        # keypoints were visible in a full minibatch. (Returning to the example,
        # this means that the one visible keypoint contributes as much as each
        # of the N keypoints.)
        model.StopGradient(
            'keypoint_loss_normalizer', 'keypoint_loss_normalizer'
        )
        loss_kps = model.net.Mul(
            ['loss_kps', 'keypoint_loss_normalizer'], 'loss_kps_normalized'
        )
    loss_gradients = blob_utils_get_loss_gradients(model, [loss_kps])
    model.AddLosses(loss_kps)
    return loss_gradients

def keypoint_rcnn_heads_add_keypoint_outputs(model, blob_in, dim):
    """Add Mask R-CNN keypoint specific outputs: keypoint heatmaps."""
    # NxKxHxW
    upsample_heatmap = (cfg.KRCNN.UP_SCALE > 1)

    if cfg.KRCNN.USE_DECONV:
        # Apply ConvTranspose to the feature representation; results in 2x
        # upsampling
        blob_in = model.ConvTranspose(
            blob_in,
            'kps_deconv',
            dim,
            cfg.KRCNN.DECONV_DIM,
            kernel=cfg.KRCNN.DECONV_KERNEL,
            pad=int(cfg.KRCNN.DECONV_KERNEL / 2 - 1),
            stride=2,
            weight_init=gauss_fill(0.01),
            bias_init=const_fill(0.0)
        )
        model.Relu('kps_deconv', 'kps_deconv')
        dim = cfg.KRCNN.DECONV_DIM

    if upsample_heatmap:
        blob_name = 'kps_score_lowres'
    else:
        blob_name = 'kps_score'

    if cfg.KRCNN.USE_DECONV_OUTPUT:
        # Use ConvTranspose to predict heatmaps; results in 2x upsampling
        blob_out = model.ConvTranspose(
            blob_in,
            blob_name,
            dim,
            cfg.KRCNN.NUM_KEYPOINTS,
            kernel=cfg.KRCNN.DECONV_KERNEL,
            pad=int(cfg.KRCNN.DECONV_KERNEL / 2 - 1),
            stride=2,
            weight_init=(cfg.KRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=const_fill(0.0)
        )
    else:
        # Use Conv to predict heatmaps; does no upsampling
        blob_out = model.Conv(
            blob_in,
            blob_name,
            dim,
            cfg.KRCNN.NUM_KEYPOINTS,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(cfg.KRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=const_fill(0.0)
        )

    if upsample_heatmap:
        # Increase heatmap output size via bilinear upsampling
        blob_out = model.BilinearInterpolation(
            blob_out, 'kps_score', cfg.KRCNN.NUM_KEYPOINTS,
            cfg.KRCNN.NUM_KEYPOINTS, cfg.KRCNN.UP_SCALE
        )

    return blob_out

def _add_roi_keypoint_head(
    model, add_roi_keypoint_head_func, blob_in, dim_in, spatial_scale_in
):
    """Add a keypoint prediction head to the model."""
    # Capture model graph before adding the mask head
    bbox_net = copy.deepcopy(model.net.Proto())
    # Add the keypoint head
    blob_keypoint_head, dim_keypoint_head = add_roi_keypoint_head_func(
        model, blob_in, dim_in, spatial_scale_in
    )
    # Add the keypoint output
    blob_keypoint = keypoint_rcnn_heads_add_keypoint_outputs(
        model, blob_keypoint_head, dim_keypoint_head
    )

    if not model.train:  # == inference
        # Inference uses a cascade of box predictions, then keypoint predictions
        # This requires separate nets for box and keypoint prediction.
        # So we extract the keypoint prediction net, store it as its own
        # network, then restore model.net to be the bbox-only network
        model.keypoint_net, keypoint_blob_out = c2_utils_SuffixNet(
            'keypoint_net', model.net, len(bbox_net.op), blob_keypoint
        )
        model.net._net = bbox_net
        loss_gradients = None
    else:
        loss_gradients = keypoint_rcnn_heads_add_keypoint_losses(model)
    return loss_gradients

def mask_rcnn_heads_add_mask_rcnn_losses(model, blob_mask):
    """Add Mask R-CNN specific losses."""
    loss_mask = model.net.SigmoidCrossEntropyLoss(
        [blob_mask, 'masks_int32'],
        'loss_mask',
        scale=model.GetLossScale() * cfg.MRCNN.WEIGHT_LOSS_MASK
    )
    loss_gradients = blob_utils_get_loss_gradients(model, [loss_mask])
    model.AddLosses('loss_mask')
    return loss_gradients

def BlobReferenceList(blob_ref_or_list):
    """Ensure that the argument is returned as a list of BlobReferences."""
    if isinstance(blob_ref_or_list, core.BlobReference):
        return [blob_ref_or_list]
    elif type(blob_ref_or_list) in (list, tuple):
        for b in blob_ref_or_list:
            assert isinstance(b, core.BlobReference)
        return blob_ref_or_list
    else:
        raise TypeError(
            'blob_ref_or_list must be a BlobReference or a list/tuple of '
            'BlobReferences'
        )

def c2_utils_SuffixNet(name, net, prefix_len, outputs):
    """Returns a new Net from the given Net (`net`) that includes only the ops
    after removing the first `prefix_len` number of ops. The new Net is thus a
    suffix of `net`. Blobs listed in `outputs` are registered as external output
    blobs.
    """
    outputs = BlobReferenceList(outputs)
    for output in outputs:
        assert net.BlobIsDefined(output)
    new_net = net.Clone(name)

    del new_net.Proto().op[:]
    del new_net.Proto().external_input[:]
    del new_net.Proto().external_output[:]

    # Add suffix ops
    new_net.Proto().op.extend(net.Proto().op[prefix_len:])
    # Add external input blobs
    # Treat any undefined blobs as external inputs
    input_names = [
        i for op in new_net.Proto().op for i in op.input
        if not new_net.BlobIsDefined(i)]
    new_net.Proto().external_input.extend(input_names)
    # Add external output blobs
    output_names = [str(o) for o in outputs]
    new_net.Proto().external_output.extend(output_names)
    return new_net, [new_net.GetBlobRef(o) for o in output_names]

def mask_rcnn_heads_add_mask_rcnn_outputs(model, blob_in, dim):
    """Add Mask R-CNN specific outputs: either mask logits or probs."""
    num_cls = cfg.MODEL.NUM_CLASSES if cfg.MRCNN.CLS_SPECIFIC_MASK else 1

    if cfg.MRCNN.USE_FC_OUTPUT:
        # Predict masks with a fully connected layer (ignore 'fcn' in the blob
        # name)
        blob_out = model.FC(
            blob_in,
            'mask_fcn_logits',
            dim,
            num_cls * cfg.MRCNN.RESOLUTION**2,
            weight_init=gauss_fill(0.001),
            bias_init=const_fill(0.0)
        )
    else:
        # Predict mask using Conv

        # Use GaussianFill for class-agnostic mask prediction; fills based on
        # fan-in can be too large in this case and cause divergence
        fill = (
            cfg.MRCNN.CONV_INIT
            if cfg.MRCNN.CLS_SPECIFIC_MASK else 'GaussianFill'
        )
        blob_out = model.Conv(
            blob_in,
            'mask_fcn_logits',
            dim,
            num_cls,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(fill, {'std': 0.001}),
            bias_init=const_fill(0.0)
        )

        if cfg.MRCNN.UPSAMPLE_RATIO > 1:
            blob_out = model.BilinearInterpolation(
                'mask_fcn_logits', 'mask_fcn_logits_up', num_cls, num_cls,
                cfg.MRCNN.UPSAMPLE_RATIO
            )

    if not model.train:  # == if test
        blob_out = model.net.Sigmoid(blob_out, 'mask_fcn_probs')

    return blob_out

def _add_roi_mask_head(
    model, add_roi_mask_head_func, blob_in, dim_in, spatial_scale_in
):
    """Add a mask prediction head to the model."""
    # Capture model graph before adding the mask head
    bbox_net = copy.deepcopy(model.net.Proto())
    # Add the mask head
    blob_mask_head, dim_mask_head = add_roi_mask_head_func(
        model, blob_in, dim_in, spatial_scale_in
    )
    # Add the mask output
    blob_mask = mask_rcnn_heads_add_mask_rcnn_outputs(
        model, blob_mask_head, dim_mask_head
    )

    if not model.train:  # == inference
        # Inference uses a cascade of box predictions, then mask predictions.
        # This requires separate nets for box and mask prediction.
        # So we extract the mask prediction net, store it as its own network,
        # then restore model.net to be the bbox-only network
        model.mask_net, blob_mask = c2_utils_SuffixNet(
            'mask_net', model.net, len(bbox_net.op), blob_mask
        )
        model.net._net = bbox_net
        loss_gradients = None
    else:
        loss_gradients = mask_rcnn_heads_add_mask_rcnn_losses(model, blob_mask)
    return loss_gradients

def fast_rcnn_heads_add_fast_rcnn_losses(model):
    """Add losses for RoI classification and bounding box regression."""
    cls_prob, loss_cls = model.net.SoftmaxWithLoss(
        ['cls_score', 'labels_int32'], ['cls_prob', 'loss_cls'],
        scale=model.GetLossScale()
    )
    loss_bbox = model.net.SmoothL1Loss(
        [
            'bbox_pred', 'bbox_targets', 'bbox_inside_weights',
            'bbox_outside_weights'
        ],
        'loss_bbox',
        scale=model.GetLossScale()
    )
    loss_gradients = blob_utils_get_loss_gradients(model, [loss_cls, loss_bbox])
    model.Accuracy(['cls_prob', 'labels_int32'], 'accuracy_cls')
    model.AddLosses(['loss_cls', 'loss_bbox'])
    model.AddMetrics('accuracy_cls')
    return loss_gradients

def fast_rcnn_heads_add_fast_rcnn_outputs(model, blob_in, dim):
    """Add RoI classification and bounding box regression output ops."""
    # Box classification layer
    model.FC(
        blob_in,
        'cls_score',
        dim,
        model.num_classes,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        model.Softmax('cls_score', 'cls_prob', engine='CUDNN')
    # Box regression layer
    num_bbox_reg_classes = (
        2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else model.num_classes
    )
    model.FC(
        blob_in,
        'bbox_pred',
        dim,
        num_bbox_reg_classes * 4,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )

def _add_fast_rcnn_head(
    model, add_roi_box_head_func, blob_in, dim_in, spatial_scale_in
):
    """Add a Fast R-CNN head to the model."""
    blob_frcn, dim_frcn = add_roi_box_head_func(
        model, blob_in, dim_in, spatial_scale_in
    )
    fast_rcnn_heads_add_fast_rcnn_outputs(model, blob_frcn, dim_frcn)
    if model.train:
        loss_gradients = fast_rcnn_heads_add_fast_rcnn_losses(model)
    else:
        loss_gradients = None
    return loss_gradients

def _narrow_to_fpn_roi_levels(blobs, spatial_scales):
    """Return only the blobs and spatial scales that will be used for RoI heads.
    Inputs `blobs` and `spatial_scales` may include extra blobs and scales that
    are used for RPN proposals, but not for RoI heads.
    """
    # Code only supports case when RPN and ROI min levels are the same
    assert cfg.FPN.RPN_MIN_LEVEL == cfg.FPN.ROI_MIN_LEVEL
    # RPN max level can be >= to ROI max level
    assert cfg.FPN.RPN_MAX_LEVEL >= cfg.FPN.ROI_MAX_LEVEL
    # FPN RPN max level might be > FPN ROI max level in which case we
    # need to discard some leading conv blobs (blobs are ordered from
    # max/coarsest level to min/finest level)
    num_roi_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1
    return blobs[-num_roi_levels:], spatial_scales[-num_roi_levels:]

def add_single_scale_rpn_losses(model):
    """Add losses for a single scale RPN model (i.e., no FPN)."""
    # Spatially narrow the full-sized RPN label arrays to match the feature map
    # shape
    model.net.SpatialNarrowAs(
        ['rpn_labels_int32_wide', 'rpn_cls_logits'], 'rpn_labels_int32'
    )
    for key in ('targets', 'inside_weights', 'outside_weights'):
        model.net.SpatialNarrowAs(
            ['rpn_bbox_' + key + '_wide', 'rpn_bbox_pred'], 'rpn_bbox_' + key
        )
    loss_rpn_cls = model.net.SigmoidCrossEntropyLoss(
        ['rpn_cls_logits', 'rpn_labels_int32'],
        'loss_rpn_cls',
        scale=model.GetLossScale()
    )
    loss_rpn_bbox = model.net.SmoothL1Loss(
        [
            'rpn_bbox_pred', 'rpn_bbox_targets', 'rpn_bbox_inside_weights',
            'rpn_bbox_outside_weights'
        ],
        'loss_rpn_bbox',
        beta=1. / 9.,
        scale=model.GetLossScale()
    )
    loss_gradients = blob_utils_get_loss_gradients(
        model, [loss_rpn_cls, loss_rpn_bbox]
    )
    model.AddLosses(['loss_rpn_cls', 'loss_rpn_bbox'])
    return loss_gradients

def add_single_scale_rpn_outputs(model, blob_in, dim_in, spatial_scale):
    """Add RPN outputs to a single scale model (i.e., no FPN)."""
    anchors = generate_anchors(
        stride=1. / spatial_scale,
        sizes=cfg.RPN.SIZES,
        aspect_ratios=cfg.RPN.ASPECT_RATIOS
    )
    num_anchors = anchors.shape[0]
    dim_out = dim_in
    # RPN hidden representation
    model.Conv(
        blob_in,
        'conv_rpn',
        dim_in,
        dim_out,
        kernel=3,
        pad=1,
        stride=1,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    model.Relu('conv_rpn', 'conv_rpn')
    # Proposal classification scores
    model.Conv(
        'conv_rpn',
        'rpn_cls_logits',
        dim_in,
        num_anchors,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    # Proposal bbox regression deltas
    model.Conv(
        'conv_rpn',
        'rpn_bbox_pred',
        dim_in,
        4 * num_anchors,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )

    if not model.train or cfg.MODEL.FASTER_RCNN:
        # Proposals are needed during:
        #  1) inference (== not model.train) for RPN only and Faster R-CNN
        #  OR
        #  2) training for Faster R-CNN
        # Otherwise (== training for RPN only), proposals are not needed
        model.net.Sigmoid('rpn_cls_logits', 'rpn_cls_probs')
        model.GenerateProposals(
            ['rpn_cls_probs', 'rpn_bbox_pred', 'im_info'],
            ['rpn_rois', 'rpn_roi_probs'],
            anchors=anchors,
            spatial_scale=spatial_scale
        )

    if cfg.MODEL.FASTER_RCNN:
        if model.train:
            # Add op that generates training labels for in-network RPN proposals
            model.GenerateProposalLabels(['rpn_rois', 'roidb', 'im_info'])
        else:
            # Alias rois to rpn_rois for inference
            model.net.Alias('rpn_rois', 'rois')

def blob_utils_get_loss_gradients(model, loss_blobs):
    """Generate a gradient of 1 for each loss specified in 'loss_blobs'"""
    loss_gradients = {}
    for b in loss_blobs:
        loss_grad = model.net.ConstantFill(b, [b + '_grad'], value=1.0)
        loss_gradients[str(b)] = str(loss_grad)
    return loss_gradients

def FPN_add_fpn_rpn_losses(model):
    """Add RPN on FPN specific losses."""
    loss_gradients = {}
    for lvl in range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1):
        slvl = str(lvl)
        # Spatially narrow the full-sized RPN label arrays to match the feature map
        # shape
        model.net.SpatialNarrowAs(
            ['rpn_labels_int32_wide_fpn' + slvl, 'rpn_cls_logits_fpn' + slvl],
            'rpn_labels_int32_fpn' + slvl
        )
        for key in ('targets', 'inside_weights', 'outside_weights'):
            model.net.SpatialNarrowAs(
                [
                    'rpn_bbox_' + key + '_wide_fpn' + slvl,
                    'rpn_bbox_pred_fpn' + slvl
                ],
                'rpn_bbox_' + key + '_fpn' + slvl
            )
        loss_rpn_cls_fpn = model.net.SigmoidCrossEntropyLoss(
            ['rpn_cls_logits_fpn' + slvl, 'rpn_labels_int32_fpn' + slvl],
            'loss_rpn_cls_fpn' + slvl,
            normalize=0,
            scale=(
                model.GetLossScale() / cfg.TRAIN.RPN_BATCH_SIZE_PER_IM /
                cfg.TRAIN.IMS_PER_BATCH
            )
        )
        # Normalization by (1) RPN_BATCH_SIZE_PER_IM and (2) IMS_PER_BATCH is
        # handled by (1) setting bbox outside weights and (2) SmoothL1Loss
        # normalizes by IMS_PER_BATCH
        loss_rpn_bbox_fpn = model.net.SmoothL1Loss(
            [
                'rpn_bbox_pred_fpn' + slvl, 'rpn_bbox_targets_fpn' + slvl,
                'rpn_bbox_inside_weights_fpn' + slvl,
                'rpn_bbox_outside_weights_fpn' + slvl
            ],
            'loss_rpn_bbox_fpn' + slvl,
            beta=1. / 9.,
            scale=model.GetLossScale(),
        )
        loss_gradients.update(
            blob_utils_get_loss_gradients(model, [loss_rpn_cls_fpn, loss_rpn_bbox_fpn])
        )
        model.AddLosses(['loss_rpn_cls_fpn' + slvl, 'loss_rpn_bbox_fpn' + slvl])
    return loss_gradients

def const_fill(value):
    """Constant fill helper to reduce verbosity."""
    return ('ConstantFill', {'value': value})

def gauss_fill(std):
    """Gaussian fill helper to reduce verbosity."""
    return ('GaussianFill', {'std': std})

def FPN_add_fpn_rpn_outputs(model, blobs_in, dim_in, spatial_scales):
    """Add RPN on FPN specific outputs."""
    num_anchors = len(cfg.FPN.RPN_ASPECT_RATIOS)
    dim_out = dim_in

    k_max = cfg.FPN.RPN_MAX_LEVEL  # coarsest level of pyramid
    k_min = cfg.FPN.RPN_MIN_LEVEL  # finest level of pyramid
    assert len(blobs_in) == k_max - k_min + 1
    for lvl in range(k_min, k_max + 1):
        bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
        sc = spatial_scales[k_max - lvl]  # in reversed order
        slvl = str(lvl)

        if lvl == k_min:
            # Create conv ops with randomly initialized weights and
            # zeroed biases for the first FPN level; these will be shared by
            # all other FPN levels
            # RPN hidden representation
            conv_rpn_fpn = model.Conv(
                bl_in,
                'conv_rpn_fpn' + slvl,
                dim_in,
                dim_out,
                kernel=3,
                pad=1,
                stride=1,
                weight_init=gauss_fill(0.01),
                bias_init=const_fill(0.0)
            )
            model.Relu(conv_rpn_fpn, conv_rpn_fpn)
            # Proposal classification scores
            rpn_cls_logits_fpn = model.Conv(
                conv_rpn_fpn,
                'rpn_cls_logits_fpn' + slvl,
                dim_in,
                num_anchors,
                kernel=1,
                pad=0,
                stride=1,
                weight_init=gauss_fill(0.01),
                bias_init=const_fill(0.0)
            )
            # Proposal bbox regression deltas
            rpn_bbox_pred_fpn = model.Conv(
                conv_rpn_fpn,
                'rpn_bbox_pred_fpn' + slvl,
                dim_in,
                4 * num_anchors,
                kernel=1,
                pad=0,
                stride=1,
                weight_init=gauss_fill(0.01),
                bias_init=const_fill(0.0)
            )
        else:
            # Share weights and biases
            sk_min = str(k_min)
            # RPN hidden representation
            conv_rpn_fpn = model.ConvShared(
                bl_in,
                'conv_rpn_fpn' + slvl,
                dim_in,
                dim_out,
                kernel=3,
                pad=1,
                stride=1,
                weight='conv_rpn_fpn' + sk_min + '_w',
                bias='conv_rpn_fpn' + sk_min + '_b'
            )
            model.Relu(conv_rpn_fpn, conv_rpn_fpn)
            # Proposal classification scores
            rpn_cls_logits_fpn = model.ConvShared(
                conv_rpn_fpn,
                'rpn_cls_logits_fpn' + slvl,
                dim_in,
                num_anchors,
                kernel=1,
                pad=0,
                stride=1,
                weight='rpn_cls_logits_fpn' + sk_min + '_w',
                bias='rpn_cls_logits_fpn' + sk_min + '_b'
            )
            # Proposal bbox regression deltas
            rpn_bbox_pred_fpn = model.ConvShared(
                conv_rpn_fpn,
                'rpn_bbox_pred_fpn' + slvl,
                dim_in,
                4 * num_anchors,
                kernel=1,
                pad=0,
                stride=1,
                weight='rpn_bbox_pred_fpn' + sk_min + '_w',
                bias='rpn_bbox_pred_fpn' + sk_min + '_b'
            )

        if not model.train or cfg.MODEL.FASTER_RCNN:
            # Proposals are needed during:
            #  1) inference (== not model.train) for RPN only and Faster R-CNN
            #  OR
            #  2) training for Faster R-CNN
            # Otherwise (== training for RPN only), proposals are not needed
            lvl_anchors = generate_anchors(
                stride=2.**lvl,
                sizes=(cfg.FPN.RPN_ANCHOR_START_SIZE * 2.**(lvl - k_min), ),
                aspect_ratios=cfg.FPN.RPN_ASPECT_RATIOS
            )
            rpn_cls_probs_fpn = model.net.Sigmoid(
                rpn_cls_logits_fpn, 'rpn_cls_probs_fpn' + slvl
            )
            model.GenerateProposals(
                [rpn_cls_probs_fpn, rpn_bbox_pred_fpn, 'im_info'],
                ['rpn_rois_fpn' + slvl, 'rpn_roi_probs_fpn' + slvl],
                anchors=lvl_anchors,
                spatial_scale=sc
            )

def rpn_heads_add_generic_rpn_outputs(model, blob_in, dim_in, spatial_scale_in):
    """Add RPN outputs (objectness classification and bounding box regression)
    to an RPN model. Abstracts away the use of FPN.
    """
    loss_gradients = None
    if cfg.FPN.FPN_ON:
        # Delegate to the FPN module
        FPN_add_fpn_rpn_outputs(model, blob_in, dim_in, spatial_scale_in)
        if cfg.MODEL.FASTER_RCNN:
            # CollectAndDistributeFpnRpnProposals also labels proposals when in
            # training mode
            model.CollectAndDistributeFpnRpnProposals()
        if model.train:
            loss_gradients = FPN_add_fpn_rpn_losses(model)
    else:
        # Not using FPN, add RPN to a single scale
        add_single_scale_rpn_outputs(model, blob_in, dim_in, spatial_scale_in)
        if model.train:
            loss_gradients = add_single_scale_rpn_losses(model)
    return loss_gradients

def c2_utils_BlobReferenceList(blob_ref_or_list):
    """Ensure that the argument is returned as a list of BlobReferences."""
    if isinstance(blob_ref_or_list, core.BlobReference):
        return [blob_ref_or_list]
    elif type(blob_ref_or_list) in (list, tuple):
        for b in blob_ref_or_list:
            assert isinstance(b, core.BlobReference)
        return blob_ref_or_list
    else:
        raise TypeError(
            'blob_ref_or_list must be a BlobReference or a list/tuple of '
            'BlobReferences'
        )

def build_generic_detection_model(
    model,
    add_conv_body_func,
    add_roi_box_head_func=None,
    add_roi_mask_head_func=None,
    add_roi_keypoint_head_func=None,
    add_roi_body_uv_head_func=None,
    freeze_conv_body=False
):
    def _single_gpu_build_func(model):
        """Build the model on a single GPU. Can be called in a loop over GPUs
        with name and device scoping to create a data parallel model.
        """
        # Add the conv body (called "backbone architecture" in papers)
        # E.g., ResNet-50, ResNet-50-FPN, ResNeXt-101-FPN, etc.
        blob_conv, dim_conv, spatial_scale_conv = add_conv_body_func(model)
        if freeze_conv_body:
            for b in c2_utils_BlobReferenceList(blob_conv):
                model.StopGradient(b, b)

        if not model.train:  # == inference
            # Create a net that can be used to execute the conv body on an image
            # (without also executing RPN or any other network heads)
            model.conv_body_net = model.net.Clone('conv_body_net')

        head_loss_gradients = {
            'rpn': None,
            'box': None,
            'mask': None,
            'keypoints': None,
            'body_uv' : None,
        }

        if cfg.RPN.RPN_ON:
            # Add the RPN head
            head_loss_gradients['rpn'] = rpn_heads_add_generic_rpn_outputs(
                model, blob_conv, dim_conv, spatial_scale_conv
            )

        if cfg.FPN.FPN_ON:
            # After adding the RPN head, restrict FPN blobs and scales to
            # those used in the RoI heads
            blob_conv, spatial_scale_conv = _narrow_to_fpn_roi_levels(
                blob_conv, spatial_scale_conv
            )

        if not cfg.MODEL.RPN_ONLY:
            # Add the Fast R-CNN head
            head_loss_gradients['box'] = _add_fast_rcnn_head(
                model, add_roi_box_head_func, blob_conv, dim_conv,
                spatial_scale_conv
            )

        if cfg.MODEL.MASK_ON:
            # Add the mask head
            head_loss_gradients['mask'] = _add_roi_mask_head(
                model, add_roi_mask_head_func, blob_conv, dim_conv,
                spatial_scale_conv
            )

        if cfg.MODEL.KEYPOINTS_ON:
            # Add the keypoint head
            head_loss_gradients['keypoint'] = _add_roi_keypoint_head(
                model, add_roi_keypoint_head_func, blob_conv, dim_conv,
                spatial_scale_conv
            )

        if cfg.MODEL.BODY_UV_ON:
            # Add the body UV head
            head_loss_gradients['body_uv'] = _add_roi_body_uv_head(
                model, add_roi_body_uv_head_func, blob_conv, dim_conv,
                spatial_scale_conv
            )

        if model.train:
            loss_gradients = {}
            for lg in head_loss_gradients.values():
                if lg is not None:
                    loss_gradients.update(lg)
            return loss_gradients
        else:
            return None

    optim_build_data_parallel_model(model, _single_gpu_build_func)
    return model

def get_group_gn(dim):
    """
    get number of groups used by GroupNorm, based on number of channels
    """
    dim_per_gp = cfg.GROUP_NORM.DIM_PER_GP
    num_groups = cfg.GROUP_NORM.NUM_GROUPS

    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0
        group_gn = num_groups
    return group_gn

def add_topdown_lateral_module(
    model, fpn_top, fpn_lateral, fpn_bottom, dim_top, dim_lateral
):
    """Add a top-down lateral module."""
    # Lateral 1x1 conv
    if cfg.FPN.USE_GN:
        # use GroupNorm
        lat = model.ConvGN(
            fpn_lateral,
            fpn_bottom + '_lateral',
            dim_in=dim_lateral,
            dim_out=dim_top,
            group_gn=get_group_gn(dim_top),
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(
                const_fill(0.0) if cfg.FPN.ZERO_INIT_LATERAL
                else ('XavierFill', {})),
            bias_init=const_fill(0.0)
        )
    else:
        lat = model.Conv(
            fpn_lateral,
            fpn_bottom + '_lateral',
            dim_in=dim_lateral,
            dim_out=dim_top,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(
                const_fill(0.0)
                if cfg.FPN.ZERO_INIT_LATERAL else ('XavierFill', {})
            ),
            bias_init=const_fill(0.0)
        )
    # Top-down 2x upsampling
    td = model.net.UpsampleNearest(fpn_top, fpn_bottom + '_topdown', scale=2)
    # Sum lateral and top-down
    model.net.Sum([lat, td], fpn_bottom)

def get_min_max_levels():
    """The min and max FPN levels required for supporting RPN and/or RoI
    transform operations on multiple FPN levels.
    """
    min_level = LOWEST_BACKBONE_LVL
    max_level = HIGHEST_BACKBONE_LVL
    if cfg.FPN.MULTILEVEL_RPN and not cfg.FPN.MULTILEVEL_ROIS:
        max_level = cfg.FPN.RPN_MAX_LEVEL
        min_level = cfg.FPN.RPN_MIN_LEVEL
    if not cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
        max_level = cfg.FPN.ROI_MAX_LEVEL
        min_level = cfg.FPN.ROI_MIN_LEVEL
    if cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
        max_level = max(cfg.FPN.RPN_MAX_LEVEL, cfg.FPN.ROI_MAX_LEVEL)
        min_level = min(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.ROI_MIN_LEVEL)
    return min_level, max_level

def add_fpn(model, fpn_level_info):
    """Add FPN connections based on the model described in the FPN paper."""
    # FPN levels are built starting from the highest/coarest level of the
    # backbone (usually "conv5"). First we build down, recursively constructing
    # lower/finer resolution FPN levels. Then we build up, constructing levels
    # that are even higher/coarser than the starting level.
    fpn_dim = cfg.FPN.DIM
    min_level, max_level = get_min_max_levels()
    # Count the number of backbone stages that we will generate FPN levels for
    # starting from the coarest backbone stage (usually the "conv5"-like level)
    # E.g., if the backbone level info defines stages 4 stages: "conv5",
    # "conv4", ... "conv2" and min_level=2, then we end up with 4 - (2 - 2) = 4
    # backbone stages to add FPN to.
    num_backbone_stages = (
        len(fpn_level_info.blobs) - (min_level - LOWEST_BACKBONE_LVL)
    )

    lateral_input_blobs = fpn_level_info.blobs[:num_backbone_stages]
    output_blobs = [
        'fpn_inner_{}'.format(s)
        for s in fpn_level_info.blobs[:num_backbone_stages]
    ]
    fpn_dim_lateral = fpn_level_info.dims
    xavier_fill = ('XavierFill', {})

    # For the coarsest backbone level: 1x1 conv only seeds recursion
    if cfg.FPN.USE_GN:
        # use GroupNorm
        c = model.ConvGN(
            lateral_input_blobs[0],
            output_blobs[0],  # note: this is a prefix
            dim_in=fpn_dim_lateral[0],
            dim_out=fpn_dim,
            group_gn=get_group_gn(fpn_dim),
            kernel=1,
            pad=0,
            stride=1,
            weight_init=xavier_fill,
            bias_init=const_fill(0.0)
        )
        output_blobs[0] = c  # rename it
    else:
        model.Conv(
            lateral_input_blobs[0],
            output_blobs[0],
            dim_in=fpn_dim_lateral[0],
            dim_out=fpn_dim,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=xavier_fill,
            bias_init=const_fill(0.0)
        )

    #
    # Step 1: recursively build down starting from the coarsest backbone level
    #

    # For other levels add top-down and lateral connections
    for i in range(num_backbone_stages - 1):
        add_topdown_lateral_module(
            model,
            output_blobs[i],             # top-down blob
            lateral_input_blobs[i + 1],  # lateral blob
            output_blobs[i + 1],         # next output blob
            fpn_dim,                     # output dimension
            fpn_dim_lateral[i + 1]       # lateral input dimension
        )

    # Post-hoc scale-specific 3x3 convs
    blobs_fpn = []
    spatial_scales = []
    for i in range(num_backbone_stages):
        if cfg.FPN.USE_GN:
            # use GroupNorm
            fpn_blob = model.ConvGN(
                output_blobs[i],
                'fpn_{}'.format(fpn_level_info.blobs[i]),
                dim_in=fpn_dim,
                dim_out=fpn_dim,
                group_gn=get_group_gn(fpn_dim),
                kernel=3,
                pad=1,
                stride=1,
                weight_init=xavier_fill,
                bias_init=const_fill(0.0)
            )
        else:
            fpn_blob = model.Conv(
                output_blobs[i],
                'fpn_{}'.format(fpn_level_info.blobs[i]),
                dim_in=fpn_dim,
                dim_out=fpn_dim,
                kernel=3,
                pad=1,
                stride=1,
                weight_init=xavier_fill,
                bias_init=const_fill(0.0)
            )
        blobs_fpn += [fpn_blob]
        spatial_scales += [fpn_level_info.spatial_scales[i]]

    #
    # Step 2: build up starting from the coarsest backbone level
    #

    # Check if we need the P6 feature map
    if not cfg.FPN.EXTRA_CONV_LEVELS and max_level == HIGHEST_BACKBONE_LVL + 1:
        # Original FPN P6 level implementation from our CVPR'17 FPN paper
        P6_blob_in = blobs_fpn[0]
        P6_name = P6_blob_in + '_subsampled_2x'
        # Use max pooling to simulate stride 2 subsampling
        P6_blob = model.MaxPool(P6_blob_in, P6_name, kernel=1, pad=0, stride=2)
        blobs_fpn.insert(0, P6_blob)
        spatial_scales.insert(0, spatial_scales[0] * 0.5)

    # Coarser FPN levels introduced for RetinaNet
    if cfg.FPN.EXTRA_CONV_LEVELS and max_level > HIGHEST_BACKBONE_LVL:
        fpn_blob = fpn_level_info.blobs[0]
        dim_in = fpn_level_info.dims[0]
        for i in range(HIGHEST_BACKBONE_LVL + 1, max_level + 1):
            fpn_blob_in = fpn_blob
            if i > HIGHEST_BACKBONE_LVL + 1:
                fpn_blob_in = model.Relu(fpn_blob, fpn_blob + '_relu')
            fpn_blob = model.Conv(
                fpn_blob_in,
                'fpn_' + str(i),
                dim_in=dim_in,
                dim_out=fpn_dim,
                kernel=3,
                pad=1,
                stride=2,
                weight_init=xavier_fill,
                bias_init=const_fill(0.0)
            )
            dim_in = fpn_dim
            blobs_fpn.insert(0, fpn_blob)
            spatial_scales.insert(0, spatial_scales[0] * 0.5)

    return blobs_fpn, fpn_dim, spatial_scales

def add_fpn_onto_conv_body(
    model, conv_body_func, fpn_level_info_func, P2only=False
):
    """Add the specified conv body to the model and then add FPN levels to it.
    """
    # Note: blobs_conv is in revsersed order: [fpn5, fpn4, fpn3, fpn2]
    # similarly for dims_conv: [2048, 1024, 512, 256]
    # similarly for spatial_scales_fpn: [1/32, 1/16, 1/8, 1/4]

    conv_body_func(model)
    blobs_fpn, dim_fpn, spatial_scales_fpn = add_fpn(
        model, fpn_level_info_func()
    )

    if P2only:
        # use only the finest level
        return blobs_fpn[-1], dim_fpn, spatial_scales_fpn[-1]
    else:
        # use all levels
        return blobs_fpn, dim_fpn, spatial_scales_fpn

def fpn_level_info_ResNet101_conv5():
    return FpnLevelInfo(
        blobs=('res5_2_sum', 'res4_22_sum', 'res3_3_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )

def add_residual_block(
    model,
    prefix,
    blob_in,
    dim_in,
    dim_out,
    dim_inner,
    dilation,
    stride_init=2,
    inplace_sum=False
):
    """Add a residual block to the model."""
    # prefix = res<stage>_<sub_stage>, e.g., res2_3

    # Max pooling is performed prior to the first stage (which is uniquely
    # distinguished by dim_in = 64), thus we keep stride = 1 for the first stage
    stride = stride_init if (
        dim_in != dim_out and dim_in != 64 and dilation == 1
    ) else 1

    # transformation blob
    tr = globals()[cfg.RESNETS.TRANS_FUNC](
        model,
        blob_in,
        dim_in,
        dim_out,
        stride,
        prefix,
        dim_inner,
        group=cfg.RESNETS.NUM_GROUPS,
        dilation=dilation
    )

    # sum -> ReLU
    # shortcut function: by default using bn; support gn
    add_shortcut = globals()[cfg.RESNETS.SHORTCUT_FUNC]
    sc = add_shortcut(model, prefix, blob_in, dim_in, dim_out, stride)
    if inplace_sum:
        s = model.net.Sum([tr, sc], tr)
    else:
        s = model.net.Sum([tr, sc], prefix + '_sum')

    return model.Relu(s, s)

def add_stage(
    model,
    prefix,
    blob_in,
    n,
    dim_in,
    dim_out,
    dim_inner,
    dilation,
    stride_init=2
):
    """Add a ResNet stage to the model by stacking n residual blocks."""
    # e.g., prefix = res2
    for i in range(n):
        blob_in = add_residual_block(
            model,
            '{}_{}'.format(prefix, i),
            blob_in,
            dim_in,
            dim_out,
            dim_inner,
            dilation,
            stride_init,
            # Not using inplace for the last block;
            # it may be fetched externally or used by FPN
            inplace_sum=i < n - 1
        )
        dim_in = dim_out
    return blob_in, dim_in

def add_ResNet_convX_body(model, block_counts, freeze_at=2):
    """Add a ResNet body from input data up through the res5 (aka conv5) stage.
    The final res5/conv5 stage may be optionally excluded (hence convX, where
    X = 4 or 5)."""
    assert freeze_at in [0, 2, 3, 4, 5]

    # add the stem (by default, conv1 and pool1 with bn; can support gn)
    p, dim_in = globals()[cfg.RESNETS.STEM_FUNC](model, 'data')

    dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
    (n1, n2, n3) = block_counts[:3]
    s, dim_in = add_stage(model, 'res2', p, n1, dim_in, 256, dim_bottleneck, 1)
    if freeze_at == 2:
        model.StopGradient(s, s)
    s, dim_in = add_stage(
        model, 'res3', s, n2, dim_in, 512, dim_bottleneck * 2, 1
    )
    if freeze_at == 3:
        model.StopGradient(s, s)
    s, dim_in = add_stage(
        model, 'res4', s, n3, dim_in, 1024, dim_bottleneck * 4, 1
    )
    if freeze_at == 4:
        model.StopGradient(s, s)
    if len(block_counts) == 4:
        n4 = block_counts[3]
        s, dim_in = add_stage(
            model, 'res5', s, n4, dim_in, 2048, dim_bottleneck * 8,
            cfg.RESNETS.RES5_DILATION
        )
        if freeze_at == 5:
            model.StopGradient(s, s)
        return s, dim_in, 1. / 32. * cfg.RESNETS.RES5_DILATION
    else:
        return s, dim_in, 1. / 16.

def ResNet_add_ResNet101_conv5_body(model):
    return add_ResNet_convX_body(model, (3, 4, 23, 3))

def FPN_add_fpn_ResNet101_conv5_body(model):
    return add_fpn_onto_conv_body(
        model, ResNet_add_ResNet101_conv5_body, fpn_level_info_ResNet101_conv5
    )

def bottleneck_transformation(
    model,
    blob_in,
    dim_in,
    dim_out,
    stride,
    prefix,
    dim_inner,
    dilation=1,
    group=1
):
    """Add a bottleneck transformation to the model."""
    # In original resnet, stride=2 is on 1x1.
    # In fb.torch resnet, stride=2 is on 3x3.
    (str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)

    # conv 1x1 -> BN -> ReLU
    cur = model.ConvAffine(
        blob_in,
        prefix + '_branch2a',
        dim_in,
        dim_inner,
        kernel=1,
        stride=str1x1,
        pad=0,
        inplace=True
    )
    cur = model.Relu(cur, cur)

    # conv 3x3 -> BN -> ReLU
    cur = model.ConvAffine(
        cur,
        prefix + '_branch2b',
        dim_inner,
        dim_inner,
        kernel=3,
        stride=str3x3,
        pad=1 * dilation,
        dilation=dilation,
        group=group,
        inplace=True
    )
    cur = model.Relu(cur, cur)

    # conv 1x1 -> BN (no ReLU)
    # NB: for now this AffineChannel op cannot be in-place due to a bug in C2
    # gradient computation for graphs like this
    cur = model.ConvAffine(
        cur,
        prefix + '_branch2c',
        dim_inner,
        dim_out,
        kernel=1,
        stride=1,
        pad=0,
        inplace=False
    )
    return cur

def basic_bn_shortcut(model, prefix, blob_in, dim_in, dim_out, stride):
    """ For a pre-trained network that used BN. An AffineChannel op replaces BN
    during fine-tuning.
    """

    if dim_in == dim_out:
        return blob_in

    c = model.Conv(
        blob_in,
        prefix + '_branch1',
        dim_in,
        dim_out,
        kernel=1,
        stride=stride,
        no_bias=1
    )
    return model.AffineChannel(c, prefix + '_branch1_bn', dim=dim_out)

def basic_bn_stem(model, data, **kwargs):
    """Add a basic ResNet stem. For a pre-trained network that used BN.
    An AffineChannel op replaces BN during fine-tuning.
    """

    dim = 64
    p = model.Conv(data, 'conv1', 3, dim, 7, pad=3, stride=2, no_bias=1)
    p = model.AffineChannel(p, 'res_conv1_bn', dim=dim, inplace=True)
    p = model.Relu(p, p)
    p = model.MaxPool(p, 'pool1', kernel=3, pad=1, stride=2)
    return p, dim

def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'detectron.modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: {}'.format(func_name))
        raise

def body_uv_rcnn_heads_add_roi_body_uv_head_v1convX(model, blob_in, dim_in, spatial_scale):
    """v1convX design: X * (conv)."""
    hidden_dim = cfg.BODY_UV_RCNN.CONV_HEAD_DIM
    kernel_size = cfg.BODY_UV_RCNN.CONV_HEAD_KERNEL
    pad_size = kernel_size // 2
    current = model.RoIFeatureTransform(
        blob_in,
        '_[body_uv]_roi_feat',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current,
            'body_conv_fcn' + str(i + 1),
            dim_in,
            hidden_dim,
            kernel_size,
            stride=1,
            pad=pad_size,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = hidden_dim

    return current, hidden_dim

def generalized_rcnn(model):
    """This model type handles:
      - Fast R-CNN
      - RPN only (not integrated with Fast R-CNN)
      - Faster R-CNN (stagewise training from NIPS paper)
      - Faster R-CNN (end-to-end joint training)
      - Mask R-CNN (stagewise training from NIPS paper)
      - Mask R-CNN (end-to-end joint training)
    """
    return build_generic_detection_model(
        model,
        eval(str(cfg.MODEL.CONV_BODY).replace(".","_")),
        add_roi_box_head_func=[None if cfg.FAST_RCNN.ROI_BOX_HEAD == "" else eval(str(cfg.FAST_RCNN.ROI_BOX_HEAD).replace(".","_"))][0],
        add_roi_mask_head_func=[None if cfg.MRCNN.ROI_MASK_HEAD == "" else eval(str(cfg.MRCNN.ROI_MASK_HEAD).replace(".","_"))][0],
        add_roi_keypoint_head_func=[None if cfg.KRCNN.ROI_KEYPOINTS_HEAD == "" else eval(str(cfg.KRCNN.ROI_KEYPOINTS_HEAD).replace(".","_"))][0],
        add_roi_body_uv_head_func=[None if cfg.BODY_UV_RCNN.ROI_HEAD == "" else eval(str(cfg.BODY_UV_RCNN.ROI_HEAD).replace(".","_"))][0],
        freeze_conv_body=cfg.TRAIN.FREEZE_CONV_BODY
    )

def fast_rcnn_heads_add_roi_2mlp_head(model, blob_in, dim_in, spatial_scale):
    """Add a ReLU MLP with two hidden layers."""
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    model.FC(roi_feat, 'fc6', dim_in * roi_size * roi_size, hidden_dim)
    model.Relu('fc6', 'fc6')
    model.FC('fc6', 'fc7', hidden_dim, hidden_dim)
    model.Relu('fc7', 'fc7')
    return 'fc7', hidden_dim

def model_builder_create(model_type_func, train=False, gpu_id=0):
    """Generic model creation function that dispatches to specific model
    building functions.

    By default, this function will generate a data parallel model configured to
    run on cfg.NUM_GPUS devices. However, you can restrict it to build a model
    targeted to a specific GPU by specifying gpu_id. This is used by
    optimizer.build_data_parallel_model() during test time.
    """
    model = DetectionModelHelper(
        name=model_type_func,
        train=train,
        num_classes=cfg.MODEL.NUM_CLASSES,
        init_params=train
    )
    model.only_build_forward_pass = False
    model.target_gpu_id = gpu_id
    return eval(str(model_type_func).replace(".","_"))(model)

def configure_bbox_reg_weights(model, saved_cfg):
    """Compatibility for old models trained with bounding box regression
    mean/std normalization (instead of fixed weights).
    """
    if 'MODEL' not in saved_cfg or 'BBOX_REG_WEIGHTS' not in saved_cfg.MODEL:
        logger.warning('Model from weights file was trained before config key '
                       'MODEL.BBOX_REG_WEIGHTS was added. Forcing '
                       'MODEL.BBOX_REG_WEIGHTS = (1., 1., 1., 1.) to ensure '
                       'correct **inference** behavior.')
        # Generally we don't allow modifying the config, but this is a one-off
        # hack to support some very old models
        is_immutable = cfg.is_immutable()
        cfg.immutable(False)
        cfg.MODEL.BBOX_REG_WEIGHTS = (1., 1., 1., 1.)
        cfg.immutable(is_immutable)
        #logger.info('New config:')
        #logger.info(pprint.pformat(cfg))
        assert not model.train, (
            'This model was trained with an older version of the code that '
            'used bounding box regression mean/std normalization. It can no '
            'longer be used for training. To upgrade it to a trainable model '
            'please use fb/compat/convert_bbox_reg_normalized_model.py.'
        )

def load_object(file_name):
    with open(file_name, 'rb') as f:
        # The default encoding used while unpickling is 7-bit (ASCII.) However,
        # the blobs are arbitrary 8-bit bytes which don't agree. The absolute
        # correct way to do this is to use `encoding="bytes"` and then interpret
        # the blob names either as ASCII, or better, as unicode utf-8. A
        # reasonable fix, however, is to treat it the encoding as 8-bit latin1
        # (which agrees with the first 256 characters of Unicode anyway.)
        if six.PY2:
            return pickle.load(f)
        else:
            return pickle.load(f, encoding='latin1')

def net_utils_initialize_gpu_from_weights_file(model, weights_file, gpu_id=0):
    """Initialize a network with ops on a specific GPU.

    If you use CUDA_VISIBLE_DEVICES to target specific GPUs, Caffe2 will
    automatically map logical GPU ids (starting from 0) to the physical GPUs
    specified in CUDA_VISIBLE_DEVICES.
    """
    #logger.info('Loading weights from: {}'.format(weights_file))
    ws_blobs = workspace.Blobs()
    src_blobs = load_object(weights_file)

    if 'cfg' in src_blobs:
        saved_cfg = load_cfg(src_blobs['cfg'])
        configure_bbox_reg_weights(model, saved_cfg)
    if 'blobs' in src_blobs:
        # Backwards compat--dictionary used to be only blobs, now they are
        # stored under the 'blobs' key
        src_blobs = src_blobs['blobs']
    # with open(weights_file, 'r') as f:
    #     src_blobs = pickle.load(f)
    # if 'cfg' in src_blobs:
    #     saved_cfg = load_cfg(src_blobs['cfg'])
    #     configure_bbox_reg_weights(model, saved_cfg)
    # if 'blobs' in src_blobs:
    #     # Backwards compat--dictionary used to be only blobs, now they are
    #     # stored under the 'blobs' key
    #     src_blobs = src_blobs['blobs']
    # Initialize weights on GPU gpu_id only
    unscoped_param_names = OrderedDict()  # Print these out in model order
    for blob in model.params:
        unscoped_param_names[c2_utils_UnscopeName(str(blob))] = True
    with c2_utils_NamedCudaScope(gpu_id):
        for unscoped_param_name in unscoped_param_names.keys():
            if (unscoped_param_name.find(']_') >= 0 and
                    unscoped_param_name not in src_blobs):
                # Special case for sharing initialization from a pretrained
                # model:
                # If a blob named '_[xyz]_foo' is in model.params and not in
                # the initialization blob dictionary, then load source blob
                # 'foo' into destination blob '_[xyz]_foo'
                src_name = unscoped_param_name[
                    unscoped_param_name.find(']_') + 2:]
            else:
                src_name = unscoped_param_name
            if src_name not in src_blobs:
                #logger.info('{:s} not found'.format(src_name))
                continue
            dst_name = core.ScopedName(unscoped_param_name)
            has_momentum = src_name + '_momentum' in src_blobs
            has_momentum_str = ' [+ momentum]' if has_momentum else ''
            logger.debug(
                '{:s}{:} loaded from weights file into {:s}: {}'.format(
                    src_name, has_momentum_str, dst_name, src_blobs[src_name]
                    .shape
                )
            )
            if dst_name in ws_blobs:
                # If the blob is already in the workspace, make sure that it
                # matches the shape of the loaded blob
                ws_blob = workspace.FetchBlob(dst_name)
                assert ws_blob.shape == src_blobs[src_name].shape, \
                    ('Workspace blob {} with shape {} does not match '
                     'weights file shape {}').format(
                        src_name,
                        ws_blob.shape,
                        src_blobs[src_name].shape)
            workspace.FeedBlob(
                dst_name,
                src_blobs[src_name].astype(np.float32, copy=False))
            if has_momentum:
                workspace.FeedBlob(
                    dst_name + '_momentum',
                    src_blobs[src_name + '_momentum'].astype(
                        np.float32, copy=False))

    # We preserve blobs that are in the weights file but not used by the current
    # model. We load these into CPU memory under the '__preserve__/' namescope.
    # These blobs will be stored when saving a model to a weights file. This
    # feature allows for alternating optimization of Faster R-CNN in which blobs
    # unused by one step can still be preserved forward and used to initialize
    # another step.
    for src_name in src_blobs.keys():
        if (src_name not in unscoped_param_names and
                not src_name.endswith('_momentum') and
                src_blobs[src_name] is not None):
            with c2_utils_CpuScope():
                workspace.FeedBlob(
                    '__preserve__/{:s}'.format(src_name), src_blobs[src_name])
                logger.debug(
                    '{:s} preserved in workspace (unused)'.format(src_name))

def infer_engine_initialize_model_from_cfg(weights_file, gpu_id=0):
    """Initialize a model from the global cfg. Loads test-time weights and
    creates the networks in the Caffe2 workspace.
    """
    model = model_builder_create(cfg.MODEL.TYPE, train=False, gpu_id=gpu_id)
    net_utils_initialize_gpu_from_weights_file(
        model, weights_file, gpu_id=gpu_id,
    )
    model_builder_add_inference_inputs(model)
    workspace.CreateNet(model.net)
    workspace.CreateNet(model.conv_body_net)
    if cfg.MODEL.MASK_ON:
        workspace.CreateNet(model.mask_net)
    if cfg.MODEL.KEYPOINTS_ON:
        workspace.CreateNet(model.keypoint_net)
    if cfg.MODEL.BODY_UV_ON:
        workspace.CreateNet(model.body_uv_net)
    return model

def setup_logging(name):
    FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
    # Manually clear root loggers to prevent any module that may have called
    # logging.basicConfig() from blocking our logging setup
    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(name)
    return logger

def cache_url(url_or_file, cache_dir):
    """Download the file specified by the URL to the cache_dir and return the
    path to the cached file. If the argument is not a URL, simply return it as
    is.
    """
    is_url = re.match(r'^(?:http)s?://', url_or_file, re.IGNORECASE) is not None

    if not is_url:
        return url_or_file
    #
    url = url_or_file
    #
    Len_filename  = len( url.split('/')[-1] )
    BASE_URL  =  url[0:-Len_filename-1]
    #
    cache_file_path = url.replace(BASE_URL, cache_dir)
    if os.path.exists(cache_file_path):
        #assert_cache_file_is_ok(url, cache_file_path)
        return cache_file_path

    cache_file_dir = os.path.dirname(cache_file_path)
    if not os.path.exists(cache_file_dir):
        os.makedirs(cache_file_dir)

    #logger.info('Downloading remote file {} to {}'.format(url, cache_file_path))
    download_url(url, cache_file_path)
    #assert_cache_file_is_ok(url, cache_file_path)
    return cache_file_path

def c2_utils_UnscopeName(possibly_scoped_name):
    """Remove any name scoping from a (possibly) scoped name. For example,
    convert the name 'gpu_0/foo' to 'foo'."""
    assert isinstance(possibly_scoped_name, string_types)
    return possibly_scoped_name[
        possibly_scoped_name.rfind(scope._NAMESCOPE_SEPARATOR) + 1:]

def _get_lr_change_ratio(cur_lr, new_lr):
    eps = 1e-10
    ratio = np.max(
        (new_lr / np.max((cur_lr, eps)), cur_lr / np.max((new_lr, eps)))
    )
    return ratio

def _filter_boxes(boxes, min_size, im_info):
    """Only keep boxes with both sides >= min_size and center within the image.
    """

    # Compute the width and height of the proposal boxes as measured in the original
    # image coordinate system (this is required to avoid "Negative Areas Found"
    # assertions in other parts of the code that measure).
    im_scale = im_info[2]
    ws_orig_scale = (boxes[:, 2] - boxes[:, 0]) / im_scale + 1
    hs_orig_scale = (boxes[:, 3] - boxes[:, 1]) / im_scale + 1
    # To avoid numerical issues we require the min_size to be at least 1 pixel in the
    # original image
    min_size = np.maximum(min_size, 1)
    # Proposal center is computed relative to the scaled input image
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    x_ctr = boxes[:, 0] + ws / 2.
    y_ctr = boxes[:, 1] + hs / 2.
    keep = np.where(
        (ws_orig_scale >= min_size)
        & (hs_orig_scale >= min_size)
        & (x_ctr < im_info[1])
        & (y_ctr < im_info[0])
    )[0]
    return keep

def box_utils_clip_tiled_boxes(boxes, im_shape):
    """Clip boxes to image boundaries. im_shape is [height, width] and boxes
    has shape (N, 4 * num_tiled_boxes)."""
    assert boxes.shape[1] % 4 == 0, \
        'boxes.shape[1] is {:d}, but must be divisible by 4.'.format(
        boxes.shape[1]
    )
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

def box_utils_nms(dets, thresh):
    """Apply classic DPM-style greedy NMS."""
    if dets.shape[0] == 0:
        return []
    return cython_nms.nms(dets, thresh)

def fast_rcnn_roi_data_get_fast_rcnn_blob_names(is_training=True):
    """Fast R-CNN blob names."""
    # rois blob: holds R regions of interest, each is a 5-tuple
    # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
    # rectangle (x1, y1, x2, y2)
    blob_names = ['rois']
    if is_training:
        # labels_int32 blob: R categorical labels in [0, ..., K] for K
        # foreground classes plus background
        blob_names += ['labels_int32']
    if is_training:
        # bbox_targets blob: R bounding-box regression targets with 4
        # targets per class
        blob_names += ['bbox_targets']
        # bbox_inside_weights blob: At most 4 targets per roi are active
        # this binary vector sepcifies the subset of active targets
        blob_names += ['bbox_inside_weights']
        blob_names += ['bbox_outside_weights']
    if is_training and cfg.MODEL.MASK_ON:
        # 'mask_rois': RoIs sampled for training the mask prediction branch.
        # Shape is (#masks, 5) in format (batch_idx, x1, y1, x2, y2).
        blob_names += ['mask_rois']
        # 'roi_has_mask': binary labels for the RoIs specified in 'rois'
        # indicating if each RoI has a mask or not. Note that in some cases
        # a *bg* RoI will have an all -1 (ignore) mask associated with it in
        # the case that no fg RoIs can be sampled. Shape is (batchsize).
        blob_names += ['roi_has_mask_int32']
        # 'masks_int32' holds binary masks for the RoIs specified in
        # 'mask_rois'. Shape is (#fg, M * M) where M is the ground truth
        # mask size.
        blob_names += ['masks_int32']
    if is_training and cfg.MODEL.KEYPOINTS_ON:
        # 'keypoint_rois': RoIs sampled for training the keypoint prediction
        # branch. Shape is (#instances, 5) in format (batch_idx, x1, y1, x2,
        # y2).
        blob_names += ['keypoint_rois']
        # 'keypoint_locations_int32': index of keypoint in
        # KRCNN.HEATMAP_SIZE**2 sized array. Shape is (#instances). Used in
        # SoftmaxWithLoss.
        blob_names += ['keypoint_locations_int32']
        # 'keypoint_weights': weight assigned to each target in
        # 'keypoint_locations_int32'. Shape is (#instances). Used in
        # SoftmaxWithLoss.
        blob_names += ['keypoint_weights']
        # 'keypoint_loss_normalizer': optional normalization factor to use if
        # cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS is False.
        blob_names += ['keypoint_loss_normalizer']
        ########################

    if is_training and cfg.MODEL.BODY_UV_ON:
        blob_names += ['body_uv_rois']
        blob_names += ['roi_has_body_uv_int32']
        #########
        # ###################################################
        blob_names += ['body_uv_ann_labels']
        blob_names += ['body_uv_ann_weights']
        # #################################################
        blob_names += ['body_uv_X_points']
        blob_names += ['body_uv_Y_points']
        blob_names += ['body_uv_Ind_points']
        blob_names += ['body_uv_I_points']
        blob_names += ['body_uv_U_points']
        blob_names += ['body_uv_V_points']
        blob_names += ['body_uv_point_weights']

    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        # Support for FPN multi-level rois without bbox reg isn't
        # implemented (... and may never be implemented)
        k_max = cfg.FPN.ROI_MAX_LEVEL
        k_min = cfg.FPN.ROI_MIN_LEVEL
        # Same format as rois blob, but one per FPN level
        for lvl in range(k_min, k_max + 1):
            blob_names += ['rois_fpn' + str(lvl)]
        blob_names += ['rois_idx_restore_int32']
        if is_training:
            if cfg.MODEL.MASK_ON:
                for lvl in range(k_min, k_max + 1):
                    blob_names += ['mask_rois_fpn' + str(lvl)]
                blob_names += ['mask_rois_idx_restore_int32']
            if cfg.MODEL.KEYPOINTS_ON:
                for lvl in range(k_min, k_max + 1):
                    blob_names += ['keypoint_rois_fpn' + str(lvl)]
                blob_names += ['keypoint_rois_idx_restore_int32']
            if cfg.MODEL.BODY_UV_ON:
                for lvl in range(k_min, k_max + 1):
                    blob_names += ['body_uv_rois_fpn' + str(lvl)]
                blob_names += ['body_uv_rois_idx_restore_int32']
    return blob_names

def blob_utils_py_op_copy_blob(blob_in, blob_out):
    """Copy a numpy ndarray given as blob_in into the Caffe2 CPUTensor blob
    given as blob_out. Supports float32 and int32 blob data types. This function
    is intended for copying numpy data into a Caffe2 blob in PythonOps.
    """
    # Some awkward voodoo required by Caffe2 to support int32 blobs
    needs_int32_init = False
    try:
        _ = blob.data.dtype  # noqa
    except Exception:
        needs_int32_init = blob_in.dtype == np.int32
    if needs_int32_init:
        # init can only take a list (failed on tuple)
        blob_out.init(list(blob_in.shape), caffe2_pb2.TensorProto.INT32)
    else:
        blob_out.reshape(blob_in.shape)
    blob_out.data[...] = blob_in

def keypoint_rcnn_roi_data_finalize_keypoint_minibatch(blobs, valid):
    """Finalize the minibatch after blobs for all minibatch images have been
    collated.
    """
    min_count = cfg.KRCNN.MIN_KEYPOINT_COUNT_FOR_VALID_MINIBATCH
    num_visible_keypoints = np.sum(blobs['keypoint_weights'])
    valid = (
        valid and len(blobs['keypoint_weights']) > 0 and
        num_visible_keypoints > min_count
    )
    # Normalizer to use if cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS is False.
    # See modeling.model_builder.add_keypoint_losses
    norm = num_visible_keypoints / (
        cfg.TRAIN.IMS_PER_BATCH * cfg.TRAIN.BATCH_SIZE_PER_IM *
        cfg.TRAIN.FG_FRACTION * cfg.KRCNN.NUM_KEYPOINTS
    )
    blobs['keypoint_loss_normalizer'] = np.array(norm, dtype=np.float32)
    return valid

def fpn_add_multilevel_roi_blobs(
    blobs, blob_prefix, rois, target_lvls, lvl_min, lvl_max
):
    """Add RoI blobs for multiple FPN levels to the blobs dict.

    blobs: a dict mapping from blob name to numpy ndarray
    blob_prefix: name prefix to use for the FPN blobs
    rois: the source rois as a 2D numpy array of shape (N, 5) where each row is
      an roi and the columns encode (batch_idx, x1, y1, x2, y2)
    target_lvls: numpy array of shape (N, ) indicating which FPN level each roi
      in rois should be assigned to
    lvl_min: the finest (highest resolution) FPN level (e.g., 2)
    lvl_max: the coarest (lowest resolution) FPN level (e.g., 6)
    """
    rois_idx_order = np.empty((0, ))
    rois_stacked = np.zeros((0, 5), dtype=np.float32)  # for assert
    for lvl in range(lvl_min, lvl_max + 1):
        idx_lvl = np.where(target_lvls == lvl)[0]
        blobs[blob_prefix + '_fpn' + str(lvl)] = rois[idx_lvl, :]
        rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
        rois_stacked = np.vstack(
            [rois_stacked, blobs[blob_prefix + '_fpn' + str(lvl)]]
        )
    rois_idx_restore = np.argsort(rois_idx_order).astype(np.int32, copy=False)
    blobs[blob_prefix + '_idx_restore_int32'] = rois_idx_restore
    # Sanity check that restore order is correct
    assert (rois_stacked[rois_idx_restore] == rois).all()

def _add_multilevel_rois(blobs):
    """By default training RoIs are added for a single feature map level only.
    When using FPN, the RoIs must be distributed over different FPN levels
    according the level assignment heuristic (see: modeling.FPN.
    map_rois_to_fpn_levels).
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL

    def _distribute_rois_over_fpn_levels(rois_blob_name):
        """Distribute rois over the different FPN levels."""
        # Get target level for each roi
        # Recall blob rois are in (batch_idx, x1, y1, x2, y2) format, hence take
        # the box coordinates from columns 1:5
        target_lvls = fpn_map_rois_to_fpn_levels(
            blobs[rois_blob_name][:, 1:5], lvl_min, lvl_max
        )
        # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
        fpn_add_multilevel_roi_blobs(
            blobs, rois_blob_name, blobs[rois_blob_name], target_lvls, lvl_min,
            lvl_max
        )

    _distribute_rois_over_fpn_levels('rois')
    if cfg.MODEL.MASK_ON:
        _distribute_rois_over_fpn_levels('mask_rois')
    if cfg.MODEL.KEYPOINTS_ON:
        _distribute_rois_over_fpn_levels('keypoint_rois')
    if cfg.MODEL.BODY_UV_ON:
        _distribute_rois_over_fpn_levels('body_uv_rois')

def segm_utils_GetDensePoseMask(Polys):
    MaskGen = np.zeros([256,256])
    for i in range(1,15):
        if(Polys[i-1]):
            current_mask = mask_util.decode(Polys[i-1])
            MaskGen[current_mask>0] = i
    return MaskGen

def body_uv_rcnn_roi_data_add_body_uv_rcnn_blobs(blobs, sampled_boxes, roidb, im_scale, batch_idx):
    IsFlipped = roidb['flipped']
    M = cfg.BODY_UV_RCNN.HEATMAP_SIZE
    #
    polys_gt_inds = np.where(roidb['ignore_UV_body'] == 0)[0]
    boxes_from_polys = [roidb['boxes'][i,:] for i in polys_gt_inds]
    if not(boxes_from_polys):
        pass
    else:
        boxes_from_polys = np.vstack(boxes_from_polys)
    boxes_from_polys = np.array(boxes_from_polys)

    fg_inds = np.where(blobs['labels_int32'] > 0)[0]
    roi_has_mask = np.zeros( blobs['labels_int32'].shape )

    if (bool(boxes_from_polys.any()) & (fg_inds.shape[0] > 0) ):
        rois_fg = sampled_boxes[fg_inds]
        #
        rois_fg.astype(np.float32, copy=False)
        boxes_from_polys.astype(np.float32, copy=False)
        #
        overlaps_bbfg_bbpolys = box_utils_bbox_overlaps(
            rois_fg.astype(np.float32, copy=False),
            boxes_from_polys.astype(np.float32, copy=False))
        fg_polys_value = np.max(overlaps_bbfg_bbpolys, axis=1)
        fg_inds = fg_inds[fg_polys_value>0.7]

    if (bool(boxes_from_polys.any()) & (fg_inds.shape[0] > 0) ):
        for jj in fg_inds:
            roi_has_mask[jj] = 1
         
        # Create blobs for densepose supervision.
        ################################################## The mask
        All_labels = blob_utils_zeros((fg_inds.shape[0], M ** 2), int32=True)
        All_Weights = blob_utils_zeros((fg_inds.shape[0], M ** 2), int32=True)
        ################################################# The points
        X_points = blob_utils_zeros((fg_inds.shape[0], 196), int32=False)
        Y_points = blob_utils_zeros((fg_inds.shape[0], 196), int32=False)
        Ind_points = blob_utils_zeros((fg_inds.shape[0], 196), int32=True)
        I_points = blob_utils_zeros((fg_inds.shape[0], 196), int32=True)
        U_points = blob_utils_zeros((fg_inds.shape[0], 196), int32=False)
        V_points = blob_utils_zeros((fg_inds.shape[0], 196), int32=False)
        Uv_point_weights = blob_utils_zeros((fg_inds.shape[0], 196), int32=False)
        #################################################

        rois_fg = sampled_boxes[fg_inds]
        overlaps_bbfg_bbpolys = box_utils_bbox_overlaps(
            rois_fg.astype(np.float32, copy=False),
            boxes_from_polys.astype(np.float32, copy=False))
        fg_polys_inds = np.argmax(overlaps_bbfg_bbpolys, axis=1)

        for i in range(rois_fg.shape[0]):
            #
            fg_polys_ind = polys_gt_inds[ fg_polys_inds[i] ]
            #
            Ilabel = segm_utils_GetDensePoseMask( roidb['dp_masks'][ fg_polys_ind ] )
            #
            GT_I = np.array(roidb['dp_I'][ fg_polys_ind ])
            GT_U = np.array(roidb['dp_U'][ fg_polys_ind ])
            GT_V = np.array(roidb['dp_V'][ fg_polys_ind ])
            GT_x = np.array(roidb['dp_x'][ fg_polys_ind ])
            GT_y = np.array(roidb['dp_y'][ fg_polys_ind ])
            GT_weights = np.ones(GT_I.shape).astype(np.float32)
            #
            ## Do the flipping of the densepose annotation !
            if(IsFlipped):
                GT_I,GT_U,GT_V,GT_x,GT_y,Ilabel = DP.get_symmetric_densepose(GT_I,GT_U,GT_V,GT_x,GT_y,Ilabel)
            #
            roi_fg = rois_fg[i]
            roi_gt = boxes_from_polys[fg_polys_inds[i],:]
            #
            x1 = roi_fg[0]  ;   x2 = roi_fg[2]
            y1 = roi_fg[1]  ;   y2 = roi_fg[3]
            #
            x1_source = roi_gt[0];  x2_source = roi_gt[2]
            y1_source = roi_gt[1];  y2_source = roi_gt[3]
            #
            x_targets  = ( np.arange(x1,x2, (x2 - x1)/M ) - x1_source ) * ( 256. / (x2_source-x1_source) )  
            y_targets  = ( np.arange(y1,y2, (y2 - y1)/M ) - y1_source ) * ( 256. / (y2_source-y1_source) )  
            #
            x_targets = x_targets[0:M] ## Strangely sometimes it can be M+1, so make sure size is OK!
            y_targets = y_targets[0:M]
            #
            [X_targets,Y_targets] = np.meshgrid( x_targets, y_targets )
            New_Index = cv2.remap(Ilabel,X_targets.astype(np.float32), Y_targets.astype(np.float32), interpolation=cv2.INTER_NEAREST, borderMode= cv2.BORDER_CONSTANT, borderValue=(0))
            #
            All_L = np.zeros(New_Index.shape)
            All_W = np.ones(New_Index.shape)
            #
            All_L = New_Index
            #
            gt_length_x = x2_source - x1_source
            gt_length_y = y2_source - y1_source
            #
            GT_y =  ((  GT_y / 256. * gt_length_y  ) + y1_source - y1 ) *  ( M /  ( y2 - y1 ) )
            GT_x =  ((  GT_x / 256. * gt_length_x  ) + x1_source - x1 ) *  ( M /  ( x2 - x1 ) )
            #
            GT_I[GT_y<0] = 0
            GT_I[GT_y>(M-1)] = 0
            GT_I[GT_x<0] = 0
            GT_I[GT_x>(M-1)] = 0
            #
            points_inside = GT_I>0
            GT_U = GT_U[points_inside]
            GT_V = GT_V[points_inside]
            GT_x = GT_x[points_inside]
            GT_y = GT_y[points_inside]
            GT_weights = GT_weights[points_inside]
            GT_I = GT_I[points_inside]
            #
            X_points[i, 0:len(GT_x)] = GT_x
            Y_points[i, 0:len(GT_y)] = GT_y
            Ind_points[i, 0:len(GT_I)] = i
            I_points[i, 0:len(GT_I)] = GT_I
            U_points[i, 0:len(GT_U)] = GT_U
            V_points[i, 0:len(GT_V)] = GT_V
            Uv_point_weights[i, 0:len(GT_weights)] = GT_weights
            #
            All_labels[i, :] = np.reshape(All_L.astype(np.int32), M ** 2)
            All_Weights[i, :] = np.reshape(All_W.astype(np.int32), M ** 2)
            ##
    else:
        bg_inds = np.where(blobs['labels_int32'] == 0)[0]
        #
        if(len(bg_inds)==0):
            rois_fg = sampled_boxes[0].reshape((1, -1))
        else:
            rois_fg = sampled_boxes[bg_inds[0]].reshape((1, -1))

        roi_has_mask[0] = 1
        #
        X_points = blob_utils_zeros((1, 196), int32=False)
        Y_points = blob_utils_zeros((1, 196), int32=False)
        Ind_points = blob_utils_zeros((1, 196), int32=True)
        I_points = blob_utils_zeros((1,196), int32=True)
        U_points = blob_utils_zeros((1, 196), int32=False)
        V_points = blob_utils_zeros((1, 196), int32=False)
        Uv_point_weights = blob_utils_zeros((1, 196), int32=False)
        #
        All_labels = -blob_utils_ones((1, M ** 2), int32=True) * 0 ## zeros
        All_Weights = -blob_utils_ones((1, M ** 2), int32=True) * 0 ## zeros
    #
    rois_fg *= im_scale
    repeated_batch_idx = batch_idx * blob_utils_ones((rois_fg.shape[0], 1))
    rois_fg = np.hstack((repeated_batch_idx, rois_fg))
    #
    K = cfg.BODY_UV_RCNN.NUM_PATCHES
    #
    U_points = np.tile( U_points , [1,K+1] )
    V_points = np.tile( V_points , [1,K+1] )
    Uv_Weight_Points = np.zeros(U_points.shape)
    #
    for jjj in range(1,K+1):
        Uv_Weight_Points[ : , jjj * I_points.shape[1]  : (jjj+1) * I_points.shape[1] ] = ( I_points == jjj ).astype(np.float32)
    #
    ################
    # Update blobs dict with Mask R-CNN blobs
    ###############
    #
    blobs['body_uv_rois'] = np.array(rois_fg)
    blobs['roi_has_body_uv_int32'] = np.array(roi_has_mask).astype(np.int32)
    ##
    blobs['body_uv_ann_labels'] = np.array(All_labels).astype(np.int32)
    blobs['body_uv_ann_weights'] = np.array(All_Weights).astype(np.float32)
    #
    ##########################
    blobs['body_uv_X_points'] = X_points.astype(np.float32)
    blobs['body_uv_Y_points'] = Y_points.astype(np.float32)
    blobs['body_uv_Ind_points'] = Ind_points.astype(np.float32)
    blobs['body_uv_I_points'] = I_points.astype(np.float32)
    blobs['body_uv_U_points'] = U_points.astype(np.float32)  #### VERY IMPORTANT :   These are switched here :
    blobs['body_uv_V_points'] = V_points.astype(np.float32)
    blobs['body_uv_point_weights'] = Uv_Weight_Points.astype(np.float32)
    ###################

def keypoint_utils_keypoints_to_heatmap_labels(keypoints, rois):
    """Encode keypoint location in the target heatmap for use in
    SoftmaxWithLoss.
    """
    # Maps keypoints from the half-open interval [x1, x2) on continuous image
    # coordinates to the closed interval [0, HEATMAP_SIZE - 1] on discrete image
    # coordinates. We use the continuous <-> discrete conversion from Heckbert
    # 1990 ("What is the coordinate of a pixel?"): d = floor(c) and c = d + 0.5,
    # where d is a discrete coordinate and c is a continuous coordinate.
    assert keypoints.shape[2] == cfg.KRCNN.NUM_KEYPOINTS

    shape = (len(rois), cfg.KRCNN.NUM_KEYPOINTS)
    heatmaps = blob_utils_zeros(shape)
    weights = blob_utils_zeros(shape)

    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = cfg.KRCNN.HEATMAP_SIZE / (rois[:, 2] - rois[:, 0])
    scale_y = cfg.KRCNN.HEATMAP_SIZE / (rois[:, 3] - rois[:, 1])

    for kp in range(keypoints.shape[2]):
        vis = keypoints[:, 2, kp] > 0
        x = keypoints[:, 0, kp].astype(np.float32)
        y = keypoints[:, 1, kp].astype(np.float32)
        # Since we use floor below, if a keypoint is exactly on the roi's right
        # or bottom boundary, we shift it in by eps (conceptually) to keep it in
        # the ground truth heatmap.
        x_boundary_inds = np.where(x == rois[:, 2])[0]
        y_boundary_inds = np.where(y == rois[:, 3])[0]
        x = (x - offset_x) * scale_x
        x = np.floor(x)
        if len(x_boundary_inds) > 0:
            x[x_boundary_inds] = cfg.KRCNN.HEATMAP_SIZE - 1

        y = (y - offset_y) * scale_y
        y = np.floor(y)
        if len(y_boundary_inds) > 0:
            y[y_boundary_inds] = cfg.KRCNN.HEATMAP_SIZE - 1

        valid_loc = np.logical_and(
            np.logical_and(x >= 0, y >= 0),
            np.logical_and(
                x < cfg.KRCNN.HEATMAP_SIZE, y < cfg.KRCNN.HEATMAP_SIZE))

        valid = np.logical_and(valid_loc, vis)
        valid = valid.astype(np.int32)

        lin_ind = y * cfg.KRCNN.HEATMAP_SIZE + x
        heatmaps[:, kp] = lin_ind * valid
        weights[:, kp] = valid

    return heatmaps, weights

def _within_box(points, boxes):
    """Validate which keypoints are contained inside a given box.

    points: Nx2xK
    boxes: Nx4
    output: NxK
    """
    x_within = np.logical_and(
        points[:, 0, :] >= np.expand_dims(boxes[:, 0], axis=1),
        points[:, 0, :] <= np.expand_dims(boxes[:, 2], axis=1)
    )
    y_within = np.logical_and(
        points[:, 1, :] >= np.expand_dims(boxes[:, 1], axis=1),
        points[:, 1, :] <= np.expand_dims(boxes[:, 3], axis=1)
    )
    return np.logical_and(x_within, y_within)

def keypoint_rcnn_roi_data_add_keypoint_rcnn_blobs(
    blobs, roidb, fg_rois_per_image, fg_inds, im_scale, batch_idx
):
    """Add Mask R-CNN keypoint specific blobs to the given blobs dictionary."""
    # Note: gt_inds must match how they're computed in
    # datasets.json_dataset._merge_proposal_boxes_into_roidb
    gt_inds = np.where(roidb['gt_classes'] > 0)[0]
    max_overlaps = roidb['max_overlaps']
    gt_keypoints = roidb['gt_keypoints']

    ind_kp = gt_inds[roidb['box_to_gt_ind_map']]
    within_box = _within_box(gt_keypoints[ind_kp, :, :], roidb['boxes'])
    vis_kp = gt_keypoints[ind_kp, 2, :] > 0
    is_visible = np.sum(np.logical_and(vis_kp, within_box), axis=1) > 0
    kp_fg_inds = np.where(
        np.logical_and(max_overlaps >= cfg.TRAIN.FG_THRESH, is_visible)
    )[0]

    kp_fg_rois_per_this_image = np.minimum(fg_rois_per_image, kp_fg_inds.size)
    if kp_fg_inds.size > kp_fg_rois_per_this_image:
        kp_fg_inds = np.random.choice(
            kp_fg_inds, size=kp_fg_rois_per_this_image, replace=False
        )

    sampled_fg_rois = roidb['boxes'][kp_fg_inds]
    box_to_gt_ind_map = roidb['box_to_gt_ind_map'][kp_fg_inds]

    num_keypoints = gt_keypoints.shape[2]
    sampled_keypoints = -np.ones(
        (len(sampled_fg_rois), gt_keypoints.shape[1], num_keypoints),
        dtype=gt_keypoints.dtype
    )
    for ii in range(len(sampled_fg_rois)):
        ind = box_to_gt_ind_map[ii]
        if ind >= 0:
            sampled_keypoints[ii, :, :] = gt_keypoints[gt_inds[ind], :, :]
            assert np.sum(sampled_keypoints[ii, 2, :]) > 0

    heats, weights = keypoint_utils_keypoints_to_heatmap_labels(
        sampled_keypoints, sampled_fg_rois
    )

    shape = (sampled_fg_rois.shape[0] * cfg.KRCNN.NUM_KEYPOINTS, 1)
    heats = heats.reshape(shape)
    weights = weights.reshape(shape)

    sampled_fg_rois *= im_scale
    repeated_batch_idx = batch_idx * blob_utils_ones(
        (sampled_fg_rois.shape[0], 1)
    )
    sampled_fg_rois = np.hstack((repeated_batch_idx, sampled_fg_rois))

    blobs['keypoint_rois'] = sampled_fg_rois
    blobs['keypoint_locations_int32'] = heats.astype(np.int32, copy=False)
    blobs['keypoint_weights'] = weights

def _expand_to_class_specific_mask_targets(masks, mask_class_labels):
    """Expand masks from shape (#masks, M ** 2) to (#masks, #classes * M ** 2)
    to encode class specific mask targets.
    """
    assert masks.shape[0] == mask_class_labels.shape[0]
    M = cfg.MRCNN.RESOLUTION

    # Target values of -1 are "don't care" / ignore labels
    mask_targets = -blob_utils_ones(
        (masks.shape[0], cfg.MODEL.NUM_CLASSES * M**2), int32=True
    )

    for i in range(masks.shape[0]):
        cls = int(mask_class_labels[i])
        start = M**2 * cls
        end = start + M**2
        # Ignore background instance
        # (only happens when there is no fg samples in an image)
        if cls > 0:
            mask_targets[i, start:end] = masks[i, :]

    return mask_targets

def segm_utils_polys_to_mask_wrt_box(polygons, box, M):
    """Convert from the COCO polygon segmentation format to a binary mask
    encoded as a 2D array of data type numpy.float32. The polygon segmentation
    is understood to be enclosed in the given box and rasterized to an M x M
    mask. The resulting mask is therefore of shape (M, M).
    """
    w = box[2] - box[0]
    h = box[3] - box[1]

    w = np.maximum(w, 1)
    h = np.maximum(h, 1)

    polygons_norm = []
    for poly in polygons:
        p = np.array(poly, dtype=np.float32)
        p[0::2] = (p[0::2] - box[0]) * M / w
        p[1::2] = (p[1::2] - box[1]) * M / h
        polygons_norm.append(p)

    rle = mask_util.frPyObjects(polygons_norm, M, M)
    mask = np.array(mask_util.decode(rle), dtype=np.float32)
    # Flatten in case polygons was a list
    mask = np.sum(mask, axis=2)
    mask = np.array(mask > 0, dtype=np.float32)
    return mask

def segm_utils_polys_to_boxes(polys):
    """Convert a list of polygons into an array of tight bounding boxes."""
    boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
    for i in range(len(polys)):
        poly = polys[i]
        x0 = min(min(p[::2]) for p in poly)
        x1 = max(max(p[::2]) for p in poly)
        y0 = min(min(p[1::2]) for p in poly)
        y1 = max(max(p[1::2]) for p in poly)
        boxes_from_polys[i, :] = [x0, y0, x1, y1]

    return boxes_from_polys

def mask_rcnn_roi_data_add_mask_rcnn_blobs(blobs, sampled_boxes, roidb, im_scale, batch_idx):
    """Add Mask R-CNN specific blobs to the input blob dictionary."""
    # Prepare the mask targets by associating one gt mask to each training roi
    # that has a fg (non-bg) class label.
    M = cfg.MRCNN.RESOLUTION
    polys_gt_inds = np.where(
        (roidb['gt_classes'] > 0) & (roidb['is_crowd'] == 0)
    )[0]
    polys_gt = [roidb['segms'][i] for i in polys_gt_inds]
    boxes_from_polys = segm_utils_polys_to_boxes(polys_gt)
    fg_inds = np.where(blobs['labels_int32'] > 0)[0]
    roi_has_mask = blobs['labels_int32'].copy()
    roi_has_mask[roi_has_mask > 0] = 1

    if fg_inds.shape[0] > 0:
        # Class labels for the foreground rois
        mask_class_labels = blobs['labels_int32'][fg_inds]
        masks = blob_utils_zeros((fg_inds.shape[0], M**2), int32=True)

        # Find overlap between all foreground rois and the bounding boxes
        # enclosing each segmentation
        rois_fg = sampled_boxes[fg_inds]
        overlaps_bbfg_bbpolys = box_utils_bbox_overlaps(
            rois_fg.astype(np.float32, copy=False),
            boxes_from_polys.astype(np.float32, copy=False)
        )
        # Map from each fg rois to the index of the mask with highest overlap
        # (measured by bbox overlap)
        fg_polys_inds = np.argmax(overlaps_bbfg_bbpolys, axis=1)

        # add fg targets
        for i in range(rois_fg.shape[0]):
            fg_polys_ind = fg_polys_inds[i]
            poly_gt = polys_gt[fg_polys_ind]
            roi_fg = rois_fg[i]
            # Rasterize the portion of the polygon mask within the given fg roi
            # to an M x M binary image
            mask = segm_utils_polys_to_mask_wrt_box(poly_gt, roi_fg, M)
            mask = np.array(mask > 0, dtype=np.int32)  # Ensure it's binary
            masks[i, :] = np.reshape(mask, M**2)
    else:  # If there are no fg masks (it does happen)
        # The network cannot handle empty blobs, so we must provide a mask
        # We simply take the first bg roi, given it an all -1's mask (ignore
        # label), and label it with class zero (bg).
        bg_inds = np.where(blobs['labels_int32'] == 0)[0]
        # rois_fg is actually one background roi, but that's ok because ...
        rois_fg = sampled_boxes[bg_inds[0]].reshape((1, -1))
        # We give it an -1's blob (ignore label)
        masks = -blob_utils_ones((1, M**2), int32=True)
        # We label it with class = 0 (background)
        mask_class_labels = blob_utils_zeros((1, ))
        # Mark that the first roi has a mask
        roi_has_mask[0] = 1

    if cfg.MRCNN.CLS_SPECIFIC_MASK:
        masks = _expand_to_class_specific_mask_targets(masks, mask_class_labels)

    # Scale rois_fg and format as (batch_idx, x1, y1, x2, y2)
    rois_fg *= im_scale
    repeated_batch_idx = batch_idx * blob_utils_ones((rois_fg.shape[0], 1))
    rois_fg = np.hstack((repeated_batch_idx, rois_fg))

    # Update blobs dict with Mask R-CNN blobs
    blobs['mask_rois'] = rois_fg
    blobs['roi_has_mask_int32'] = roi_has_mask
    blobs['masks_int32'] = masks

def blob_utils_ones(shape, int32=False):
    """Return a blob of all ones of the given shape with the correct float or
    int data type.
    """
    return np.ones(shape, dtype=np.int32 if int32 else np.float32)

def blob_utils_zeros(shape, int32=False):
    """Return a blob of all zeros of the given shape with the correct float or
    int data type.
    """
    return np.zeros(shape, dtype=np.int32 if int32 else np.float32)

def _expand_bbox_targets(bbox_target_data):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    num_bbox_reg_classes = cfg.MODEL.NUM_CLASSES
    if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
        num_bbox_reg_classes = 2  # bg and fg

    clss = bbox_target_data[:, 0]
    bbox_targets = blob_utils_zeros((clss.size, 4 * num_bbox_reg_classes))
    bbox_inside_weights = blob_utils_zeros(bbox_targets.shape)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights

def _sample_rois(roidb, im_scale, batch_idx):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    rois_per_image = int(cfg.TRAIN.BATCH_SIZE_PER_IM)
    fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
    max_overlaps = roidb['max_overlaps']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(
            fg_inds, size=fg_rois_per_this_image, replace=False
        )

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where(
        (max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
        (max_overlaps >= cfg.TRAIN.BG_THRESH_LO)
    )[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(
            bg_inds, size=bg_rois_per_this_image, replace=False
        )

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Label is the class each RoI has max overlap with
    sampled_labels = roidb['max_classes'][keep_inds]
    sampled_labels[fg_rois_per_this_image:] = 0  # Label bg RoIs with class 0
    sampled_boxes = roidb['boxes'][keep_inds]

    bbox_targets, bbox_inside_weights = _expand_bbox_targets(
        roidb['bbox_targets'][keep_inds, :]
    )
    bbox_outside_weights = np.array(
        bbox_inside_weights > 0, dtype=bbox_inside_weights.dtype
    )

    # Scale rois and format as (batch_idx, x1, y1, x2, y2)
    sampled_rois = sampled_boxes * im_scale
    repeated_batch_idx = batch_idx * blob_utils_ones((sampled_rois.shape[0], 1))
    sampled_rois = np.hstack((repeated_batch_idx, sampled_rois))

    # Base Fast R-CNN blobs
    blob_dict = dict(
        labels_int32=sampled_labels.astype(np.int32, copy=False),
        rois=sampled_rois,
        bbox_targets=bbox_targets,
        bbox_inside_weights=bbox_inside_weights,
        bbox_outside_weights=bbox_outside_weights
    )

    # Optionally add Mask R-CNN blobs
    if cfg.MODEL.MASK_ON:
        mask_rcnn_roi_data_add_mask_rcnn_blobs(
            blob_dict, sampled_boxes, roidb, im_scale, batch_idx
        )

    # Optionally add Keypoint R-CNN blobs
    if cfg.MODEL.KEYPOINTS_ON:
        keypoint_rcnn_roi_data_add_keypoint_rcnn_blobs(
            blob_dict, roidb, fg_rois_per_image, fg_inds, im_scale, batch_idx
        )

    # Optionally body UV R-CNN blobs
    if cfg.MODEL.BODY_UV_ON:
        body_uv_rcnn_roi_data_add_body_uv_rcnn_blobs(
            blob_dict, sampled_boxes, roidb, im_scale, batch_idx
        )

    return blob_dict

def fast_rcnn_roi_data_add_fast_rcnn_blobs(blobs, im_scales, roidb):
    """Add blobs needed for training Fast R-CNN style models."""
    # Sample training RoIs from each image and append them to the blob lists
    for im_i, entry in enumerate(roidb):
        frcn_blobs = _sample_rois(entry, im_scales[im_i], im_i)
        for k, v in frcn_blobs.items():
            blobs[k].append(v)
    # Concat the training blob lists into tensors
    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)
    # Add FPN multilevel training RoIs, if configured
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois(blobs)

    # Perform any final work and validity checks after the collating blobs for
    # all minibatch images
    valid = True
    if cfg.MODEL.KEYPOINTS_ON:
        valid = keypoint_rcnn_roi_data_finalize_keypoint_minibatch(blobs, valid)

    return valid

def box_utils_bbox_transform_inv(boxes, gt_boxes, weights=(1.0, 1.0, 1.0, 1.0)):
    """Inverse transform that computes target bounding-box regression deltas
    given proposal boxes and ground-truth boxes. The weights argument should be
    a 4-tuple of multiplicative weights that are applied to the regression
    target.

    In older versions of this code (and in py-faster-rcnn), the weights were set
    such that the regression deltas would have unit standard deviation on the
    training dataset. Presently, rather than computing these statistics exactly,
    we use a fixed set of weights (10., 10., 5., 5.) by default. These are
    approximately the weights one would get from COCO using the previous unit
    stdev heuristic.
    """
    ex_widths = boxes[:, 2] - boxes[:, 0] + 1.0
    ex_heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ex_ctr_x = boxes[:, 0] + 0.5 * ex_widths
    ex_ctr_y = boxes[:, 1] + 0.5 * ex_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    wx, wy, ww, wh = weights
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * np.log(gt_widths / ex_widths)
    targets_dh = wh * np.log(gt_heights / ex_heights)

    targets = np.vstack((targets_dx, targets_dy, targets_dw,
                         targets_dh)).transpose()
    return targets

def compute_bbox_regression_targets(entry):
    """Compute bounding-box regression targets for an image."""
    # Indices of ground-truth ROIs
    rois = entry['boxes']
    overlaps = entry['max_overlaps']
    labels = entry['max_classes']
    gt_inds = np.where((entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
    # Targets has format (class, tx, ty, tw, th)
    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    if len(gt_inds) == 0:
        # Bail if the image has no ground-truth ROIs
        return targets

    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = box_utils_bbox_overlaps(
        rois[ex_inds, :].astype(dtype=np.float32, copy=False),
        rois[gt_inds, :].astype(dtype=np.float32, copy=False))

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]
    # Use class "1" for all boxes if using class_agnostic_bbox_reg
    targets[ex_inds, 0] = (
        1 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else labels[ex_inds])
    targets[ex_inds, 1:] = box_utils_bbox_transform_inv(
        ex_rois, gt_rois, cfg.MODEL.BBOX_REG_WEIGHTS)
    return targets

def roidb_utils_add_bbox_regression_targets(roidb):
    """Add information needed to train bounding-box regressors."""
    for entry in roidb:
        entry['bbox_targets'] = compute_bbox_regression_targets(entry)

def _add_class_assignments(roidb):
    """Compute object category assignment for each box associated with each
    roidb entry.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        entry['max_classes'] = max_classes
        entry['max_overlaps'] = max_overlaps
        # sanity checks
        # if max overlap is 0, the class must be background (class 0)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # if max overlap > 0, the class must be a fg class (not class 0)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)

def box_utils_xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')

def _filter_crowd_proposals(roidb, crowd_thresh):
    """Finds proposals that are inside crowd regions and marks them as
    overlap = -1 with each ground-truth rois, which means they will be excluded
    from training.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        crowd_inds = np.where(entry['is_crowd'] == 1)[0]
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        if len(crowd_inds) == 0 or len(non_gt_inds) == 0:
            continue
        crowd_boxes = box_utils_xyxy_to_xywh(entry['boxes'][crowd_inds, :])
        non_gt_boxes = box_utils_xyxy_to_xywh(entry['boxes'][non_gt_inds, :])
        iscrowd_flags = [int(True)] * len(crowd_inds)
        ious = COCOmask.iou(non_gt_boxes, crowd_boxes, iscrowd_flags)
        bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
        gt_overlaps[non_gt_inds[bad_inds], :] = -1
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)

def _merge_proposal_boxes_into_roidb(roidb, box_list):
    """Add proposal boxes to each roidb entry."""
    assert len(box_list) == len(roidb)
    for i, entry in enumerate(roidb):
        boxes = box_list[i]
        num_boxes = boxes.shape[0]
        gt_overlaps = np.zeros(
            (num_boxes, entry['gt_overlaps'].shape[1]),
            dtype=entry['gt_overlaps'].dtype
        )
        box_to_gt_ind_map = -np.ones(
            (num_boxes), dtype=entry['box_to_gt_ind_map'].dtype
        )

        # Note: unlike in other places, here we intentionally include all gt
        # rois, even ones marked as crowd. Boxes that overlap with crowds will
        # be filtered out later (see: _filter_crowd_proposals).
        gt_inds = np.where(entry['gt_classes'] > 0)[0]
        if len(gt_inds) > 0:
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_classes = entry['gt_classes'][gt_inds]
            proposal_to_gt_overlaps = box_utils_bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False)
            )
            # Gt box that overlaps each input box the most
            # (ties are broken arbitrarily by class order)
            argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
            # Amount of that overlap
            maxes = proposal_to_gt_overlaps.max(axis=1)
            # Those boxes with non-zero overlap with gt boxes
            I = np.where(maxes > 0)[0]
            # Record max overlaps with the class of the appropriate gt box
            gt_overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            box_to_gt_ind_map[I] = gt_inds[argmaxes[I]]
        entry['boxes'] = np.append(
            entry['boxes'],
            boxes.astype(entry['boxes'].dtype, copy=False),
            axis=0
        )
        entry['gt_classes'] = np.append(
            entry['gt_classes'],
            np.zeros((num_boxes), dtype=entry['gt_classes'].dtype)
        )
        entry['seg_areas'] = np.append(
            entry['seg_areas'],
            np.zeros((num_boxes), dtype=entry['seg_areas'].dtype)
        )
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(
            entry['is_crowd'],
            np.zeros((num_boxes), dtype=entry['is_crowd'].dtype)
        )
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'],
            box_to_gt_ind_map.astype(
                entry['box_to_gt_ind_map'].dtype, copy=False
            )
        )

def json_dataset_add_proposals(roidb, rois, scales, crowd_thresh):
    """Add proposal boxes (rois) to an roidb that has ground-truth annotations
    but no proposals. If the proposals are not at the original image scale,
    specify the scale factor that separate them in scales.
    """
    box_list = []
    for i in range(len(roidb)):
        inv_im_scale = 1. / scales[i]
        idx = np.where(rois[:, 0] == i)[0]
        box_list.append(rois[idx, 1:] * inv_im_scale)
    _merge_proposal_boxes_into_roidb(roidb, box_list)
    if crowd_thresh > 0:
        _filter_crowd_proposals(roidb, crowd_thresh)
    _add_class_assignments(roidb)

def blob_utils_deserialize(arr):
    """Unserialize a Python object from an array of float32 values fetched from
    a workspace. See serialize().
    """
    return pickle.loads(arr.astype(np.uint8).tobytes())

def box_utils_boxes_area(boxes):
    """Compute the area of an array of boxes."""
    w = (boxes[:, 2] - boxes[:, 0] + 1)
    h = (boxes[:, 3] - boxes[:, 1] + 1)
    areas = w * h
    assert np.all(areas >= 0), 'Negative areas founds'
    return areas

def fpn_map_rois_to_fpn_levels(rois, k_min, k_max):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """
    # Compute level ids
    s = np.sqrt(box_utils_boxes_area(rois))
    s0 = cfg.FPN.ROI_CANONICAL_SCALE  # default: 224
    lvl0 = cfg.FPN.ROI_CANONICAL_LEVEL  # default: 4

    # Eqn.(1) in FPN paper
    target_lvls = np.floor(lvl0 + np.log2(s / s0 + 1e-6))
    target_lvls = np.clip(target_lvls, k_min, k_max)
    return target_lvls

def distribute(rois, label_blobs, outputs, train):
    """To understand the output blob order see return value of
    detectron.roi_data.fast_rcnn.get_fast_rcnn_blob_names(is_training=False)
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    lvls = fpn_map_rois_to_fpn_levels(rois[:, 1:5], lvl_min, lvl_max)

    outputs[0].reshape(rois.shape)
    outputs[0].data[...] = rois

    # Create new roi blobs for each FPN level
    # (See: modeling.FPN.add_multilevel_roi_blobs which is similar but annoying
    # to generalize to support this particular case.)
    rois_idx_order = np.empty((0, ))
    for output_idx, lvl in enumerate(range(lvl_min, lvl_max + 1)):
        idx_lvl = np.where(lvls == lvl)[0]
        blob_roi_level = rois[idx_lvl, :]
        outputs[output_idx + 1].reshape(blob_roi_level.shape)
        outputs[output_idx + 1].data[...] = blob_roi_level
        rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
    rois_idx_restore = np.argsort(rois_idx_order)
    blob_utils_py_op_copy_blob(rois_idx_restore.astype(np.int32), outputs[-1])

def collect(inputs, is_training):
    cfg_key = 'TRAIN' if is_training else 'TEST'
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    k_max = cfg.FPN.RPN_MAX_LEVEL
    k_min = cfg.FPN.RPN_MIN_LEVEL
    num_lvls = k_max - k_min + 1
    roi_inputs = inputs[:num_lvls]
    score_inputs = inputs[num_lvls:]
    if is_training:
        score_inputs = score_inputs[:-2]

    # rois are in [[batch_idx, x0, y0, x1, y2], ...] format
    # Combine predictions across all levels and retain the top scoring
    rois = np.concatenate([blob.data for blob in roi_inputs])
    scores = np.concatenate([blob.data for blob in score_inputs]).squeeze()
    inds = np.argsort(-scores)[:post_nms_topN]
    rois = rois[inds, :]
    return rois

def box_utils_bbox_transform(boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0)):
    """Forward transform that maps proposal boxes to predicted ground-truth
    boxes using bounding-box regression deltas. See bbox_transform_inv for a
    description of the weights argument.
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    # Prevent sending too large values into np.exp()
    dw = np.minimum(dw, cfg.BBOX_XFORM_CLIP)
    dh = np.minimum(dh, cfg.BBOX_XFORM_CLIP)

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

    return pred_boxes

def c2_utils_CudaDevice(gpu_id):
    """Create a Cuda device."""
    return core.DeviceOption(caffe2_pb2.CUDA, gpu_id)

@contextlib.contextmanager
def c2_utils_CudaScope(gpu_id):
    """Create a CUDA device scope for GPU device `gpu_id`."""
    gpu_dev = c2_utils_CudaDevice(gpu_id)
    with core.DeviceScope(gpu_dev):
        yield

@contextlib.contextmanager
def c2_utils_GpuNameScope(gpu_id):
    """Create a name scope for GPU device `gpu_id`."""
    with core.NameScope('gpu_{:d}'.format(gpu_id)):
        yield

@contextlib.contextmanager
def c2_utils_NamedCudaScope(gpu_id):
    """Creates a GPU name scope and CUDA device scope. This function is provided
    to reduce `with ...` nesting levels."""
    with c2_utils_GpuNameScope(gpu_id):
        with c2_utils_CudaScope(gpu_id):
            yield

@contextlib.contextmanager
def c2_utils_CpuScope():
    """Create a CPU device scope."""
    cpu_dev = core.DeviceOption(caffe2_pb2.CPU)
    with core.DeviceScope(cpu_dev):
        yield

class DensePoseMethods:
    def __init__(self):
        #
        ALP_UV = loadmat( os.path.join(os.path.dirname(__file__), 'assets/UV_Processed.mat')  )
        self.FaceIndices = np.array( ALP_UV['All_FaceIndices']).squeeze()
        self.FacesDensePose = ALP_UV['All_Faces']-1
        self.U_norm = ALP_UV['All_U_norm'].squeeze()
        self.V_norm = ALP_UV['All_V_norm'].squeeze()
        self.All_vertices =  ALP_UV['All_vertices'][0]
        ## Info to compute symmetries.
        self.SemanticMaskSymmetries = [0,1,3,2,5,4,7,6,9,8,11,10,13,12,14]
        self.Index_Symmetry_List = [1,2,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17,20,19,22,21,24,23];
        UV_symmetry_filename = os.path.join(os.path.dirname(__file__), 'assets/UV_symmetry_transforms.mat')
        self.UV_symmetry_transformations = loadmat( UV_symmetry_filename )
    

    def get_symmetric_densepose(self,I,U,V,x,y,Mask):
        ### This is a function to get the mirror symmetric UV labels.
        Labels_sym= np.zeros(I.shape)
        U_sym= np.zeros(U.shape)
        V_sym= np.zeros(V.shape)
        ###
        for i in ( range(24)):
            if i+1 in I:
                Labels_sym[I == (i+1)] = self.Index_Symmetry_List[i]
                jj = np.where(I == (i+1))
                ###
                U_loc = (U[jj]*255).astype(np.int64)
                V_loc = (V[jj]*255).astype(np.int64)
                ###
                V_sym[jj] = self.UV_symmetry_transformations['V_transforms'][0,i][V_loc,U_loc]
                U_sym[jj] = self.UV_symmetry_transformations['U_transforms'][0,i][V_loc,U_loc]
        ##
        Mask_flip = np.fliplr(Mask)
        Mask_flipped = np.zeros(Mask.shape)
        #
        for i in ( range(14)):
            Mask_flipped[Mask_flip == (i+1)] = self.SemanticMaskSymmetries[i+1]
        #
        [y_max , x_max ] = Mask_flip.shape
        y_sym = y
        x_sym = x_max-x
        #
        return Labels_sym , U_sym , V_sym , x_sym , y_sym , Mask_flipped
    
    
    
    def barycentric_coordinates_exists(self,P0, P1, P2, P):
        u = P1 - P0
        v = P2 - P0
        w = P - P0
        #
        vCrossW = np.cross(v,w)
        vCrossU = np.cross(v, u)
        if (np.dot(vCrossW, vCrossU) < 0):
            return False;
        #
        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)
        #
        if (np.dot(uCrossW, uCrossV) < 0):
            return False;
        #
        denom = np.sqrt((uCrossV**2).sum())
        r = np.sqrt((vCrossW**2).sum())/denom
        t = np.sqrt((uCrossW**2).sum())/denom
        #
        return((r <=1) & (t <= 1) & (r + t <= 1))

    def barycentric_coordinates(self,P0, P1, P2, P):
        u = P1 - P0
        v = P2 - P0
        w = P - P0
        #
        vCrossW = np.cross(v,w)
        vCrossU = np.cross(v, u)
        #
        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)
        #
        denom = np.sqrt((uCrossV**2).sum())
        r = np.sqrt((vCrossW**2).sum())/denom
        t = np.sqrt((uCrossW**2).sum())/denom
        #
        return(1-(r+t),r,t)

    def IUV2FBC( self, I_point , U_point, V_point):
        P = [ U_point , V_point , 0 ]
        FaceIndicesNow  = np.where( self.FaceIndices == I_point )
        FacesNow = self.FacesDensePose[FaceIndicesNow]
        #
        P_0 = np.vstack( (self.U_norm[FacesNow][:,0], self.V_norm[FacesNow][:,0], np.zeros(self.U_norm[FacesNow][:,0].shape))).transpose()
        P_1 = np.vstack( (self.U_norm[FacesNow][:,1], self.V_norm[FacesNow][:,1], np.zeros(self.U_norm[FacesNow][:,1].shape))).transpose()
        P_2 = np.vstack( (self.U_norm[FacesNow][:,2], self.V_norm[FacesNow][:,2], np.zeros(self.U_norm[FacesNow][:,2].shape))).transpose()
        #

        for i, [P0,P1,P2] in enumerate( zip(P_0,P_1,P_2)) :
            if(self.barycentric_coordinates_exists(P0, P1, P2, P)):
                [bc1,bc2,bc3] = self.barycentric_coordinates(P0, P1, P2, P)
                return(FaceIndicesNow[0][i],bc1,bc2,bc3)
        #
        # If the found UV is not inside any faces, select the vertex that is closest!
        #
        D1 = scipy.spatial.distance.cdist( np.array( [U_point,V_point])[np.newaxis,:] , P_0[:,0:2]).squeeze()
        D2 = scipy.spatial.distance.cdist( np.array( [U_point,V_point])[np.newaxis,:] , P_1[:,0:2]).squeeze()
        D3 = scipy.spatial.distance.cdist( np.array( [U_point,V_point])[np.newaxis,:] , P_2[:,0:2]).squeeze()
        #
        minD1 = D1.min()
        minD2 = D2.min()
        minD3 = D3.min()
        #
        if((minD1< minD2) & (minD1< minD3)):
            return(  FaceIndicesNow[0][np.argmin(D1)] , 1.,0.,0. )
        elif((minD2< minD1) & (minD2< minD3)):
            return(  FaceIndicesNow[0][np.argmin(D2)] , 0.,1.,0. )
        else:
            return(  FaceIndicesNow[0][np.argmin(D3)] , 0.,0.,1. )


    def FBC2PointOnSurface( self, FaceIndex, bc1,bc2,bc3,Vertices ):
        ##
        Vert_indices = self.All_vertices[self.FacesDensePose[FaceIndex]]-1
        ##
        p = Vertices[Vert_indices[0],:] * bc1 +  \
            Vertices[Vert_indices[1],:] * bc2 +  \
            Vertices[Vert_indices[2],:] * bc3 
        ##
        return(p)    
 
class CollectAndDistributeFpnRpnProposalsOp(object):
    def __init__(self, train):
        self._train = train

    def forward(self, inputs, outputs):
        """See modeling.detector.CollectAndDistributeFpnRpnProposals for
        inputs/outputs documentation.
        """
        # inputs is
        # [rpn_rois_fpn2, ..., rpn_rois_fpn6,
        #  rpn_roi_probs_fpn2, ..., rpn_roi_probs_fpn6]
        # If training with Faster R-CNN, then inputs will additionally include
        #  + [roidb, im_info]
        rois = collect(inputs, self._train)
        if self._train:
            # During training we reuse the data loader code. We populate roidb
            # entries on the fly using the rois generated by RPN.
            # im_info: [[im_height, im_width, im_scale], ...]
            im_info = inputs[-1].data
            im_scales = im_info[:, 2]
            roidb = blob_utils_deserialize(inputs[-2].data)
            # For historical consistency with the original Faster R-CNN
            # implementation we are *not* filtering crowd proposals.
            # This choice should be investigated in the future (it likely does
            # not matter).
            json_dataset_add_proposals(roidb, rois, im_scales, crowd_thresh=0)
            roidb_utils_add_bbox_regression_targets(roidb)
            # Compute training labels for the RPN proposals; also handles
            # distributing the proposals over FPN levels
            output_blob_names = fast_rcnn_roi_data_get_fast_rcnn_blob_names()
            blobs = {k: [] for k in output_blob_names}
            fast_rcnn_roi_data_add_fast_rcnn_blobs(blobs, im_scales, roidb)
            for i, k in enumerate(output_blob_names):
                blob_utils_py_op_copy_blob(blobs[k], outputs[i])
        else:
            # For inference we have a special code path that avoids some data
            # loader overhead
            distribute(rois, None, outputs, self._train)

class GenerateProposalLabelsOp(object):

    def forward(self, inputs, outputs):
        """See modeling.detector.GenerateProposalLabels for inputs/outputs
        documentation.
        """
        # During training we reuse the data loader code. We populate roidb
        # entries on the fly using the rois generated by RPN.
        # im_info: [[im_height, im_width, im_scale], ...]
        rois = inputs[0].data
        roidb = blob_utils_deserialize(inputs[1].data)
        im_info = inputs[2].data
        im_scales = im_info[:, 2]
        output_blob_names = fast_rcnn_roi_data_get_fast_rcnn_blob_names()
        # For historical consistency with the original Faster R-CNN
        # implementation we are *not* filtering crowd proposals.
        # This choice should be investigated in the future (it likely does
        # not matter).
        json_dataset_add_proposals(roidb, rois, im_scales, crowd_thresh=0)
        roidb_utils_add_bbox_regression_targets(roidb)
        blobs = {k: [] for k in output_blob_names}
        fast_rcnn_roi_data_add_fast_rcnn_blobs(blobs, im_scales, roidb)
        for i, k in enumerate(output_blob_names):
            blob_utils_py_op_copy_blob(blobs[k], outputs[i])

class GenerateProposalsOp(object):
    """Output object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, anchors, spatial_scale, train):
        self._anchors = anchors
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = 1. / spatial_scale
        self._train = train

    def forward(self, inputs, outputs):
        """See modeling.detector.GenerateProposals for inputs/outputs
        documentation.
        """
        # 1. for each location i in a (H, W) grid:
        #      generate A anchor boxes centered on cell i
        #      apply predicted bbox deltas to each of the A anchors at cell i
        # 2. clip predicted boxes to image
        # 3. remove predicted boxes with either height or width < threshold
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take the top pre_nms_topN proposals before NMS
        # 6. apply NMS with a loose threshold (0.7) to the remaining proposals
        # 7. take after_nms_topN proposals after NMS
        # 8. return the top proposals

        # predicted probability of fg object for each RPN anchor
        scores = inputs[0].data
        # predicted achors transformations
        bbox_deltas = inputs[1].data
        # input image (height, width, scale), in which scale is the scale factor
        # applied to the original dataset image to get the network input image
        im_info = inputs[2].data
        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]
        # Enumerate all shifted positions on the (H, W) grid
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y, copy=False)
        # Convert to (K, 4), K=H*W, where the columns are (dx, dy, dx, dy)
        # shift pointing to each grid location
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Broacast anchors over shifts to enumerate all anchors at all positions
        # in the (H, W) grid:
        #   - add A anchors of shape (1, A, 4) to
        #   - K shifts of shape (K, 1, 4) to get
        #   - all shifted anchors of shape (K, A, 4)
        #   - reshape to (K*A, 4) shifted anchors
        num_images = inputs[0].shape[0]
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = self._anchors[np.newaxis, :, :] + shifts[:, np.newaxis, :]
        all_anchors = all_anchors.reshape((K * A, 4))

        rois = np.empty((0, 5), dtype=np.float32)
        roi_probs = np.empty((0, 1), dtype=np.float32)
        for im_i in range(num_images):
            im_i_boxes, im_i_probs = self.proposals_for_one_image(
                im_info[im_i, :], all_anchors, bbox_deltas[im_i, :, :, :],
                scores[im_i, :, :, :]
            )
            batch_inds = im_i * np.ones(
                (im_i_boxes.shape[0], 1), dtype=np.float32
            )
            im_i_rois = np.hstack((batch_inds, im_i_boxes))
            rois = np.append(rois, im_i_rois, axis=0)
            roi_probs = np.append(roi_probs, im_i_probs, axis=0)

        outputs[0].reshape(rois.shape)
        outputs[0].data[...] = rois
        if len(outputs) > 1:
            outputs[1].reshape(roi_probs.shape)
            outputs[1].data[...] = roi_probs

    def proposals_for_one_image(
        self, im_info, all_anchors, bbox_deltas, scores
    ):
        # Get mode-dependent configuration
        cfg_key = 'TRAIN' if self._train else 'TEST'
        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
        min_size = cfg[cfg_key].RPN_MIN_SIZE
        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #   - bbox deltas will be (4 * A, H, W) format from conv output
        #   - transpose to (H, W, 4 * A)
        #   - reshape to (H * W * A, 4) where rows are ordered by (H, W, A)
        #     in slowest to fastest order to match the enumerated anchors
        bbox_deltas = bbox_deltas.transpose((1, 2, 0)).reshape((-1, 4))

        # Same story for the scores:
        #   - scores are (A, H, W) format from conv output
        #   - transpose to (H, W, A)
        #   - reshape to (H * W * A, 1) where rows are ordered by (H, W, A)
        #     to match the order of anchors and bbox_deltas
        scores = scores.transpose((1, 2, 0)).reshape((-1, 1))

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        if pre_nms_topN <= 0 or pre_nms_topN >= len(scores):
            order = np.argsort(-scores.squeeze())
        else:
            # Avoid sorting possibly large arrays; First partition to get top K
            # unsorted and then sort just those (~20x faster for 200k scores)
            inds = np.argpartition(
                -scores.squeeze(), pre_nms_topN
            )[:pre_nms_topN]
            order = np.argsort(-scores[inds].squeeze())
            order = inds[order]
        bbox_deltas = bbox_deltas[order, :]
        all_anchors = all_anchors[order, :]
        scores = scores[order]

        # Transform anchors into proposals via bbox transformations
        proposals = box_utils_bbox_transform(
            all_anchors, bbox_deltas, (1.0, 1.0, 1.0, 1.0))

        # 2. clip proposals to image (may result in proposals with zero area
        # that will be removed in the next step)
        proposals = box_utils_clip_tiled_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < min_size
        keep = _filter_boxes(proposals, min_size, im_info)
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 6. apply loose nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        if nms_thresh > 0:
            keep = box_utils_nms(np.hstack((proposals, scores)), nms_thresh)
            if post_nms_topN > 0:
                keep = keep[:post_nms_topN]
            proposals = proposals[keep, :]
            scores = scores[keep]
        return proposals, scores

class DetectionModelHelper(cnn.CNNModelHelper):
    def __init__(self, **kwargs):
        # Handle args specific to the DetectionModelHelper, others pass through
        # to CNNModelHelper
        self.train = kwargs.get('train', False)
        self.num_classes = kwargs.get('num_classes', -1)
        assert self.num_classes > 0, 'num_classes must be > 0'
        for k in ('train', 'num_classes'):
            if k in kwargs:
                del kwargs[k]
        kwargs['order'] = 'NCHW'
        # Defensively set cudnn_exhaustive_search to False in case the default
        # changes in CNNModelHelper. The detection code uses variable size
        # inputs that might not play nicely with cudnn_exhaustive_search.
        kwargs['cudnn_exhaustive_search'] = False
        super(DetectionModelHelper, self).__init__(**kwargs)
        self.roi_data_loader = None
        self.losses = []
        self.metrics = []
        self.do_not_update_params = []  # Param on this list are not updated
        self.net.Proto().type = cfg.MODEL.EXECUTION_TYPE
        self.net.Proto().num_workers = cfg.NUM_GPUS * 4
        self.prev_use_cudnn = self.use_cudnn
        self.gn_params = []  # Param on this list are GroupNorm parameters

    def TrainableParams(self, gpu_id=-1):
        """Get the blob names for all trainable parameters, possibly filtered by
        GPU id.
        """
        return [
            p for p in self.params
            if (
                p in self.param_to_grad and   # p has a gradient
                p not in self.do_not_update_params and  # not on the blacklist
                (gpu_id == -1 or  # filter for gpu assignment, if gpu_id set
                 str(p).find('gpu_{}'.format(gpu_id)) == 0)
            )]

    def AffineChannel(self, blob_in, blob_out, dim, inplace=False):
        """Affine transformation to replace BN in networks where BN cannot be
        used (e.g., because the minibatch size is too small).

        The operations can be done in place to save memory.
        """
        blob_out = blob_out or self.net.NextName()
        param_prefix = blob_out

        scale = self.create_param(
            param_name=param_prefix + '_s',
            initializer=initializers.Initializer("ConstantFill", value=1.),
            tags=ParameterTags.WEIGHT,
            shape=[dim, ],
        )
        bias = self.create_param(
            param_name=param_prefix + '_b',
            initializer=initializers.Initializer("ConstantFill", value=0.),
            tags=ParameterTags.BIAS,
            shape=[dim, ],
        )
        if inplace:
            return self.net.AffineChannel([blob_in, scale, bias], blob_in)
        else:
            return self.net.AffineChannel([blob_in, scale, bias], blob_out)

    def GenerateProposals(self, blobs_in, blobs_out, anchors, spatial_scale):
        """Op for generating RPN porposals.

        blobs_in:
          - 'rpn_cls_probs': 4D tensor of shape (N, A, H, W), where N is the
            number of minibatch images, A is the number of anchors per
            locations, and (H, W) is the spatial size of the prediction grid.
            Each value represents a "probability of object" rating in [0, 1].
          - 'rpn_bbox_pred': 4D tensor of shape (N, 4 * A, H, W) of predicted
            deltas for transformation anchor boxes into RPN proposals.
          - 'im_info': 2D tensor of shape (N, 3) where the three columns encode
            the input image's [height, width, scale]. Height and width are
            for the input to the network, not the original image; scale is the
            scale factor used to scale the original image to the network input
            size.

        blobs_out:
          - 'rpn_rois': 2D tensor of shape (R, 5), for R RPN proposals where the
            five columns encode [batch ind, x1, y1, x2, y2]. The boxes are
            w.r.t. the network input, which is a *scaled* version of the
            original image; these proposals must be scaled by 1 / scale (where
            scale comes from im_info; see above) to transform it back to the
            original input image coordinate system.
          - 'rpn_roi_probs': 1D tensor of objectness probability scores
            (extracted from rpn_cls_probs; see above).
        """
        name = 'GenerateProposalsOp:' + ','.join([str(b) for b in blobs_in])
        # spatial_scale passed to the Python op is only used in convert_pkl_to_pb
        self.net.Python(
            GenerateProposalsOp(anchors, spatial_scale, self.train).forward
        )(blobs_in, blobs_out, name=name, spatial_scale=spatial_scale)
        return blobs_out

    def GenerateProposalLabels(self, blobs_in):
        """Op for generating training labels for RPN proposals. This is used
        when training RPN jointly with Fast/Mask R-CNN (as in end-to-end
        Faster R-CNN training).

        blobs_in:
          - 'rpn_rois': 2D tensor of RPN proposals output by GenerateProposals
          - 'roidb': roidb entries that will be labeled
          - 'im_info': See GenerateProposals doc.

        blobs_out:
          - (variable set of blobs): returns whatever blobs are required for
            training the model. It does this by querying the data loader for
            the list of blobs that are needed.
        """
        name = 'GenerateProposalLabelsOp:' + ','.join(
            [str(b) for b in blobs_in]
        )

        # The list of blobs is not known before run-time because it depends on
        # the specific model being trained. Query the data loader to get the
        # list of output blob names.
        blobs_out = fast_rcnn_roi_data_get_fast_rcnn_blob_names(
            is_training=self.train
        )
        blobs_out = [core.ScopedBlobReference(b) for b in blobs_out]

        self.net.Python(GenerateProposalLabelsOp().forward)(
            blobs_in, blobs_out, name=name
        )
        return blobs_out

    def CollectAndDistributeFpnRpnProposals(self):
        """Merge RPN proposals generated at multiple FPN levels and then
        distribute those proposals to their appropriate FPN levels. An anchor
        at one FPN level may predict an RoI that will map to another level,
        hence the need to redistribute the proposals.

        This function assumes standard blob names for input and output blobs.

        Input blobs: [rpn_rois_fpn<min>, ..., rpn_rois_fpn<max>,
                      rpn_roi_probs_fpn<min>, ..., rpn_roi_probs_fpn<max>]
          - rpn_rois_fpn<i> are the RPN proposals for FPN level i; see rpn_rois
            documentation from GenerateProposals.
          - rpn_roi_probs_fpn<i> are the RPN objectness probabilities for FPN
            level i; see rpn_roi_probs documentation from GenerateProposals.

        If used during training, then the input blobs will also include:
          [roidb, im_info] (see GenerateProposalLabels).

        Output blobs: [rois_fpn<min>, ..., rois_rpn<max>, rois,
                       rois_idx_restore]
          - rois_fpn<i> are the RPN proposals for FPN level i
          - rois_idx_restore is a permutation on the concatenation of all
            rois_fpn<i>, i=min...max, such that when applied the RPN RoIs are
            restored to their original order in the input blobs.

        If used during training, then the output blobs will also include:
          [labels, bbox_targets, bbox_inside_weights, bbox_outside_weights].
        """
        k_max = cfg.FPN.RPN_MAX_LEVEL
        k_min = cfg.FPN.RPN_MIN_LEVEL

        # Prepare input blobs
        rois_names = ['rpn_rois_fpn' + str(l) for l in range(k_min, k_max + 1)]
        score_names = [
            'rpn_roi_probs_fpn' + str(l) for l in range(k_min, k_max + 1)
        ]
        blobs_in = rois_names + score_names
        if self.train:
            blobs_in += ['roidb', 'im_info']
        blobs_in = [core.ScopedBlobReference(b) for b in blobs_in]
        name = 'CollectAndDistributeFpnRpnProposalsOp:' + ','.join(
            [str(b) for b in blobs_in]
        )

        # Prepare output blobs
        blobs_out = fast_rcnn_roi_data_get_fast_rcnn_blob_names(
            is_training=self.train
        )
        blobs_out = [core.ScopedBlobReference(b) for b in blobs_out]

        outputs = self.net.Python(
            CollectAndDistributeFpnRpnProposalsOp(self.train).forward
        )(blobs_in, blobs_out, name=name)

        return outputs

    def DropoutIfTraining(self, blob_in, dropout_rate):
        """Add dropout to blob_in if the model is in training mode and
        dropout_rate is > 0."""
        blob_out = blob_in
        if self.train and dropout_rate > 0:
            blob_out = self.Dropout(
                blob_in, blob_in, ratio=dropout_rate, is_test=False
            )
        return blob_out

    def RoIFeatureTransform(
        self,
        blobs_in,
        blob_out,
        blob_rois='rois',
        method='RoIPoolF',
        resolution=7,
        spatial_scale=1. / 16.,
        sampling_ratio=0
    ):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)
        has_argmax = (method == 'RoIPoolF')
        if isinstance(blobs_in, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                bl_rois = blob_rois + '_fpn' + str(lvl)
                bl_out = blob_out + '_fpn' + str(lvl)
                bl_out_list.append(bl_out)
                bl_argmax = ['_argmax_' + bl_out] if has_argmax else []
                self.net.__getattr__(method)(
                    [bl_in, bl_rois], [bl_out] + bl_argmax,
                    pooled_w=resolution,
                    pooled_h=resolution,
                    spatial_scale=sc,
                    sampling_ratio=sampling_ratio
                )
            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled, _ = self.net.Concat(
                bl_out_list, [blob_out + '_shuffled', '_concat_' + blob_out],
                axis=0
            )
            # Unshuffle to match rois from dataloader
            restore_bl = blob_rois + '_idx_restore_int32'
            xform_out = self.net.BatchPermutation(
                [xform_shuffled, restore_bl], blob_out
            )
        else:
            # Single feature level
            bl_argmax = ['_argmax_' + blob_out] if has_argmax else []
            # sampling_ratio is ignored for RoIPoolF
            xform_out = self.net.__getattr__(method)(
                [blobs_in, blob_rois], [blob_out] + bl_argmax,
                pooled_w=resolution,
                pooled_h=resolution,
                spatial_scale=spatial_scale,
                sampling_ratio=sampling_ratio
            )
        # Only return the first blob (the transformed features)
        return xform_out[0] if isinstance(xform_out, tuple) else xform_out

    def ConvShared(
        self,
        blob_in,
        blob_out,
        dim_in,
        dim_out,
        kernel,
        weight=None,
        bias=None,
        **kwargs
    ):
        """Add conv op that shares weights and/or biases with another conv op.
        """
        use_bias = (
            False if ('no_bias' in kwargs and kwargs['no_bias']) else True
        )

        if self.use_cudnn:
            kwargs['engine'] = 'CUDNN'
            kwargs['exhaustive_search'] = self.cudnn_exhaustive_search
            if self.ws_nbytes_limit:
                kwargs['ws_nbytes_limit'] = self.ws_nbytes_limit

        if use_bias:
            blobs_in = [blob_in, weight, bias]
        else:
            blobs_in = [blob_in, weight]

        if 'no_bias' in kwargs:
            del kwargs['no_bias']

        return self.net.Conv(
            blobs_in, blob_out, kernel=kernel, order=self.order, **kwargs
        )

    def BilinearInterpolation(
        self, blob_in, blob_out, dim_in, dim_out, up_scale
    ):
        """Bilinear interpolation in space of scale.

        Takes input of NxKxHxW and outputs NxKx(sH)x(sW), where s:= up_scale

        Adapted from the CVPR'15 FCN code.
        See: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
        """
        assert dim_in == dim_out
        assert up_scale % 2 == 0, 'Scale should be even'

        def upsample_filt(size):
            factor = (size + 1) // 2
            if size % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            og = np.ogrid[:size, :size]
            return ((1 - abs(og[0] - center) / factor) *
                    (1 - abs(og[1] - center) / factor))

        kernel_size = up_scale * 2
        bil_filt = upsample_filt(kernel_size)

        kernel = np.zeros(
            (dim_in, dim_out, kernel_size, kernel_size), dtype=np.float32
        )
        kernel[range(dim_out), range(dim_in), :, :] = bil_filt

        blob = self.ConvTranspose(
            blob_in,
            blob_out,
            dim_in,
            dim_out,
            kernel_size,
            stride=int(up_scale),
            pad=int(up_scale / 2),
            weight_init=('GivenTensorFill', {'values': kernel}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        self.do_not_update_params.append(self.weights[-1])
        self.do_not_update_params.append(self.biases[-1])
        return blob

    def ConvAffine(  # args in the same order of Conv()
        self, blob_in, prefix, dim_in, dim_out, kernel, stride, pad,
        group=1, dilation=1,
        weight_init=None,
        bias_init=None,
        suffix='_bn',
        inplace=False
    ):
        """ConvAffine adds a Conv op followed by a AffineChannel op (which
        replaces BN during fine tuning).
        """
        conv_blob = self.Conv(
            blob_in,
            prefix,
            dim_in,
            dim_out,
            kernel,
            stride=stride,
            pad=pad,
            group=group,
            dilation=dilation,
            weight_init=weight_init,
            bias_init=bias_init,
            no_bias=1
        )
        blob_out = self.AffineChannel(
            conv_blob, prefix + suffix, dim=dim_out, inplace=inplace
        )
        return blob_out

    def ConvGN(  # args in the same order of Conv()
        self, blob_in, prefix, dim_in, dim_out, kernel, stride, pad,
        group_gn,  # num of groups in gn
        group=1, dilation=1,
        weight_init=None,
        bias_init=None,
        suffix='_gn',
        no_conv_bias=1,
    ):
        """ConvGN adds a Conv op followed by a GroupNorm op,
        including learnable scale/bias (gamma/beta)
        """
        conv_blob = self.Conv(
            blob_in,
            prefix,
            dim_in,
            dim_out,
            kernel,
            stride=stride,
            pad=pad,
            group=group,
            dilation=dilation,
            weight_init=weight_init,
            bias_init=bias_init,
            no_bias=no_conv_bias)

        if group_gn < 1:
            logger.warning(
                'Layer: {} (dim {}): '
                'group_gn < 1; reset to 1.'.format(prefix, dim_in)
            )
            group_gn = 1

        blob_out = self.SpatialGN(
            conv_blob, prefix + suffix,
            dim_out, num_groups=group_gn,
            epsilon=cfg.GROUP_NORM.EPSILON,)

        self.gn_params.append(self.params[-1])  # add gn's bias to list
        self.gn_params.append(self.params[-2])  # add gn's scale to list
        return blob_out

    def DisableCudnn(self):
        self.prev_use_cudnn = self.use_cudnn
        self.use_cudnn = False

    def RestorePreviousUseCudnn(self):
        prev_use_cudnn = self.use_cudnn
        self.use_cudnn = self.prev_use_cudnn
        self.prev_use_cudnn = prev_use_cudnn

    def UpdateWorkspaceLr(self, cur_iter, new_lr):
        """Updates the model's current learning rate and the workspace (learning
        rate and update history/momentum blobs).
        """
        # The workspace is the one source of truth for the lr
        # The lr is always the same on all GPUs
        cur_lr = workspace.FetchBlob('gpu_0/lr')[0]
        # There are no type conversions between the lr in Python and the lr in
        # the GPU (both are float32), so exact comparision is ok
        if cur_lr != new_lr:
            ratio = _get_lr_change_ratio(cur_lr, new_lr)
            self._SetNewLr(cur_lr, new_lr)
        return new_lr

    def _SetNewLr(self, cur_lr, new_lr):
        """Do the actual work of updating the model and workspace blobs.
        """
        for i in range(cfg.NUM_GPUS):
            with c2_utils_CudaScope(i):
                workspace.FeedBlob(
                    'gpu_{}/lr'.format(i), np.array([new_lr], dtype=np.float32))
        ratio = _get_lr_change_ratio(cur_lr, new_lr)
        if cfg.SOLVER.SCALE_MOMENTUM and cur_lr > 1e-7 and \
                ratio > cfg.SOLVER.SCALE_MOMENTUM_THRESHOLD:
            self._CorrectMomentum(new_lr / cur_lr)

    def _CorrectMomentum(self, correction):
        """The MomentumSGDUpdate op implements the update V as

            V := mu * V + lr * grad,

        where mu is the momentum factor, lr is the learning rate, and grad is
        the stochastic gradient. Since V is not defined independently of the
        learning rate (as it should ideally be), when the learning rate is
        changed we should scale the update history V in order to make it
        compatible in scale with lr * grad.
        """
        for i in range(cfg.NUM_GPUS):
            with c2_utils_CudaScope(i):
                for param in self.TrainableParams(gpu_id=i):
                    op = core.CreateOperator(
                        'Scale', [param + '_momentum'], [param + '_momentum'],
                        scale=correction)
                    workspace.RunOperatorOnce(op)

    def GetLossScale(self):
        """Allow a way to configure the loss scale dynamically.

        This may be used in a distributed data parallel setting.
        """
        return 1.0 / cfg.NUM_GPUS

    def AddLosses(self, losses):
        if not isinstance(losses, list):
            losses = [losses]
        # Conversion to str allows losses to include BlobReferences
        losses = [c2_utils_UnscopeName(str(l)) for l in losses]
        self.losses = list(set(self.losses + losses))

    def AddMetrics(self, metrics):
        if not isinstance(metrics, list):
            metrics = [metrics]
        self.metrics = list(set(self.metrics + metrics))

class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.reset()

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def reset(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

class AttrDict(dict):

    IMMUTABLE = '__immutable__'

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not self.__dict__[AttrDict.IMMUTABLE]:
            if name in self.__dict__:
                self.__dict__[name] = value
            else:
                self[name] = value
        else:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but AttrDict is immutable'.
                format(name, value)
            )

    def immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested AttrDicts.
        """
        self.__dict__[AttrDict.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)
        for v in self.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]

c2_utils_import_detectron_ops()
cv2.ocl.setUseOpenCL(False)

DP = DensePoseMethods()

def main(args_im_or_folder, args_cfg, args_output_dir, args_image_ext, args_weights):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args_cfg)
    cfg.NUM_GPUS = 1
    args_weights = cache_url(args_weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine_initialize_model_from_cfg(args_weights)
    dummy_coco_dataset = dummy_datasets_get_coco_dataset()

    if os.path.isdir(args_im_or_folder):
        im_list = glob.iglob(args_im_or_folder + '/*.' + args_image_ext)
    else:
        im_list = [args_im_or_folder]

    for i, im_name in enumerate(im_list):
        out_name = os.path.join(
            args_output_dir, '{}'.format(os.path.basename(im_name) + '.pdf')
        )
        #logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils_NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine_im_detect_all(
                model, im, None, timers=timers
            )

        vis_utils_vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args_output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            cls_bodys,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2
        )

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    main('data/', 'assets/config.yaml', 
        'output/', 'jpg', 
        'https://dl.fbaipublicfiles.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl')
