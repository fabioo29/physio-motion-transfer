```
    MESA_GL_VERSION_OVERRIDE=4.1 CUDA_VISIBLE_DEVICES=0 python main.py
```

BN_MOMENTUM = 0.1

VERTEX_IDS = {
    'smplh': {
        'nose':         332,
        'reye':         6260,
        'leye':         2800,
        'rear':         4071,
        'lear':         583,
        'rthumb':       6191,
        'rindex':       5782,
        'rmiddle':      5905,
        'rring':        6016,
        'rpinky':       6133,
        'lthumb':       2746,
        'lindex':       2319,
        'lmiddle':      2445,
        'lring':        2556,
        'lpinky':       2673,
        'LBigToe':      3216,
        'LSmallToe':    3226,
        'LHeel':        3387,
        'RBigToe':      6617,
        'RSmallToe':    6624,
        'RHeel':        6787
    },
    'smplx': {
        'nose':         9120,
        'reye':         9929,
        'leye':         9448,
        'rear':         616,
        'lear':         6,
        'rthumb':       8079,
        'rindex':       7669,
        'rmiddle':      7794,
        'rring':        7905,
        'rpinky':       8022,
        'lthumb':       5361,
        'lindex':       4933,
        'lmiddle':      5058,
        'lring':        5169,
        'lpinky':       5286,
        'LBigToe':      5770,
        'LSmallToe':    5780,
        'LHeel':        8846,
        'RBigToe':      8463,
        'RSmallToe':    8474,
        'RHeel':        8635
    }
}

import imageio, cv2, sys, logging
import os, keyboard, torchvision
import pyrender, trimesh, math
from pyrender.constants import RenderFlags
import os.path as osp
import numpy as np
from PIL import Image
from collections import namedtuple
import glob, random
sys.path.append('.')

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from assets import config
from assets import constants
from assets.config import args

if sys.version_info[0] == 3:
    import _pickle as pkl
else:
    import cPickle as pkl

os.environ['PYOPENGL_PLATFORM'] = 'egl'

ModelOutput = namedtuple('ModelOutput',
                         ['vertices', 'joints','joints_h36m17', 'joints_smpl24','full_pose', 'betas',
                          'global_orient','body_pose', 'expression','jaw_pose',
                          'left_hand_pose', 'right_hand_pose'])
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)

def build_model():
    if args.backbone in Backbones:
        backbone = Backbones[args.backbone]()
    else:
        raise "Backbone not recognized!!"
    if args.model_version in Heads:
        ROMP = Heads[args.model_version]
    else:
        raise "Head not recognized!!"
    model = ROMP(backbone=backbone)
    return model

def copy_state_dict(cur_state_dict, pre_state_dict, prefix = 'module.', drop_prefix='', \
    ignore_layer='_result_parser.params_map_parser.smpl_model',fix_loaded=False):
    success_layers, failed_layers = [], []
    def _get_params(key):
        key = key.replace(drop_prefix,'')
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None

    for k in cur_state_dict.keys():
        v = _get_params(k)
        if v is None:
            if ignore_layer not in k:
                failed_layers.append(k)
            continue
        cur_state_dict[k].copy_(v)
        if prefix in k and prefix!='':
            k=k.split(prefix)[1]
        success_layers.append(k)
    if len(failed_layers)>0:
        logging.info('missing parameters of layers:{}'.format(failed_layers))

    if fix_loaded and len(failed_layers)>0:
        print('fixing the layers that were loaded successfully, while train the layers that failed,')
        for k in cur_state_dict.keys():
            if k in success_layers:
                cur_state_dict[k].requires_grad=False

    return success_layers

def load_model(path, model, prefix = 'module.', drop_prefix='',optimizer=None, **kwargs):
    logging.info('using fine_tune model: {}'.format(path))
    if os.path.exists(path):
        pretrained_model = torch.load(path, map_location=torch.device('cpu'))
        current_model = model.state_dict()
        if isinstance(pretrained_model, dict):
            if 'model_state_dict' in pretrained_model:
                pretrained_model = pretrained_model['model_state_dict']
        copy_state_dict(current_model, pretrained_model, prefix = prefix, drop_prefix=drop_prefix, **kwargs)
    else:
        logging.warning('model {} not exist!'.format(path))
    return model

def get_remove_keys(dt, keys=[]):
    targets = []
    for key in keys:
        targets.append(dt[key])
    for key in keys:
        del dt[key]
    return targets

def reorganize_items(items, reorganize_idx):
    items_new = [[] for _ in range(len(items))]
    for idx, item in enumerate(items):
        for ridx in reorganize_idx:
            items_new[idx].append(item[ridx])
    return items_new

def save_obj(verts, faces, obj_mesh_name='mesh.obj'):
    #print('Saving:',obj_mesh_name)
    with open(obj_mesh_name, 'w') as fp:
        for v in verts:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

        for f in faces: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0] + 1, f[1] + 1, f[2] + 1) )
            
def save_meshes(reorganize_idx, outputs, output_dir, smpl_faces):
    vids_org = np.unique(reorganize_idx)
    for idx, vid in enumerate(vids_org):
        verts_vids = np.where(reorganize_idx==vid)[0]
        img_path = outputs['meta_data']['imgpath'][verts_vids[0]]
        obj_name = os.path.join(output_dir, '{}'.format(os.path.basename(img_path))).replace('.mp4','').replace('.jpg','').replace('.png','')+'.obj'
        for subject_idx, batch_idx in enumerate(verts_vids):
            save_obj(outputs['verts'][batch_idx].detach().cpu().numpy().astype(np.float16), \
                smpl_faces,obj_name.replace('.obj', '_{}.obj'.format(subject_idx)))

def convert_cam_to_3d_trans(cams, weight=2.):
    trans3d = []
    (s, tx, ty) = cams
    depth, dx, dy = 1./s, tx/s, ty/s
    trans3d = np.array([dx, dy, depth])*weight
    return trans3d

def img_preprocess(image, imgpath, input_size=512, ds='internet', single_img_input=False):
    image = image[:,:,::-1]
    image_size = image.shape[:2][::-1]
    image_org = Image.fromarray(image)
    
    resized_image_size = (float(input_size)/max(image_size) * np.array(image_size) // 2 * 2).astype(np.int)
    padding = tuple((input_size-resized_image_size)//2)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([resized_image_size[1],resized_image_size[0]], interpolation=3),
        torchvision.transforms.Pad(padding, fill=0, padding_mode='constant'),
        #torchvision.transforms.ToTensor(),
        ])
    image = torch.from_numpy(np.array(transform(image_org))).float()

    padding_org = tuple((max(image_size)-np.array(image_size))//2)
    transform_org = torchvision.transforms.Compose([
        torchvision.transforms.Pad(padding_org, fill=0, padding_mode='constant'),
        torchvision.transforms.Resize((input_size*2, input_size*2), interpolation=3), #max(image_size)//2,max(image_size)//2
        #torchvision.transforms.ToTensor(),
        ])
    image_org = torch.from_numpy(np.array(transform_org(image_org)))
    padding_org = (np.array(list(padding_org))*float(input_size*2/max(image_size))).astype(np.int)
    if padding_org[0]>0:
        image_org[:,:padding_org[0]] = 255 
        image_org[:,-padding_org[0]:] = 255
    if padding_org[1]>0:
        image_org[:padding_org[1]] = 255 
        image_org[-padding_org[1]:] = 255 

    offsets = np.array([image_size[1],image_size[0],resized_image_size[0],\
        resized_image_size[0]+padding[1],resized_image_size[1],resized_image_size[1]+padding[0],padding[1],\
        resized_image_size[0],padding[0],resized_image_size[1]],dtype=np.int)
    offsets = torch.from_numpy(offsets).float()

    name = os.path.basename(imgpath)

    if single_img_input:
        image = image.unsqueeze(0).contiguous()
        image_org = image_org.unsqueeze(0).contiguous()
        offsets = offsets.unsqueeze(0).contiguous()
        imgpath, name, ds = [imgpath], [name], [ds]
    input_data = {
        'image': image,
        'image_org': image_org,
        'imgpath': imgpath,
        'offsets': offsets,
        'name': name,
        'data_set':ds }
    return input_data

def get_video_bn(video_file_path):
    return os.path.basename(video_file_path)\
    .replace('.mp4', '').replace('.avi', '').replace('.webm', '').replace('.gif', '')

def frames2video(images, video_name,fps=30):
    writer = imageio.get_writer(video_name, format='mp4', mode='I', fps=fps)

    for image in images:
        writer.append_data(image)
    writer.close()

def get_renderer(test=False,resolution = (512,512,3),part_segment=False):
    faces = pkl.load(open(os.path.join(args.smpl_model_path,'smpl','SMPL_NEUTRAL.pkl'),'rb'), encoding='latin1')['f']
    renderer = Renderer(faces,resolution=resolution[:2])
    
    return renderer

def make_heatmaps(image, heatmaps):
    heatmaps = heatmaps.mul(255)\
                       .clamp(0, 255)\
                       .byte()\
                       .detach().cpu().numpy()

    num_joints, height, width = heatmaps.shape
    image_resized = cv2.resize(image, (int(width), int(height)))

    image_grid = np.zeros((height, (num_joints+1)*width, 3), dtype=np.uint8)

    for j in range(num_joints):
        # add_joints(image_resized, joints[:, j, :])
        heatmap = heatmaps[j, :, :]
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        image_fused = colored_heatmap*0.7 + image_resized*0.3

        width_begin = width * (j+1)
        width_end = width * (j+2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image_resized

    return image_grid

def draw_skeleton(image, pts, bones=None, cm=None,put_text=False,r=3):
    for i,pt in enumerate(pts):
        if len(pt)>1:
            if pt[0]>0 and pt[1]>0:
                image = cv2.circle(image,(int(pt[0]), int(pt[1])),r,(255,0,0),-1)
                if put_text:
                    img=cv2.putText(image,str(i),(int(pt[0]), int(pt[1])),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),1)

    if cm is None:
        set_colors = np.array([[255,0,0] for i in range(len(bones))]).astype(np.int)
    else:
        if len(bones)>len(cm):
            cm = np.concatenate([cm for _ in range(len(bones)//len(cm)+1)],0)
        set_colors = cm[:len(bones)].astype(np.int)
    bones = np.concatenate([bones,set_colors],1).tolist()

    for line in bones:
        pa = pts[line[0]]
        pb = pts[line[1]]
        if (pa>0).all() and (pb>0).all():
            xa,ya,xb,yb = int(pa[0]),int(pa[1]),int(pb[0]),int(pb[1])
            image = cv2.line(image,(xa,ya),(xb,yb),(int(line[2]), int(line[3]), int(line[4])),r)
    return image

def draw_skeleton_multiperson(image, pts_group,**kwargs):
    for pts in pts_group:
        image = draw_skeleton(image, pts, **kwargs)
    return image

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def BHWC_to_BCHW(x):
    """
    :param x: torch tensor, B x H x W x C
    :return:  torch tensor, B x C x H x W
    """
    return x.unsqueeze(1).transpose(1, -1).squeeze(-1)

def process_gt_center(center_normed):
    center_list = []
    valid_mask = center_normed[:,:,0]>-1
    valid_inds = torch.where(valid_mask)
    valid_batch_inds, valid_person_ids = valid_inds[0], valid_inds[1]
    center_gt = ((center_normed+1)/2*args.centermap_size).long()
    center_gt_valid = center_gt[valid_mask]
    return (valid_batch_inds, valid_person_ids, center_gt_valid)

def get_coord_maps(size=128):
    xx_ones = torch.ones([1, size], dtype=torch.int32)
    xx_ones = xx_ones.unsqueeze(-1)

    xx_range = torch.arange(size, dtype=torch.int32).unsqueeze(0)
    xx_range = xx_range.unsqueeze(1)

    xx_channel = torch.matmul(xx_ones, xx_range)
    xx_channel = xx_channel.unsqueeze(-1)

    yy_ones = torch.ones([1, size], dtype=torch.int32)
    yy_ones = yy_ones.unsqueeze(1)

    yy_range = torch.arange(size, dtype=torch.int32).unsqueeze(0)
    yy_range = yy_range.unsqueeze(-1)

    yy_channel = torch.matmul(yy_range, yy_ones)
    yy_channel = yy_channel.unsqueeze(-1)

    xx_channel = xx_channel.permute(0, 3, 1, 2)
    yy_channel = yy_channel.permute(0, 3, 1, 2)

    xx_channel = xx_channel.float() / (size - 1)
    yy_channel = yy_channel.float() / (size - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    out = torch.cat([xx_channel, yy_channel], dim=1)
    return out

def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap

def nms(det, pool_func=None):
    maxm = pool_func(det)
    maxm = torch.eq(maxm, det).float()
    det = det * maxm
    return det

def batch_orth_proj(X, camera, mode='2d',keep_dim=False):
    camera = camera.view(-1, 1, 3)
    X_camed = X[:,:,:2] * camera[:, :, 0].unsqueeze(-1)
    X_camed += camera[:, :, 1:]
    if keep_dim:
        X_camed = torch.cat([X_camed, X[:,:,2].unsqueeze(-1)],-1)
    return X_camed

def vertices_kp3d_projection(outputs):
    params_dict, vertices, j3ds = outputs['params'], outputs['verts'], outputs['j3d']
    verts_camed = batch_orth_proj(vertices, params_dict['cam'], mode='3d',keep_dim=True)
    pj3d = batch_orth_proj(j3ds, params_dict['cam'], mode='2d')
    projected_outputs = {'verts_camed': verts_camed, 'pj2d': pj3d[:,:,:2]}
    return projected_outputs

def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q

def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis

def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3,3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32,
                           device=rotation_matrix.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa

def rot6d_to_rotmat(x):
    x = x.view(-1,3,2)

    # Normalize the first vector
    b1 = F.normalize(x[:, :, 0], dim=1, eps=1e-6)

    dot_prod = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
    # Compute the second vector by finding the orthogonal complement to it
    b2 = F.normalize(x[:, :, 1] - dot_prod * b1, dim=-1, eps=1e-6)

    # Finish building the basis by taking the cross product
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)

    return rot_mats
    
def rot6D_to_angular(rot6D):
    batch_size = rot6D.shape[0]
    pred_rotmat = rot6d_to_rotmat(rot6D).view(batch_size, -1, 3, 3)
    pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(batch_size, -1)
    return pose

def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)

def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)

def blend_shapes(betas, shape_disps):
    ''' Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape

def vertices2joints(J_regressor, vertices):
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    把四元组的系数转化成旋转矩阵。四元组表示三维旋转
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)

def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    #print(rot_mats.shape, rel_joints.shape,)
    transforms_mat = transform_mat(
        rot_mats.contiguous().view(-1, 3, 3),
        rel_joints.contiguous().view(-1, 3, 1)).contiguous().view(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms

def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents,
        lbs_weights, pose2rot=True, dtype=torch.float32):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(
            pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature, posedirs) \
            .view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed

def get_image_cut_box(leftTop, rightBottom, ExpandsRatio, Center = None):
    try:
        l = len(ExpandsRatio)
    except:
        ExpandsRatio = [ExpandsRatio, ExpandsRatio, ExpandsRatio, ExpandsRatio]

    def _expand_crop_box(lt, rb, scale):
        center = (lt + rb) / 2.0
        xl, xr, yt, yb = lt[0] - center[0], rb[0] - center[0], lt[1] - center[1], rb[1] - center[1]

        xl, xr, yt, yb = xl * scale[0], xr * scale[1], yt * scale[2], yb * scale[3]
        #expand it
        lt, rb = np.array([center[0] + xl, center[1] + yt]), np.array([center[0] + xr, center[1] + yb])
        lb, rt = np.array([center[0] + xl, center[1] + yb]), np.array([center[0] + xr, center[1] + yt])
        center = (lt + rb) / 2
        return center, lt, rt, rb, lb

    if Center == None:
        Center = (leftTop + rightBottom) // 2

    Center, leftTop, rightTop, rightBottom, leftBottom = _expand_crop_box(leftTop, rightBottom, ExpandsRatio)

    #把包围盒全弄成正方形的样子，尽量保证形状不发生变化
    offset = (rightBottom - leftTop) // 2

    cx = offset[0]
    cy = offset[1]

    r = max(cx, cy)

    cx = r
    cy = r

    x = int(Center[0])
    y = int(Center[1])

    return [x - cx, y - cy], [x + cx, y + cy]

def off_set_pts(keyPoints, leftTop):
    result = keyPoints.copy()
    result[:, 0] -= leftTop[0]#-offset[0]
    result[:, 1] -= leftTop[1]#-offset[1]
    return result

def process_image(originImage, full_kps):
    height       = originImage.shape[0]
    width        = originImage.shape[1]
    
    original_shape = originImage.shape
    channels     = originImage.shape[2] if len(originImage.shape) >= 3 else 1
    scale = 1.
    leftTop = np.array([0.,0.])
    rightBottom = np.array([width,height],dtype=np.float32)
    leftTop, rightBottom = get_image_cut_box(leftTop, rightBottom, scale)

    lt = [int(leftTop[0]), int(leftTop[1])]
    rb = [int(rightBottom[0]), int(rightBottom[1])]

    lt[0] = max(0, lt[0])
    lt[1] = max(0, lt[1])
    rb[0] = min(rb[0], width)
    rb[1] = min(rb[1], height)

    leftTop      = np.array([int(leftTop[0]), int(leftTop[1])])
    rightBottom  = np.array([int(rightBottom[0] + 0.5), int(rightBottom[1] + 0.5)])

    length = max(rightBottom[1] - leftTop[1]+1, rightBottom[0] - leftTop[0]+1)

    dstImage = np.zeros(shape = [length,length,channels], dtype = np.uint8)
    orgImage_white_bg = np.ones(shape = [length,length,channels], dtype = np.uint8)*255
    offset = np.array([lt[0] - leftTop[0], lt[1] - leftTop[1]])
    size   = [rb[0] - lt[0], rb[1] - lt[1]]
    dstImage[offset[1]:size[1] + offset[1], offset[0]:size[0] + offset[0], :] = originImage[lt[1]:rb[1], lt[0]:rb[0],:]
    orgImage_white_bg[offset[1]:size[1] + offset[1], offset[0]:size[0] + offset[0], :] = originImage[lt[1]:rb[1], lt[0]:rb[0],:]

    return dstImage, orgImage_white_bg, [off_set_pts(kps_i, leftTop) for kps_i in full_kps],(offset,lt,rb,size,original_shape[:2])

def smpl_model_create(model_path, model_type='smpl',
           **kwargs):
    ''' Method for creating a model from a path and a model type

        Parameters
        ----------
        model_path: str
            Either the path to the model you wish to load or a folder,
            where each subfolder contains the differents types, i.e.:
            model_path:
            |
            |-- smpl
                |-- SMPL_FEMALE
                |-- SMPL_NEUTRAL
                |-- SMPL_MALE
            |-- smplh
                |-- SMPLH_FEMALE
                |-- SMPLH_MALE
            |-- smplx
                |-- SMPLX_FEMALE
                |-- SMPLX_NEUTRAL
                |-- SMPLX_MALE
        model_type: str, optional
            When model_path is a folder, then this parameter specifies  the
            type of model to be loaded
        **kwargs: dict
            Keyword arguments

        Returns
        -------
            body_model: nn.Module
                The PyTorch module that implements the corresponding body model
        Raises
        ------
            ValueError: In case the model type is not one of SMPL, SMPLH or
            SMPLX
    '''

    # If it's a folder, assume
    if osp.isdir(model_path):
        model_path = os.path.join(model_path, model_type)

    if model_type.lower() == 'smpl':
        return SMPL(model_path, **kwargs)
    else:
        raise ValueError('Unknown model type {}, exiting!'.format(model_type))

class IBN_a(nn.Module):
    def __init__(self, planes, momentum=BN_MOMENTUM):
        super(IBN_a, self).__init__()
        half1 = int(planes/2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2, momentum=momentum)
    
    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock_IBN_a(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_IBN_a, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = IBN_a(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, BN=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BN(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P
        
class Renderer:
    def __init__(self, faces, resolution=(224,224), orig_img=False, wireframe=False):
        self.resolution = resolution

        self.faces = faces
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0)

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        #light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=5)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)

    def __call__(self, verts, cam=[1,1,0,0], angle=None, axis=None, mesh_filename=None, color=[1.0, 1.0, 0.9]):

        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)

        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)

        if mesh_filename is not None:
            mesh.export(mesh_filename)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        sx, sy, tx, ty = cam

        camera = WeakPerspectiveCamera(
            scale=[sx, sy],
            translation=[tx, ty],
            zfar=1000.
        )

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        image, _ = self.renderer.render(self.scene, flags=render_flags)
        #valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        #output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img
        #image = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return image

class Visualizer(object):
    def __init__(self,resolution = (512,512,3),input_size=512,result_img_dir = None,with_renderer=False):
        self.input_size = input_size
        if with_renderer:
            self.renderer = get_renderer(resolution=resolution)
        self.result_img_dir = result_img_dir
        self.heatmap_kpnum = 17
        self.vis_size = resolution[:2]
        self.mesh_color = (torch.Tensor([[[0.65098039, 0.74117647, 0.85882353]]])*255).long()

    def add_mesh_to_writer(self,writer, verts, name):
        writer.add_mesh(name, vertices=verts.detach().cpu(), \
            colors=self.mesh_color.repeat(verts.shape[0],verts.shape[1],1), \
            faces=self.renderer.faces.cpu(),global_step=self.global_count)

    def visualize_renderer(self, verts, images=None, reorganize_idx=None,thresh=0.2,visible_weight=0.9, scale_thresh=100):
        verts = verts.detach().cpu().numpy()
        renders = []
        for vert in verts:
            render_result = self.renderer(vert, color=[.9, .9, .8])
            renders.append(render_result)
        renders = np.array(renders)
        
        # sorting the render via the rendering area size
        if reorganize_idx is not None:
            renders_summoned = []
            for idxs in reorganize_idx:
                main_renders = renders[idxs[0]]
                main_render_mask = main_renders[:, :, -1] > thresh
                render_scale_map = np.zeros(self.vis_size)
                render_scale_map[main_render_mask] = main_render_mask.sum()
                for jdx in range(1,len(idxs)):
                    other_idx = idxs[jdx]
                    other_renders = renders[other_idx]
                    other_render_mask = other_renders[:, :, -1] > thresh
                    render_scale_map_other = np.zeros(self.vis_size)
                    render_scale_map_other[other_render_mask] = other_render_mask.sum()
                    other_render_mask = render_scale_map_other>(render_scale_map+scale_thresh)
                    render_scale_map[other_render_mask] = other_render_mask.sum()
                    main_renders[other_render_mask] = other_renders[other_render_mask]
                renders_summoned.append(main_renders)
            renders = np.array(renders_summoned)

        visible_weight = 0.9
        if images is not None:
            valid_mask = (renders[:,:, :, -1] > thresh)[:,:, :,np.newaxis]
            #renders[valid_mask, :-1] = images[valid_mask]
            if renders.shape[-1]==4:
                renders = renders[:,:,:,:-1]
            
            renders = renders * valid_mask * visible_weight + images * valid_mask * (1-visible_weight) + (1 - valid_mask) * images
        return renders.astype(np.uint8)

    def visulize_result_onorg(self, vertices, verts_camed, data, reorganize_idx=None, save_img=False,hp_aes=None, centermaps=None,**kwargs): #pkps, kps, 
        img_size=1024
        if reorganize_idx is not None:
            vids_org = np.unique(reorganize_idx)
            verts_vids, single_vids, new_idxs = [],[],[]
            count = 0
            for idx, vid in enumerate(vids_org):
                verts_vids.append(np.where(reorganize_idx==vid)[0])
                single_vids.append(np.where(reorganize_idx==vid)[0][0])
                new_idx = []
                for j in range((reorganize_idx==vid).sum()):
                    new_idx.append(count)
                    count+=1
                new_idxs.append(new_idx)
            verts_vids = np.concatenate(verts_vids)
            assert count==len(verts_vids)
        else:
            new_idxs = None
            vids_org, verts_vids, single_vids = [np.arange(data['image_org'].shape[0]) for _ in range(3)]

        images = data['image_org'].cpu().contiguous().numpy().astype(np.uint8)[single_vids]
        if images.shape[1] != self.vis_size[0]:
            images_new = []
            for image in images:
                images_new.append(cv2.resize(image, tuple(self.vis_size)))
            images = np.array(images_new)
        rendered_imgs = self.visualize_renderer(verts_camed[verts_vids], images=images, reorganize_idx=new_idxs)
        show_list = [images, rendered_imgs]

        if centermaps is not None:
            centermaps_list = []
            centermaps = torch.nn.functional.interpolate(centermaps[vids_org],size=(img_size,img_size),mode='bilinear')
            for idx,centermap in enumerate(centermaps):
                img_bk = cv2.resize(images[idx].copy(),(img_size,img_size))[:,:,::-1]
                centermaps_list.append(make_heatmaps(img_bk, centermap))

        out_list = []
        for idx in range(len(vids_org)):
            result_img = np.hstack([item[idx] for item in show_list])
            out_list.append(result_img[:,:,::-1])
            
        if save_img:
            img_names = np.array(data['imgpath'])[single_vids]
            os.makedirs(self.result_img_dir, exist_ok=True)
            for idx,result_img in enumerate(out_list):
                name = img_names[idx].split('/')[-2]+'-'+img_names[idx].split('/')[-1]
                name_save = os.path.join(self.result_img_dir,name)
                cv2.imwrite(name_save,result_img)
                if centermaps is not None:
                    cv2.imwrite(name_save.replace('.jpg','_centermap.jpg'),centermaps_list[idx])
        return np.array(out_list)

    def draw_skeleton(self, image, pts, **kwargs):
        return draw_skeleton(image, pts, **kwargs)

    def draw_skeleton_multiperson(self, image, pts, **kwargs):
        return draw_skeleton_multiperson(image, pts, **kwargs)

    def make_mp4(self,images,name):
        num = images.shape[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_movie = cv2.VideoWriter(name+'.mp4', fourcc, 50, (images.shape[2], images.shape[1]))
        for i in range(num):
            if i%100==0:
                print('Writing frame: ',i,'/',num)
            output_movie.write(images[i])

class Image_base(Dataset):
    def __init__(self, train_flag=True, regress_smpl = False):
        super(Image_base,self).__init__()
        self.heatmap_mapper = constants.joint_mapping(constants.SMPL_ALL_54, constants.COCO_17)

        self.input_shape = [args.input_size, args.input_size]
        self.high_resolution=args.high_resolution
        self.vis_size = 512 if self.high_resolution else 256
        self.labels, self.images, self.file_paths = [],[],[]

        self.root_inds = None
        self.torso_ids = [constants.SMPL_ALL_54[part] for part in ['Neck', 'Neck_LSP', 'R_Shoulder', 'L_Shoulder','Pelvis', 'R_Hip', 'L_Hip']]
        self.heatmap_res = 128
        self.joint_number = len(list(constants.SMPL_ALL_54.keys()))
        self.max_person = args.max_person

    def process_kps(self,kps,img_size,set_minus=True):
        kps = kps.astype(np.float32)
        kps[:,0] = kps[:,0]/ float(img_size[1])
        kps[:,1] = kps[:,1]/ float(img_size[0])
        kps[:,:2] = 2.0 * kps[:,:2] - 1.0

        if kps.shape[1]>2 and set_minus:
            kps[kps[:,2]<=0.03] = -2.
        kps_processed=kps[:,:2]
        for inds, kp in enumerate(kps_processed):
            x,y = kp
            if x > 1 or x < -1 or y < -1 or y > 1:
                kps_processed[inds] = -2.

        return kps_processed

    def map_kps(self,joint_org,maps=None):
        kps = joint_org[maps].copy()
        kps[maps==-1] = -2.
        return kps

    def _calc_center_(self, kps):
        vis = kps[self.torso_ids,0]>-1
        if vis.sum()>0:
            center = kps[self.torso_ids][vis].mean(0)
        elif (kps[:,0]>-1).sum()>0:
            center = kps[kps[:,0]>-1].mean(0)
        return center

    def parse_multiperson_kps(self,image, full_kps, subject_id):
        full_kps = [self.process_kps(kps_i,image.shape) for kps_i in full_kps]
        full_kp2d = np.ones((self.max_person, self.joint_number, 2))*-2.
        person_centers = np.ones((self.max_person, 2))*-2.
        subject_ids = np.ones(self.max_person)*-2
        used_person_inds = []
        filled_id = 0

        for inds in range(min(len(full_kps),self.max_person)):
            kps_i = full_kps[inds]
            center = self._calc_center_(kps_i)
            if center is None or len(center)==0:
                continue
            person_centers[filled_id] = center
            full_kp2d[filled_id] = kps_i
            subject_ids[filled_id] = subject_id[inds]
            filled_id+=1
            used_person_inds.append(inds)

        return person_centers, full_kp2d, subject_ids, used_person_inds
        
    def process_3d(self, info_3d, used_person_inds):
        dataset_name, kp3ds, params, kp3d_hands = info_3d

        kp3d_flag = np.zeros(self.max_person, dtype=np.bool)
        if kp3ds is not None:
            kp_num = kp3ds[0].shape[0]
            # -2 serves as an invisible flag
            kp3d_processed = np.ones((self.max_person,kp_num,3), dtype=np.float32)*-2.
            for inds, used_id in enumerate(used_person_inds):
                kp3d = kp3ds[used_id]
                valid_mask = self._check_kp3d_visible_parts_(kp3d)
                kp3d[~valid_mask] = -2.
                kp3d_flag[inds] = True
                kp3d_processed[inds] = kp3d
        else:
            kp3d_processed = np.ones((self.max_person,self.joint_number,3), dtype=np.float32)*-2.
            
        params_processed = np.ones((self.max_person,76), dtype=np.float32)*-10
        smpl_flag = np.zeros(self.max_person, dtype=np.bool)
        if params is not None:
            for inds, used_id in enumerate(used_person_inds):
                param = params[used_id]
                theta, beta = param[:66], param[-10:]
                params_processed[inds] = np.concatenate([theta, beta])
                smpl_flag[inds] = True
        
        return kp3d_processed, params_processed, kp3d_flag, smpl_flag

    def get_item_single_frame(self,index):

        info_2d, info_3d = self.get_image_info(index)
        dataset_name, imgpath, image, full_kps, root_trans, subject_id = info_2d
        full_kp2d_vis = [kps_i[:,-1] for kps_i in full_kps]
        img_size = (image.shape[1], image.shape[0])

        img_info = process_image(image, full_kps)
        image, image_wbg, full_kps, kps_offset = img_info
        person_centers, full_kp2d, subject_ids, used_person_inds = \
            self.parse_multiperson_kps(image, full_kps, subject_id)

        offset,lt,rb,size,_ = kps_offset
        offsets = np.array([image.shape[1],image.shape[0],lt[1],rb[1],lt[0],rb[0],offset[1],size[1],offset[0],size[0]],dtype=np.int)
        dst_image = cv2.resize(image, tuple(self.input_shape), interpolation = cv2.INTER_CUBIC)
        org_image = cv2.resize(image_wbg, (self.vis_size, self.vis_size), interpolation=cv2.INTER_CUBIC)
        kp3d, params, kp3d_flag, smpl_flag = self.process_3d(info_3d, used_person_inds)

        input_data = {
            'image': torch.from_numpy(dst_image).float(),
            'image_org': torch.from_numpy(org_image),
            'full_kp2d': torch.from_numpy(full_kp2d).float(),
            # rectify the x, y order, from x-y to y-x
            'person_centers':torch.from_numpy(person_centers[:,::-1].copy()).float(),
            'subject_ids':torch.from_numpy(subject_ids).long(),
            'kp_3d': torch.from_numpy(kp3d).float(),
            'params': torch.from_numpy(params).float(),
            'smpl_flag':torch.from_numpy(smpl_flag).bool(),
            'kp3d_flag': torch.from_numpy(kp3d_flag).bool(),
            'offsets': torch.from_numpy(offsets).float(),
            'imgpath': imgpath,
            'data_set':dataset_name,}

        return input_data

    def get_image_info(self,index):
        raise NotImplementedError

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        return self.get_item_single_frame(index)

    def _check_kp3d_visible_parts_(self, kp3ds, invisible_flag=-2.):
        visible_parts_mask = (kp3ds!=invisible_flag).sum(-1) == kp3ds.shape[-1]
        return visible_parts_mask

class PW3D(Image_base):
    def __init__(self,train_flag = False, split='test', mode='vibe', regress_smpl=True, **kwargs):
        super(PW3D,self).__init__(train_flag)
        self.data_folder = args.dataset_rootdir
        self.data3d_dir = os.path.join(self.data_folder,'sequenceFiles')
        self.image_dir = os.path.join(self.data_folder,'imageFiles')
        self.mode = mode
        self.split = split

        logging.info('Loading 3DPW in {} mode, split {}'.format(mode,self.split))
        if mode == 'vibe':
            self.annots_path = args.annot_dir #os.path.join(config.project_dir,'data/vibe_db')
            self.joint_mapper = constants.joint_mapping(constants.LSP_14,constants.SMPL_ALL_54)
            self.joint3d_mapper = constants.joint_mapping(constants.LSP_14,constants.SMPL_ALL_54)
            self.load_vibe_annots()
        elif mode == 'whole':
            self.joint_mapper = constants.joint_mapping(constants.COCO_18,constants.SMPL_ALL_54)
            self.joint3d_mapper = constants.joint_mapping(constants.SMPL_24,constants.SMPL_ALL_54)
            self.annots_path = os.path.join(self.data_folder,'annots.npz')
            if not os.path.exists(self.annots_path):
                self.pack_data()
            self.load_annots()
        else:
            raise NotImplementedError

        self.root_inds = [constants.SMPL_ALL_54['R_Hip'], constants.SMPL_ALL_54['L_Hip']]
        
        logging.info('3DPW dataset {} split total {} samples, loading mode {}'.format(self.split ,self.__len__(), self.mode))

    def __len__(self):
        return len(self.file_paths)

class Internet(Dataset):
    def __init__(self, image_folder=None, **kwargs):
        super(Internet,self).__init__()
        self.file_paths = glob.glob(os.path.join(image_folder,'*'))
        sorted(self.file_paths)
        
        print('Loading {} internet data from:{}'.format(len(self), image_folder))

    def get_image_info(self,index):
        return self.file_paths[index]
        
    def resample(self):
        return self.__getitem__(random.randint(0,len(self)))

    def get_item_single_frame(self,index):

        imgpath = self.get_image_info(index)
        image = cv2.imread(imgpath)
        if image is None:
            index = self.resample()
            imgpath = self.get_image_info(index)
            image = cv2.imread(imgpath)

        input_data = img_preprocess(image, imgpath, input_size=args.input_size)

        return input_data


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        try:
            return self.get_item_single_frame(index)
        except Exception as error:
            print(error)
            index = np.random.randint(len(self))
            return self.get_item_single_frame(index)

class SingleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset=None,**kwargs):
        assert dataset in dataset_dict, print('dataset {} not found while creating data loader!'.format(dataset))
        self.dataset = dataset_dict[dataset](**kwargs)
        self.length = len(self.dataset)            

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.length

class Base(nn.Module):
    def forward(self, meta_data, **cfg):
        if cfg['mode'] == 'train':
            return self.train_forward(meta_data, **cfg)
        elif cfg['mode'] == 'val':
            return self.val_forward(meta_data, **cfg)
        elif cfg['mode'] == 'forward':
            return self.pure_forward(meta_data, **cfg)
        else:
            raise NotImplementedError('forward mode is not recognized! please set proper mode (train/val)')

    def train_forward(self, meta_data, **cfg):
        outputs = self.feed_forward(meta_data)
        outputs, meta_data = self._result_parser.train_forward(outputs, meta_data, cfg)
        outputs['meta_data'] = meta_data
        return outputs

    @torch.no_grad()
    def val_forward(self, meta_data, **cfg):
        outputs = self.feed_forward(meta_data)
        outputs, meta_data = self._result_parser.val_forward(outputs, meta_data, cfg)

        outputs['meta_data'] = meta_data
        return outputs

    def feed_forward(self, meta_data):
        x = self.backbone(meta_data['image'].contiguous())
        outputs = self.head_forward(x)
        return outputs

    @torch.no_grad()
    def pure_forward(self, meta_data, **cfg):
        outputs = self.feed_forward(meta_data)
        return outputs

    def head_forward(self,x):
        return NotImplementedError

    def make_backbone(self):
        return NotImplementedError

    def backbone_forward(self, x):
        return NotImplementedError

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

class Base1(object):
    def __init__(self):
        self.project_dir = config.project_dir
        hparams_dict = self.load_config_dict(vars(args))
        self._init_params()

    def _build_model_(self):
        logging.info('start building model.')
        model = build_model()
        if '-1' not in self.gpu:
            model = model.cuda()
        model = load_model(self.gmodel_path, model, \
            prefix = 'module.', drop_prefix='')
        self.model = nn.DataParallel(model)
        logging.info('finished build model.')

    def _init_params(self):
        self.global_count = 0
        self.demo_cfg = {'mode':'val', 'calc_loss': False}
        self.eval_cfg = {'mode':'train', 'calc_loss': False}
        self.gpus = [int(i) for i in self.gpu.split(',')]

    def _create_single_data_loader(self, **kwargs):
        logging.info('gathering datasets')
        datasets = SingleDataset(**kwargs)
        return DataLoader(dataset = datasets, shuffle = False,batch_size = self.val_batch_size,\
                drop_last = False if self.eval else True, pin_memory = True, num_workers = self.nw)

    def load_config_dict(self, config_dict):
        hparams_dict = {}
        for i, j in config_dict.items():
            setattr(self,i,j)
            hparams_dict[i] = j

        logging.basicConfig(level=logging.INFO if self.local_rank in [-1, 0] else logging.WARN)
        logging.info(config_dict)
        logging.info('-'*66)
        return hparams_dict

    def net_forward(self, meta_data, cfg=None):
        ds_org, imgpath_org = get_remove_keys(meta_data,keys=['data_set','imgpath'])
        meta_data['batch_ids'] = torch.arange(len(meta_data['image']))
        outputs = self.model(meta_data, **cfg)

        outputs['meta_data']['data_set'], outputs['meta_data']['imgpath'] = reorganize_items([ds_org, imgpath_org], outputs['reorganize_idx'].cpu().numpy())
        return outputs

class Demo(Base1):
    def __init__(self):
        super(Demo, self).__init__()
        self._build_model_()
        self._prepare_modules_()

    def _prepare_modules_(self):
        self.model.eval()
        self.demo_dir = os.path.join(config.project_dir, 'demo')
        self.vis_size = [1024,1024,3] #[1920,1080]
        if not args.webcam and '-1' not in self.gpu:
            self.visualizer = Visualizer(resolution=self.vis_size, input_size=self.input_size,with_renderer=True)
        else:
            self.save_visualization_on_img = False
        if self.save_mesh:
            self.smpl_faces = pkl.load(open(os.path.join(args.smpl_model_path,'smpl','SMPL_NEUTRAL.pkl'),'rb'), encoding='latin1')['f']
        print('Initialization finished!')

    def run(self, image_folder):
        print('Processing {}, saving to {}'.format(image_folder, self.output_dir))
        os.makedirs(self.output_dir, exist_ok=True)
        if '-1' not in self.gpu:
            self.visualizer.result_img_dir = self.output_dir 
            
        internet_loader = self._create_single_data_loader(dataset='internet',train_flag=False, image_folder=image_folder)
        with torch.no_grad():
            for test_iter,meta_data in enumerate(internet_loader):
                outputs = self.net_forward(meta_data, cfg=self.demo_cfg)
                reorganize_idx = outputs['reorganize_idx'].cpu().numpy()
                
                if self.save_dict_results:
                    self.reorganize_results(outputs, outputs['meta_data']['imgpath'], reorganize_idx, self.output_dir)
                if self.save_visualization_on_img:
                    vis_eval_results = self.visualizer.visulize_result_onorg(outputs['verts'], outputs['verts_camed'], outputs['meta_data'], \
                    reorganize_idx, centermaps= outputs['center_map'] if self.save_centermap else None,save_img=True)#

                if self.save_mesh:
                    save_meshes(reorganize_idx, outputs, self.output_dir, self.smpl_faces) 

    def reorganize_results(self, outputs, img_paths, reorganize_idx, test_save_dir=None):
        results = {}
        cam_results = outputs['params']['cam'].detach().cpu().numpy().astype(np.float16)
        smpl_pose_results = torch.cat([outputs['params']['global_orient'], outputs['params']['body_pose']],1).detach().cpu().numpy().astype(np.float16)
        smpl_shape_results = outputs['params']['betas'].detach().cpu().numpy().astype(np.float16)
        joints_54 = outputs['j3d'].detach().cpu().numpy().astype(np.float16)
        kp3d_smpl24_results = outputs['joints_smpl24'].detach().cpu().numpy().astype(np.float16)
        kp3d_spin24_results = joints_54[:,constants.joint_mapping(constants.SMPL_ALL_54, constants.SPIN_24)]
        kp3d_op25_results = joints_54[:,constants.joint_mapping(constants.SMPL_ALL_54, constants.OpenPose_25)]
        verts_results = outputs['verts'].detach().cpu().numpy().astype(np.float16)
        pj2d_results = outputs['pj2d'].detach().cpu().numpy().astype(np.float16)

        vids_org = np.unique(reorganize_idx)
        for idx, vid in enumerate(vids_org):
            verts_vids = np.where(reorganize_idx==vid)[0]
            img_path = img_paths[verts_vids[0]]
            results[img_path] = [{} for idx in range(len(verts_vids))]
            for subject_idx, batch_idx in enumerate(verts_vids):
                results[img_path][subject_idx]['cam'] = cam_results[batch_idx]
                results[img_path][subject_idx]['pose'] = smpl_pose_results[batch_idx]
                results[img_path][subject_idx]['betas'] = smpl_shape_results[batch_idx]
                results[img_path][subject_idx]['j3d_smpl24'] = kp3d_smpl24_results[batch_idx]
                results[img_path][subject_idx]['j3d_spin24'] = kp3d_spin24_results[batch_idx]
                results[img_path][subject_idx]['j3d_op25'] = kp3d_op25_results[batch_idx]
                results[img_path][subject_idx]['verts'] = verts_results[batch_idx]
                results[img_path][subject_idx]['pj2d'] = pj2d_results[batch_idx]
                results[img_path][subject_idx]['trans'] = convert_cam_to_3d_trans(cam_results[batch_idx])

        if test_save_dir is not None:
            for img_path, result_dict in results.items():
                name = (test_save_dir+'/{}'.format(os.path.basename(img_path))).replace('.jpg','.npz').replace('.png','.npz')
                # get the results: np.load('/path/to/person_overlap.npz',allow_pickle=True)['results'][()]
                np.savez(name, results=result_dict)
        return results

    def single_image_forward(self,image):
        meta_data = img_preprocess(image, '0', input_size=args.input_size, single_img_input=True)
        if '-1' not in self.gpu:
            meta_data['image'] = meta_data['image'].cuda()
        outputs = self.net_forward(meta_data, cfg=self.demo_cfg)
        return outputs

    def process_video(self, video_file_path=None):
        capture = OpenCVCapture(video_file_path)
        video_length = int(capture.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_basename = get_video_bn(video_file_path)
        print('Processing {}, saving to {}'.format(video_file_path, self.output_dir))
        os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.isdir(self.output_dir):
            self.output_dir = video_file_path.replace(os.path.basename(video_file_path),'')

        results, result_frames = {}, []
        for frame_id in range(video_length):
            if frame_id == 30:
                break
            print('Processing video {}/{}'.format(frame_id, video_length))
            frame = capture.read()
            with torch.no_grad():
                outputs = self.single_image_forward(frame)
            vis_dict = {'image_org': outputs['meta_data']['image_org'].cpu()}
            img_paths = [str(frame_id) for _ in range(1)]
            single_batch_results = self.reorganize_results(outputs,img_paths,outputs['reorganize_idx'].cpu().numpy())
            results.update(single_batch_results)
            vis_eval_results = self.visualizer.visulize_result_onorg(outputs['verts'], outputs['verts_camed'], vis_dict, reorganize_idx=outputs['reorganize_idx'].cpu().numpy())
            result_frames.append(vis_eval_results[0])
            outputs['meta_data']['imgpath'] = img_paths
            if self.save_mesh:
                save_meshes(outputs['reorganize_idx'].cpu().numpy(), outputs, self.output_dir, self.smpl_faces)
        
        if self.save_dict_results:
            save_dict_path = os.path.join(self.output_dir, video_basename+'_results.npz')
            print('Saving parameter results to {}'.format(save_dict_path))
            np.savez(save_dict_path, results=results)

        if self.save_video_results:
            video_save_name = os.path.join(self.output_dir, video_basename+'_results.mp4')
            print('Writing results to {}'.format(video_save_name))
            frames2video(result_frames, video_save_name, fps=args.fps_save)

class OpenCVCapture:
    def __init__(self, video_file=None):
        if video_file is None:
            self.cap = cv2.VideoCapture(int(args.cam_id))
        else:
            self.cap = cv2.VideoCapture(video_file)

    def read(self):
        flag, frame = self.cap.read()
        if not flag:
          return None
        return np.flip(frame, -1).copy() # BGR to RGB

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

class HigherResolutionNet(nn.Module):

    def __init__(self, **kwargs):
        self.inplanes = 64
        super(HigherResolutionNet, self).__init__()
        self.make_baseline()
        self.backbone_channels = 32

    def load_pretrain_params(self):
        success_layer = copy_state_dict(self.state_dict(), torch.load(config.hrnet_pretrain), prefix = '', fix_loaded=True)
        
    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1,BN=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,BN=BN))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,BN=BN))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def make_baseline(self):
        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4, BN=nn.BatchNorm2d)

        self.stage2_cfg = {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC',\
            'NUM_BLOCKS': [4,4], 'NUM_CHANNELS':[32,64], 'FUSE_METHOD': 'SUM'}
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = {'NUM_MODULES': 4, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC',\
            'NUM_BLOCKS': [4,4,4], 'NUM_CHANNELS':[32,64,128], 'FUSE_METHOD': 'SUM'}
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = {'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC',\
            'NUM_BLOCKS': [4,4,4,4], 'NUM_CHANNELS':[32,64,128,256], 'FUSE_METHOD': 'SUM'}
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

    def forward(self, x):
        x = ((BHWC_to_BCHW(x)/ 255) * 2.0 - 1.0).contiguous()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        x = y_list[0]
        return x
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

class ResNet_50(nn.Module):
    def __init__(self, **kwargs):
        self.inplanes = 64
        super(ResNet_50, self).__init__()
        self.make_resnet()
        self.backbone_channels = 64
        #self.init_weights()

    def load_pretrain_params(self):
        success_layer = copy_state_dict(self.state_dict(), torch.load(config.resnet_pretrain), prefix = 'module.', fix_loaded=True)

    def make_resnet(self):
        block, layers = Bottleneck, [3, 4, 6, 3]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_resnet_layer(block, 64, layers[0])
        self.layer2 = self._make_resnet_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_resnet_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_resnet_layer(block, 512, layers[3], stride=2)

        self.deconv_layers = self._make_deconv_layer(3,(256,128,64),(4,4,4))

    def forward(self,x):
        x = ((BHWC_to_BCHW(x)/ 255) * 2.0 - 1.0).contiguous()
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.deconv_layers(x)
        return x

    def _make_resnet_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),)#,affine=False),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            if i==0:
                self.inplanes=2048
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))#,affine=False))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

class VertexJointSelector(nn.Module):

    def __init__(self, vertex_ids=None,
                 use_hands=True,
                 use_feet_keypoints=True, **kwargs):
        super(VertexJointSelector, self).__init__()

        extra_joints_idxs = []

        face_keyp_idxs = np.array([
            vertex_ids['nose'],
            vertex_ids['reye'],
            vertex_ids['leye'],
            vertex_ids['rear'],
            vertex_ids['lear']], dtype=np.int64)

        extra_joints_idxs = np.concatenate([extra_joints_idxs,
                                            face_keyp_idxs])

        if use_feet_keypoints:
            feet_keyp_idxs = np.array([vertex_ids['LBigToe'],
                                       vertex_ids['LSmallToe'],
                                       vertex_ids['LHeel'],
                                       vertex_ids['RBigToe'],
                                       vertex_ids['RSmallToe'],
                                       vertex_ids['RHeel']], dtype=np.int32)

            extra_joints_idxs = np.concatenate(
                [extra_joints_idxs, feet_keyp_idxs])

        if use_hands:
            self.tip_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

            tips_idxs = []
            for hand_id in ['l', 'r']:
                for tip_name in self.tip_names:
                    tips_idxs.append(vertex_ids[hand_id + tip_name])

            extra_joints_idxs = np.concatenate(
                [extra_joints_idxs, tips_idxs])

        self.register_buffer('extra_joints_idxs',
                             to_tensor(extra_joints_idxs, dtype=torch.long))

    def forward(self, vertices, joints):
        extra_joints = torch.index_select(vertices, 1, self.extra_joints_idxs)
        joints = torch.cat([joints, extra_joints], dim=1)

        return joints

class SMPL(nn.Module):

    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    NUM_BETAS = 10

    def __init__(self, model_path, J_reg_extra9_path=None, J_reg_h36m17_path=None,\
                 data_struct=None, betas=None, global_orient=None,\
                 body_pose=None, transl=None, dtype=torch.float32, batch_size=1,\
                 joint_mapper=None, gender='neutral', vertex_ids=None, **kwargs):
        ''' SMPL model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            data_struct: Strct
                A struct object. If given, then the parameters of the model are
                read from the object. Otherwise, the model tries to read the
                parameters from the given `model_path`. (default = None)
            global_orient: torch.tensor, optional, Bx3
                The default value for the global orientation variable.
                (default = None)
            body_pose: torch.tensor, optional, Bx(Body Joints * 3)
                The default value for the body pose variable.
                (default = None)
            betas: torch.tensor, optional, Bx10
                The default value for the shape member variable.
                (default = None)
            transl: torch.tensor, optional, Bx3
                The default value for the transl variable.
                (default = None)
            dtype: torch.dtype, optional
                The data type for the created variables
            batch_size: int, optional
                The batch size used for creating the member variables
            joint_mapper: object, optional
                An object that re-maps the joints. Useful if one wants to
                re-order the SMPL joints to some other convention (e.g. MSCOCO)
                (default = None)
            gender: str, optional
                Which gender to load
            vertex_ids: dict, optional
                A dictionary containing the indices of the extra vertices that
                will be selected
        '''

        self.gender = gender

        if data_struct is None:
            if osp.isdir(model_path):
                model_fn = 'SMPL_{}.{ext}'.format(gender.upper(), ext='pkl')
                smpl_path = os.path.join(model_path, model_fn)
            else:
                smpl_path = model_path
            assert osp.exists(smpl_path), 'Path {} does not exist!'.format(
                smpl_path)

            with open(smpl_path, 'rb') as smpl_file:
                data_struct = Struct(**pkl.load(smpl_file,
                                                   encoding='latin1'))

        super(SMPL, self).__init__()
        self.batch_size = batch_size

        if vertex_ids is None:
            # SMPL and SMPL-H share the same topology, so any extra joints can
            # be drawn from the same place
            vertex_ids = VERTEX_IDS['smplh']

        self.dtype = dtype

        #self.joint_mapper = joint_mapper

        self.vertex_joint_selector = VertexJointSelector(
            vertex_ids=vertex_ids, **kwargs)

        self.faces = data_struct.f
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))

        # The vertices of the template model
        self.register_buffer('v_template',
                             to_tensor(to_np(data_struct.v_template),
                                       dtype=dtype))
        if betas is None:
            default_betas = torch.zeros([batch_size, self.NUM_BETAS],dtype=dtype)
        else:
            if 'torch.Tensor' in str(type(betas)):
                default_betas = betas.clone().detach()
            else:
                default_betas = torch.tensor(betas,dtype=dtype)

        self.register_parameter('betas', nn.Parameter(default_betas,
                                                      requires_grad=True))

        # The shape components
        shapedirs = data_struct.shapedirs
        # The shape components
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(shapedirs), dtype=dtype))

        j_regressor = to_tensor(to_np(
            data_struct.J_regressor), dtype=dtype)
        self.register_buffer('J_regressor', j_regressor)

        if J_reg_extra9_path is not None:
            J_regressor_extra9 = np.load(J_reg_extra9_path)
            J_regressor_extra9 = to_tensor(to_np(J_regressor_extra9), dtype=dtype)
            self.register_buffer('J_regressor_extra9', J_regressor_extra9)
        else:
            self.register_buffer('J_regressor_extra9', None)

        if J_reg_h36m17_path is not None:
            H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
            J_regressor_h36m17 = np.load(J_reg_h36m17_path)[H36M_TO_J17]
            J_regressor_h36m17 = to_tensor(to_np(J_regressor_h36m17), dtype=dtype)
            self.register_buffer('J_regressor_h36m17', J_regressor_h36m17)
        else:
            self.register_buffer('J_regressor_h36m17', None)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data_struct.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer('lbs_weights',
                             to_tensor(to_np(data_struct.weights), dtype=dtype))

    def create_mean_pose(self, data_struct):
        pass

    @torch.no_grad()
    def reset_params(self, **params_dict):
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                param[:] = torch.tensor(params_dict[param_name])
            else:
                param.fill_(0)

    def get_num_verts(self):
        return self.v_template.shape[0]

    def get_num_faces(self):
        return self.faces.shape[0]

    def extra_repr(self):
        return 'Number of betas: {}'.format(self.NUM_BETAS)

    def forward(self, betas=None, body_pose=None, global_orient=None,
                transl=None, return_verts=True, return_full_pose=False,
                **kwargs):
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
        '''
        betas = betas if betas is not None else self.betas

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, dtype=self.dtype)

        joints = self.vertex_joint_selector(vertices, joints)
        joints_smpl24 = joints.clone()
        if self.J_regressor_h36m17 is not None:
            # 54 joints = 45 joints + 9 extra joints from different datasets
            joints_h36m17 = vertices2joints(self.J_regressor_h36m17, vertices)
            # use the middle of hip used in the most 2D pose datasets, not the o-th Pelvis of SMPL 24 joint
            joints_h36m17_pelvis = joints_h36m17[:,14].unsqueeze(1)
            joints_h36m17 = joints_h36m17 - joints_h36m17_pelvis

        if self.J_regressor_extra9 is not None:
            # 54 joints = 45 joints + 9 extra joints from different datasets
            joints = torch.cat([joints, vertices2joints(self.J_regressor_extra9, vertices)],1)
            # use the Pelvis of most 2D image, not the original Pelvis
            root_trans = joints[:,49].unsqueeze(1)
            if args.model_version!=1 or args.backbone!='hrnet':
                joints = joints - root_trans
                vertices =  vertices - root_trans

        output = ModelOutput(vertices=vertices,
                             global_orient=global_orient,
                             body_pose=body_pose,
                             joints=joints,
                             joints_h36m17=joints_h36m17,
                             joints_smpl24=joints_smpl24,
                             betas=betas,
                             full_pose=full_pose)

        return output

class SMPLWrapper(nn.Module):
    def __init__(self):
        super(SMPLWrapper,self).__init__()
        self.smpl_model = smpl_model_create(args.smpl_model_path, J_reg_extra9_path=args.smpl_J_reg_extra_path, J_reg_h36m17_path=args.smpl_J_reg_h37m_path, \
            batch_size=args.batch_size,model_type='smpl', gender='neutral', use_face_contour=False, ext='npz',flat_hand_mean=True, use_pca=False)
        if '-1' not in args.gpu:
            self.smpl_model = self.smpl_model.cuda()
        self.part_name = ['cam', 'global_orient', 'body_pose', 'betas']
        self.part_idx = [args.cam_dim, args.rot_dim, (args.smpl_joint_num-1)*args.rot_dim, 10]
        self.params_num = np.array(self.part_idx).sum()

    def forward(self, outputs, meta_data):
        idx_list, params_dict = [0], {}
        for i,  (idx, name) in enumerate(zip(self.part_idx,self.part_name)):
            idx_list.append(idx_list[i] + idx)
            params_dict[name] = outputs['params_pred'][:, idx_list[i]: idx_list[i+1]].contiguous()

        if args.Rot_type=='6D':
            params_dict['body_pose'] = rot6D_to_angular(params_dict['body_pose'])
            params_dict['global_orient'] = rot6D_to_angular(params_dict['global_orient'])
        N = params_dict['body_pose'].shape[0]
        params_dict['body_pose'] = torch.cat([params_dict['body_pose'], torch.zeros(N,6).to(params_dict['body_pose'].device)],1)        
        
        smpl_outs = self.smpl_model(**params_dict, return_verts=True, return_full_pose=True)

        outputs.update({'params': params_dict, 'verts': smpl_outs.vertices, 'j3d':smpl_outs.joints, \
            'joints_h36m17':smpl_outs.joints_h36m17, 'joints_smpl24':smpl_outs.joints_smpl24, 'poses':smpl_outs.full_pose})

        outputs.update(vertices_kp3d_projection(outputs))        
        
        return outputs

class CenterMap(object):
    def __init__(self,style='heatmap_adaptive_scale'):
        self.style=style
        self.size = args.centermap_size
        self.max_person = args.max_person
        self.shrink_scale = float(args.input_size//self.size)
        self.dims = 1
        self.sigma = 1
        self.conf_thresh= args.centermap_conf_thresh
        self.gk_group, self.pool_group = self.generate_kernels(args.kernel_sizes)

    def generate_kernels(self, kernel_size_list):
        gk_group, pool_group = {}, {}
        for kernel_size in set(kernel_size_list):
            x = np.arange(0, kernel_size, 1, float)
            y = x[:, np.newaxis]
            x0, y0 = (kernel_size-1)//2,(kernel_size-1)//2
            gaussian_distribution = - ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2)
            gk_group[kernel_size] = np.exp(gaussian_distribution)
            pool_group[kernel_size] = torch.nn.MaxPool2d(kernel_size, 1, (kernel_size-1)//2)
        return gk_group, pool_group

    def process_gt_CAM(self, center_normed):
        center_list = []
        valid_mask = center_normed[:,:,0]>-1
        valid_inds = torch.where(valid_mask)
        valid_batch_inds, valid_person_ids = valid_inds[0], valid_inds[1]
        center_gt = ((center_normed+1)/2*self.size).long()
        center_gt_valid = center_gt[valid_mask]
        return (valid_batch_inds, valid_person_ids, center_gt_valid)

    def parse_centermap(self, center_map):
        return self.parse_centermap_heatmap_adaptive_scale_batch(center_map)

    def parse_centermap_heatmap_adaptive_scale_batch(self, center_maps):
        center_map_nms = nms(center_maps, pool_func=self.pool_group[args.kernel_sizes[-1]])
        b, c, h, w = center_map_nms.shape
        K = self.max_person

        topk_scores, topk_inds = torch.topk(center_map_nms.reshape(b, c, -1), K)
        topk_inds = topk_inds % (h * w)
        topk_ys = (topk_inds // w).int().float()
        topk_xs = (topk_inds % w).int().float()
        # get all topk in in a batch
        topk_score, index = torch.topk(topk_scores.reshape(b, -1), K)
        # div by K because index is grouped by K(C x K shape)
        topk_clses = (index // K).int()
        topk_inds = gather_feature(topk_inds.view(b, -1, 1), index).reshape(b, K)
        topk_ys = gather_feature(topk_ys.reshape(b, -1, 1), index).reshape(b, K)
        topk_xs = gather_feature(topk_xs.reshape(b, -1, 1), index).reshape(b, K)

        mask = topk_score>self.conf_thresh
        batch_ids = torch.where(mask)[0]
        center_yxs = torch.stack([topk_ys[mask], topk_xs[mask]]).permute((1,0))
        return batch_ids, topk_inds[mask], center_yxs, topk_score[mask]

class ResultParser(nn.Module):
    def __init__(self):
        super(ResultParser,self).__init__()
        self.map_size = args.centermap_size
        self.params_map_parser = SMPLWrapper()
        self.centermap_parser = CenterMap()

    def train_forward(self, outputs, meta_data, cfg):
        outputs,meta_data = self.match_params(outputs, meta_data)
        outputs = self.params_map_parser(outputs,meta_data)
        return outputs,meta_data

    @torch.no_grad()
    def val_forward(self, outputs, meta_data, cfg):
        outputs, meta_data = self.parse_maps(outputs, meta_data, cfg)
        outputs = self.params_map_parser(outputs,meta_data)
        return outputs, meta_data

    def match_params(self, outputs, meta_data):
        gt_keys = ['params', 'full_kp2d', 'kp_3d', 'smpl_flag', 'kp3d_flag', 'subject_ids']
        exclude_keys = ['person_centers','offsets']

        center_gts_info = process_gt_center(meta_data['person_centers'])
        center_preds_info = self.centermap_parser.parse_centermap(outputs['center_map'])
        mc_centers = self.match_gt_pred(center_gts_info, center_preds_info, outputs['center_map'].device)
        batch_ids, flat_inds, person_ids = mc_centers['batch_ids'], mc_centers['flat_inds'], mc_centers['person_ids']
        if len(batch_ids)==0:
            logging.error('number of predicted center is {}'.format(batch_ids))
            batch_ids, flat_inds = torch.zeros(2).long().to(outputs['center_map'].device), (torch.ones(2)*self.map_size**2/2.).to(outputs['center_map'].device).long()
            person_ids = batch_ids.clone()
        
        params_pred = self.parameter_sampling(outputs['params_maps'], batch_ids, flat_inds, use_transform=True)
        outputs,meta_data = self.reorganize_data(outputs, meta_data, exclude_keys, gt_keys, batch_ids, person_ids)
        outputs['params_pred'] = params_pred
        return outputs,meta_data

    def match_gt_pred(self,center_gts_info, center_preds_info, device):
        vgt_batch_ids, vgt_person_ids, center_gts = center_gts_info
        vpred_batch_ids, flat_inds, cyxs, top_score = center_preds_info
        mc = {key:[] for key in ['batch_ids', 'flat_inds', 'person_ids']}

        for batch_id, person_id, center_gt in zip(vgt_batch_ids, vgt_person_ids, center_gts):
            if batch_id in vpred_batch_ids:
                center_pred = cyxs[vpred_batch_ids==batch_id]
                center_gt = center_pred[torch.argmin(torch.norm(center_pred.float()-center_gt[None].float().to(device),dim=-1))].long()
                cy, cx = torch.clamp(center_gt, 0, self.map_size-1)
                flat_ind = cy*args.centermap_size+cx
                mc['batch_ids'].append(batch_id); mc['flat_inds'].append(flat_ind); mc['person_ids'].append(person_id)
        keys_list = list(mc.keys())
        for key in keys_list:
            mc[key] = torch.Tensor(mc[key]).long().to(device)

        return mc
        
    def parameter_sampling(self, maps, batch_ids, flat_inds, use_transform=True):
        device = maps.device
        if use_transform:
            batch, channel = maps.shape[:2]
            maps = maps.view(batch, channel, -1).permute((0, 2, 1)).contiguous()
        results = maps[batch_ids,flat_inds].contiguous()
        return results

    def reorganize_gts(self, meta_data, key_list, batch_ids):
        for key in key_list:
            if isinstance(meta_data[key], torch.Tensor):
                meta_data[key] = meta_data[key][batch_ids]
            elif isinstance(meta_data[key], list):
                meta_data[key] = np.array(meta_data[key])[batch_ids.cpu().numpy()]
        return meta_data

    def reorganize_data(self, outputs, meta_data, exclude_keys, gt_keys, batch_ids, person_ids):
        exclude_keys += gt_keys
        outputs['reorganize_idx'] = meta_data['batch_ids'][batch_ids]
        info_vis = []
        for key, item in meta_data.items():
            if key not in exclude_keys:
                info_vis.append(key)

        meta_data = self.reorganize_gts(meta_data, info_vis, batch_ids)
        for gt_key in gt_keys:
            meta_data[gt_key] = meta_data[gt_key][batch_ids,person_ids]
        #meta_data['kp_2d'] = meta_data['full_kp2d']
        return outputs,meta_data

    @torch.no_grad()
    def parse_maps(self,outputs, meta_data, cfg):
        center_preds_info = self.centermap_parser.parse_centermap(outputs['center_map'])
        batch_ids, flat_inds, cyxs, top_score = center_preds_info
        if len(batch_ids)==0:
            #logging.error('number of predicted center is {}'.format(batch_ids))
            batch_ids, flat_inds = torch.zeros(2).long().to(outputs['center_map'].device), (torch.ones(2)*self.map_size**2/2.).to(outputs['center_map'].device).long()
            person_ids = batch_ids.clone()
            outputs['detection_flag'] = False
        else:
            outputs['detection_flag'] = True
        outputs['params_pred'] = self.parameter_sampling(outputs['params_maps'], batch_ids, flat_inds, use_transform=True)
        outputs['reorganize_idx'] = meta_data['batch_ids'][batch_ids]
        info_vis = ['image_org', 'offsets']
        meta_data = self.reorganize_gts(meta_data, info_vis, batch_ids)
        
        return outputs,meta_data

class ROMP(Base):
    def __init__(self, backbone=None,**kwargs):
        super(ROMP, self).__init__()
        print('Using ROMP v1')
        self.backbone = backbone
        self._result_parser = ResultParser()
        self._build_head()
        self.init_weights()
        #self.backbone.load_pretrain_params()

    def head_forward(self,x):
        x = torch.cat((x, self.coordmaps.to(x.device).repeat(x.shape[0],1,1,1)), 1)

        params_maps = self.final_layers[1](x)
        center_maps = self.final_layers[2](x)
        cam_maps = self.final_layers[3](x)
        # to make sure that scale is always a positive value
        cam_maps[:, 0] = torch.pow(1.1,cam_maps[:, 0])
        params_maps = torch.cat([cam_maps, params_maps], 1)
        output = {'params_maps':params_maps.float(), 'center_map':center_maps.float()} #, 'kp_ae_maps':kp_heatmap_ae.float()
        return output

    def _build_head(self):
        self.NUM_JOINTS = 17
        self.outmap_size = args.centermap_size
        params_num = self._result_parser.params_map_parser.params_num
        cam_dim = 3
        self.head_cfg = {'NUM_HEADS': 1, 'NUM_CHANNELS': 64, 'NUM_BASIC_BLOCKS': 2}
        self.output_cfg = {'NUM_PARAMS_MAP':params_num-cam_dim, 'NUM_CENTER_MAP':1, 'NUM_CAM_MAP':cam_dim}

        self.final_layers = self._make_final_layers(self.backbone.backbone_channels)
        self.coordmaps = get_coord_maps(128)

    def _make_final_layers(self, input_channels):
        final_layers = []
        final_layers.append(None)

        input_channels += 2
        final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_PARAMS_MAP']))
        final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP']))
        final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CAM_MAP']))

        return nn.ModuleList(final_layers)

    def trans_to_head_layers(self, input_channels,output_channels):
        trans_layers = []
        kernel_sizes, strides, paddings = self._get_trans_cfg()
        
        for kernel_size, padding, stride in zip(kernel_sizes, paddings, strides):
            trans_layers.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=output_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding),
                    nn.BatchNorm2d(output_channels, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)))
            input_channels = output_channels
        return nn.Sequential(*trans_layers)
    
    def _make_head_layers(self, input_channels, output_channels):
        head_layers = []
        num_channels = self.head_cfg['NUM_CHANNELS']

        kernel_sizes, strides, paddings = self._get_trans_cfg()
        for kernel_size, padding, stride in zip(kernel_sizes, paddings, strides):
            head_layers.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding),
                    nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)))
        
        for i in range(self.head_cfg['NUM_HEADS']):
            layers = []
            for _ in range(self.head_cfg['NUM_BASIC_BLOCKS']):
                layers.append(nn.Sequential(BasicBlock(num_channels, num_channels)))
            head_layers.append(nn.Sequential(*layers))

        head_layers.append(nn.Conv2d(in_channels=num_channels,out_channels=output_channels,\
            kernel_size=1,stride=1,padding=0))

        return nn.Sequential(*head_layers)

    def _get_trans_cfg(self):
        if self.outmap_size == 32:
            kernel_sizes = [3,3]
            paddings = [1,1]
            strides = [2,2]
        elif self.outmap_size == 64:
            kernel_sizes = [3]
            paddings = [1]
            strides = [2]
        elif self.outmap_size == 128:
            kernel_sizes = [3]
            paddings = [1]
            strides = [1]

        return kernel_sizes, strides, paddings

Backbones = {'hrnet': HigherResolutionNet, 'resnet': ResNet_50}
Heads = {1: ROMP}

blocks_dict = {
    'BASIC': BasicBlock,
    'BASIC_IBN_a': BasicBlock_IBN_a,
    'BOTTLENECK': Bottleneck
}

dataset_dict = {'pw3d':PW3D, 'internet':Internet}

def main():
    demo = Demo()
    demo.process_video(args.input_video_path)

if __name__ == '__main__':
    main()
