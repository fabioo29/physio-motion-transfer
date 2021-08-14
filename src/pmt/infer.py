import os 
import pickle as pkl
import cv2
import pyopenpose as op
import numpy as np
import time

from argparse import ArgumentParser
from tqdm import tqdm

INPUT_RESIZE = 1080
EXPORT_DECODED = False

def run_cihp_pgn(input_dir):

    #Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load input
    input_files = sorted(glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg'))) # 
    input_queue = tf.train.slice_input_producer([tf.convert_to_tensor(input_files, dtype=tf.string)], shuffle=False)
    img_contents = tf.io.read_file(input_queue[0])
    img = tf.io.decode_jpeg(img_contents, channels=3)
    # Resize to prevent OOM
    img = tf.image.resize(img, [INPUT_RESIZE, INPUT_RESIZE], preserve_aspect_ratio=True)
    img_r, img_g, img_b = tf.split(value=img, num_or_size_splits=3, axis=2)
    image = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)
    # TODO: Subtract by mean (see image_reader)
    image_rev = tf.reverse(image, tf.stack([1]))

    image_batch = tf.stack([image, image_rev])

    # Create network
    with tf.variable_scope('', reuse=False):
        net = PGNModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)

    # parsing net
    parsing_out1 = net.layers['parsing_fc']
    parsing_out2 = net.layers['parsing_rf_fc']

    # edge net
    edge_out2 = net.layers['edge_rf_fc']

    # combine resize
    parsing_out1 = tf.image.resize_images(parsing_out1, tf.shape(image_batch)[1: 3, ])
    parsing_out2 = tf.image.resize_images(parsing_out2, tf.shape(image_batch)[1: 3, ])

    edge_out2 = tf.image.resize_images(edge_out2, tf.shape(image_batch)[1: 3, ])

    raw_output = tf.reduce_mean(tf.stack([parsing_out1, parsing_out2]), axis=0)
    head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
    tail_list = tf.unstack(tail_output, num=20, axis=2)
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx]
    tail_list_rev[14] = tail_list[15]
    tail_list_rev[15] = tail_list[14]
    tail_list_rev[16] = tail_list[17]
    tail_list_rev[17] = tail_list[16]
    tail_list_rev[18] = tail_list[19]
    tail_list_rev[19] = tail_list[18]
    tail_output_rev = tf.stack(tail_list_rev, axis=2)
    tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))

    raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_output_all = tf.expand_dims(raw_output_all, dim=0)
    raw_output_all = tf.argmax(raw_output_all, axis=3)
    pred_all = tf.expand_dims(raw_output_all, dim=3)  # Create 4-d tensor.

    raw_edge = tf.reduce_mean(tf.stack([edge_out2]), axis=0)
    head_output, tail_output = tf.unstack(raw_edge, num=2, axis=0)
    tail_output_rev = tf.reverse(tail_output, tf.stack([1]))
    raw_edge_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_edge_all = tf.expand_dims(raw_edge_all, dim=0)

    # Which variables to load.
    restore_var = tf.global_variables()
    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()

        sess.run(init)
        sess.run(tf.local_variables_initializer())

        # Load weights.
        loader = tf.train.Saver(var_list=restore_var)

        ckpt = tf.train.get_checkpoint_state(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thirdparty/cihp_pgn/assets'))

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            loader.restore(sess, os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thirdparty/cihp_pgn/assets'), ckpt_name))

        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        segm_files = []

        for input_file in input_files:
            parsing_ = sess.run(pred_all)

            msk = decode_labels(parsing_, num_classes=N_CLASSES)
            parsing_im = Image.fromarray(msk[0])

            segm_files.append(np.array(parsing_im)[:, :, ::-1])

        coord.request_stop()

    return segm_files

def run_octopus(segm_frames, weights, in_dir, out_dir, opt_pose_steps, opt_shape_steps, openpose_model_dir):

    joints_2d, face_2d = [], []

    joints_2d, face_2d = openpose_process(in_dir, openpose_model_dir)

    K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))))

    model = Octopus(num=len(segm_frames))
    model.load(weights)

    segmentations = [read_segmentation_octopus(f) for f in segm_frames]

    if opt_pose_steps:
        #print('Optimizing for pose...')
        model.opt_pose(segmentations, joints_2d, opt_steps=opt_pose_steps)

    if opt_shape_steps:
        #print('Optimizing for shape...')
        model.opt_shape(segmentations, joints_2d, face_2d, opt_steps=opt_shape_steps)

    #print('Estimating shape...')
    pred = model.predict(segmentations, joints_2d)

    # Include texture coords in mesh
    vt = np.load(openpose_model_dir + 'basicModel_vt.npy')
    ft = np.load(openpose_model_dir + 'basicModel_ft.npy')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Output frame data if specified
    data_frame = write_frame_data_oct(pred['vertices'])

    return data_frame

def run_pic2tex(data, frame_dir, segm_frames, num_iter):

    # Step 1: Make unwraps

    if len(segm_frames) < 3:
        if len(segm_frames) < 2:
            segm_frames = np.array((segm_frames[0], segm_frames[0], segm_frames[0]))
        else:
            segm_frames = np.array((segm_frames[0], segm_frames[1], segm_frames[1]))

    frame_files = np.array(sorted(glob(os.path.join(frame_dir, '*.png')) + glob(os.path.join(frame_dir, '*.jpg'))))

    if len(frame_files) < 3:
        if len(frame_files) < 2:
            frame_files = np.array((frame_files[0], frame_files[0], frame_files[0]))
        else:
            frame_files = np.array((frame_files[0], frame_files[1], frame_files[1]))

    vt = np.load('thirdparty/pic2tex/assets/basicModel_vt.npy')
    ft = np.load('thirdparty/pic2tex/assets/basicModel_ft.npy')
    f = np.load('thirdparty/pic2tex/assets/basicModel_f.npy')

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
    for i, (v, frame_file, segm_file) in enumerate(zip(verts, frame_files, segm_frames)):
        frame = cv2.resize(cv2.imread(frame_file), (INPUT_RESIZE, INPUT_RESIZE)) / 255.
        segm = read_segmentation(segm_file) / 255.
        mask = np.float32(np.any(segm > 0, axis=-1))

        camera.set(v=v)

        vis, iso, iso_segm = texture.get_data(frame, camera, mask, segm)

        vises.append(vis)
        isos.append(iso)
        iso_segms.append(np.uint8(iso_segm * 255))

    # Step 2: Segm vote gmm

    iso_mask = cv2.imread('thirdparty/pic2tex/assets/tex_mask_1000.png', flags=cv2.IMREAD_GRAYSCALE) / 255.
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

    seams = np.load('thirdparty/pic2tex/assets/basicModel_seams.npy')
    edge_idx = pkl.load(open('thirdparty/pic2tex/assets/basicModel_edge_idx_1000_.pkl', 'rb'))

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

    seams = np.load('thirdparty/pic2tex/assets/basicModel_seams.npy')
    mask = cv2.imread('thirdparty/pic2tex/assets/tex_mask_1000.png', flags=cv2.IMREAD_GRAYSCALE) / 255.

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

    return np.uint8(255 * texture_agg)

def run_densepose(args_im_or_folder, args_cfg, args_weights):
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])

    merge_cfg_from_file(args_cfg)
    cfg.NUM_GPUS = 1
    args_weights = cache_url(args_weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine_initialize_model_from_cfg(args_weights)
    dummy_coco_dataset = dummy_datasets_get_coco_dataset()

    im_list = np.array(sorted(glob(os.path.join(args_im_or_folder, '*.png')) + glob(os.path.join(args_im_or_folder, '*.jpg'))))

    body_frames, dense_frames = [], []

    for im_name in im_list:
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        with c2_utils_NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine_im_detect_all(
                model, im, None, timers=timers
            )

        dense = vis_utils_vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
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

        body_frames.append(im)
        dense_frames.append(dense)

    return body_frames, dense_frames

def run_tex2shape(img_files, iuv_files, move_pose, weights_tex2shape, weights_betas):

    tex2shape_model = Tex2ShapeModel()
    betas_model = BetasModel()

    tex2shape_model.load(weights_tex2shape)
    betas_model.load(weights_betas)

    mfm = MeshFromMaps()
    savePos = []

    for img_file, iuv_file in zip(img_files, iuv_files):

        img = cv2.resize(img_file, (1024, 1024)) / 255.
        iuv_img = (cv2.resize(iuv_file, (1024, 1024)))
        unwrap = np.expand_dims(map_densepose_to_tex(img, iuv_img, 512), 0)

        iuv_img = iuv_img * 1.
        iuv_img[:, :, 1:] /= 255.
        iuv_img = np.expand_dims(iuv_img, 0)

        #print('> Estimating normal and displacement maps...')
        pred = tex2shape_model.predict(unwrap)

        #print('> Estimating betas...')
        betas = betas_model.predict(iuv_img)

        obj_frames = []

        for pose in tqdm(move_pose):
            
            mesh_obj = ""
            m = mfm.get_mesh(pred[0, :, :, :3], pred[0, :, :, 3:] / 10, betas=betas[0], pose=pose) 
            mesh_obj = mesh_write(mesh_obj, v=m['v'], f=m['f'], vn=m['vn'], vt=m['vt'], ft=m['ft'])

            obj_frames.append(mesh_obj)
        
        # savePos.append(obj_frames)

        if EXPORT_DECODED:
            if os.path.exists('assets_decoded/deformed_spml_model{}.obj'.format(len(savePos))):
                os.remove('assets_decoded/deformed_spml_model{}.obj'.format(len(savePos)))
                
            with open('assets_decoded/deformed_spml_model{}.obj'.format(len(savePos)),'w') as fp:
                fp.write(obj_frames[0])

        return obj_frames
    # return savePos[0]
    
if __name__ == '__main__':

    t = time.time()

    parser = ArgumentParser(description='PhysioMotionTransfer Python')
    parser.add_argument('-s1', '--cihp_pgn', action="store_true", help='Get body segmentation from samples')
    parser.add_argument('-s2', '--octopus', action="store_true", help='Get SMPL Body model params')
    parser.add_argument('-s3', '--pic2tex', action="store_true", help='Get model texture from samples')
    parser.add_argument('-s4', '--densepose', action="store_true", help='Get DensePose data from samples')
    parser.add_argument('-s5', '--demo_process', action="store_true", help='Get video frames pose data')
    parser.add_argument('-s6', '--tex2shape', action="store_true", help='Get SMPL Body model deformation')
    parser.add_argument('-s7', '--demo_run', action="store_true", help='Get final video frames')
    parser.add_argument('-s8', '--render_output', action="store_true", help='Render final video frames')

    args = parser.parse_args()
    
    if not os.path.exists('./assets'):
        os.makedirs('./assets')

    if args.cihp_pgn:

        if not os.path.exists('./assets_decoded'):
            os.makedirs('./assets_decoded')
        else:
            [os.remove('assets_decoded/' + file) for file in os.listdir('./assets_decoded')]

        from thirdparty.cihp_pgn.infer import *
        segm_body_data = run_cihp_pgn('body_samples/')

        pkl.dump(segm_body_data, open('assets/segm_body_data.txt', 'wb'))

        if EXPORT_DECODED:
            for i, img in enumerate(segm_body_data):
                cv2.imwrite('assets_decoded/segm_body_data{}.jpg'.format(i), img) 
    
    if args.octopus:
        from thirdparty.octopus.infer import *

        segm_body_data = pkl.load(open('assets/segm_body_data.txt', 'rb'))
        smpl_body_model = run_octopus(
            segm_body_data,
            'thirdparty/octopus/assets/octopus_weights.hdf5', 
            'body_samples',
            'output',
            5, 
            15, 
            'thirdparty/octopus/assets/'
        )

        pkl.dump(smpl_body_model, open('assets/smpl_body_model.txt', 'wb'))
    
    if args.pic2tex:
        from thirdparty.pic2tex.infer import *
        
        segm_body_data = pkl.load(open('assets/segm_body_data.txt', 'rb'))
        smpl_body_model = pkl.load(open('assets/smpl_body_model.txt', 'rb'))
        model_texture = run_pic2tex(
            smpl_body_model, 
            'body_samples',
            segm_body_data,
            15, 
        )

        pkl.dump(model_texture, open('assets/model_texture.txt', 'wb'))

        if EXPORT_DECODED:
            cv2.imwrite('assets_decoded/model_texture.jpg', model_texture)

    if args.densepose:
        from thirdparty.densepose.infer import *

        body_data, dense_body_data = run_densepose(
            'body_samples',
            'thirdparty/densepose/assets/config.yaml',
            'https://dl.fbaipublicfiles.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl'
        )

        pkl.dump(body_data, open('assets/body_data.txt', 'wb'))
        pkl.dump(dense_body_data, open('assets/dense_body_data.txt', 'wb'))

        if EXPORT_DECODED:
            for i, (img1, img2) in enumerate(zip(body_data, dense_body_data)):
                cv2.imwrite('assets_decoded/body_data{}.jpg'.format(i), img1)
                cv2.imwrite('assets_decoded/dense_body_data{}.jpg'.format(i), img2)

    if args.demo_process:
        from thirdparty.romp.infer import *
        demo = Demo()
        video_pose_data = demo.process_video('move_samples')

        pkl.dump(video_pose_data, open('assets/video_pose_data.txt', 'wb'))
    
    if args.tex2shape:
        from thirdparty.tex2shape.infer import *

        body_data = pkl.load(open('assets/body_data.txt', 'rb'))
        dense_body_data = pkl.load(open('assets/dense_body_data.txt', 'rb'))
        video_pose_data = pkl.load(open('assets/video_pose_data.txt', 'rb'))
        deformed_spml_model = run_tex2shape(
            body_data, 
            dense_body_data,
            video_pose_data,
            'thirdparty/tex2shape/assets/tex2shape_weights.hdf5',
            'thirdparty/tex2shape/assets/betas_weights.hdf5'
        )
        
        pkl.dump(deformed_spml_model, open('assets/deformed_spml_model.txt', 'wb'))

    if args.demo_run:
        from thirdparty.romp.infer import *

        deformed_spml_model = pkl.load(open('assets/deformed_spml_model.txt', 'rb'))
        model_texture = pkl.load(open('assets/model_texture.txt', 'rb'))
        demo = Demo()
        final_vid_frames = demo.run(deformed_spml_model, model_texture)

        pkl.dump(final_vid_frames, open('assets/final_vid_frames.txt', 'wb'))
    
    if args.render_output:

        final_vid_frames = pkl.load(open('assets/final_vid_frames.txt', 'rb'))
        
        if not os.path.exists('./output'):
            os.makedirs('./output')
        
        for id, image in tqdm(enumerate(final_vid_frames)):
            cv2.imwrite('output/{}.jpg'.format(id), image[0])

    print(time.time() - t)