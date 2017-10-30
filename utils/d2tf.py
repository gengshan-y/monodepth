from evaluation_utils import read_text_lines, read_file_data, generate_depth_map, get_focal_length_baseline
import cv2
# from matplotlib import pyplot as plt
import numpy as np
import os, os.path
import pdb
#from skimage import io, exposure, img_as_uint, img_as_float
#from tfrecord_utils import imwrite_tf, imread_tf_run

# Taken from https://stackoverflow.com/a/600612/119527
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if os.path.isdir(path):
            pass
        else: raise

test_files = read_text_lines( "%s/%s" % ('filenames','eigen_train_files.txt') )
gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, '/ssd0/KITTI/')


for t_id in range(len(gt_files)):
    gt_depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], cams[t_id], False, True)
    mask = np.logical_and(gt_depth < 80, gt_depth > 1e-3)
    gt_height, gt_width = gt_depth.shape
    crop = np.array([0.3324324 * gt_height,  0.91351351 * gt_height,
                                     0.0359477 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
    mask = np.logical_and(mask, crop_mask)
    gt_depth[~mask] = -1

    disp = np.zeros(gt_depth.shape)
    focal_length, baseline = get_focal_length_baseline(gt_calib[t_id], cams[t_id])
    disp[mask] = (baseline * focal_length) / gt_depth[mask]
    # disp /= gt_width

    sep1 = gt_files[t_id].find('sync') + 4
    sep2 = im_files[t_id].find('.jpg') -11

    str1 = gt_files[t_id][:sep1]
    dir1 = '%s/%s' % (str1, 'depth_0')
    mkdir_p(dir1)
    imname = im_files[t_id][sep2:]
    impath = '%s/%s' % (dir1,imname)

    # pdb.set_trace()
    # cv2.imwrite(impath, disp)
    gt_depth = gt_depth[crop[0]:crop[1],crop[2]:crop[3]]
    cv2.imwrite(impath, gt_depth)
    #im = exposure.rescale_intensity(disp, out_range='float')
    #im = img_as_uint(im)
    #io.imsave(impath, im)
    
    #t = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
    #print np.mean(abs(t[mask]-disp[mask]))

    #imwrite_tf(disp, impath)
    #t = imread_tf_run(impath)

    if t_id % 1000 == 0:
      print '%d images left' % (len(gt_files) - t_id)
