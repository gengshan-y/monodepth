from evaluation_utils import *
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nbins',type=int,default=10)
args = parser.parse_args()

test_files = read_text_lines( "%s/%s" % ('filenames','eigen_test_files.txt') )
gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, '/data/gengshay/KITTI/')

max_depth = 80
min_depth = 1# 1e-3
nbins = args.nbins

max_disp = 350./min_depth
min_disp = 350./max_depth

# bins = np.linspace(min_depth,max_depth,num=nbins)
# bins = np.logspace(np.log(min_depth),np.log(max_depth),num=nbins,base=np.e)
# bins = np.logspace(np.log(min_disp),np.log(max_disp),num=nbins,base=np.e)
bins = np.linspace(min_disp,max_disp,num=nbins)

num_samples = len(im_files)
gt_depths = []
pred_depths = []
for t_id in range(num_samples):
    camera_id = cams[t_id]  # 2 is left, 3 is right
    gt_depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True)
    gt_depths.append(gt_depth.astype(np.float32))
    
    
    mask = np.logical_and(gt_depth < max_depth, gt_depth > min_depth)
    gt_depth[~mask] = -1

    # disparity
    disp = -1 * np.ones(gt_depth.shape)
    focal_length, baseline = get_focal_length_baseline(gt_calib[t_id], cams[t_id])
    disp[mask] = (baseline * focal_length) / gt_depth[mask]
    disp_pred = np.digitize(disp,bins)-1
    tmp=(bins[np.clip(disp_pred+1, 0,nbins-1)]+bins[disp_pred])/2
    depth_pred = (baseline * focal_length) / tmp
    
    # depth
    #depth_pred = np.digitize(gt_depth,bins)-1
    #depth_pred =( bins[depth_pred]+bins[np.clip(depth_pred+1, 0,nbins-1)])/2

    pred_depths.append(depth_pred)
    

rms     = np.zeros(num_samples, np.float32)
log_rms = np.zeros(num_samples, np.float32)
abs_rel = np.zeros(num_samples, np.float32)
sq_rel  = np.zeros(num_samples, np.float32)
d1_all  = np.zeros(num_samples, np.float32)
a1      = np.zeros(num_samples, np.float32)
a2      = np.zeros(num_samples, np.float32)
a3      = np.zeros(num_samples, np.float32)
abs_rel_a = []
sq_rel_a = []
a_a = []
abs_a = []
sq_a = []
log_sq_a = []

for i in range(num_samples):
    gt_depth = gt_depths[i]
    pred_depth = pred_depths[i]

    pred_depth[pred_depth < min_depth] = min_depth
    pred_depth[pred_depth > max_depth] = max_depth
    
    mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)

    gt_height, gt_width = gt_depth.shape

    crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,   
                     0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
    mask = np.logical_and(mask, crop_mask)
    abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])
    
    abs_rel_a.append( np.abs(gt_depth[mask] - pred_depth[mask])/ gt_depth[mask])
    abs_a.append( np.abs(gt_depth[mask] - pred_depth[mask]))
    sq_rel_a.append(np.square(gt_depth[mask] - pred_depth[mask])/ gt_depth[mask])
    sq_a.append(np.square(gt_depth[mask] - pred_depth[mask]))
    log_sq_a.append(np.square(np.log(gt_depth[mask]) - np.log(pred_depth[mask])))
    a_a.append(np.max([gt_depth[mask] / pred_depth[mask],pred_depth[mask] / gt_depth[mask]],0))


# print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
print('nbins=%d'%args.nbins)
print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))
