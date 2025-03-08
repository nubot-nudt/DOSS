# -*- coding:utf-8 -*-

import os
import time
import argparse
import sys
sys.path.append('..')
import numpy as np
import torch
from tqdm import tqdm

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint

import yaml
import torch.nn.functional as F

import warnings

warnings.filterwarnings("ignore")


def main(args):
    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']

    pred_save_folder = os.path.join(args.save_folder, 'CSS_results/sequences/08/predictions/')
    ad_save_folder = os.path.join(args.save_folder, 'AnomalyDetection_results/sequences/08/predictions/')

    os.makedirs(pred_save_folder, exist_ok=True)
    os.makedirs(ad_save_folder, exist_ok=True)

    nuscenes_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(nuscenes_label_name.keys())))[1:] - 1
    unique_label_str = [nuscenes_label_name[x] for x in unique_label + 1]
    
    with open(dataset_config["label_mapping"], 'r') as stream:
        nuScenesyaml = yaml.safe_load(stream)
    ood_inv_map = nuScenesyaml['ood_inv_map']
    ood_inv_map_np = np.array(list(ood_inv_map.values()))

    my_model = model_builder.build(model_config)

    try:
        print(f"Loading pre-trained model {model_load_path}")
        my_model = load_checkpoint(model_load_path, my_model)
    except Exception as e:
        print(e)
        print("Error loading pre-trained model.")
        quit()

    my_model.to(pytorch_device)

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)

    my_model.eval()
    hist_list = []
    pbar = tqdm(total=len(val_dataset_loader))

    with torch.no_grad():
        for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, idx) in enumerate(
                val_dataset_loader):

            val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                              val_pt_fea]
            val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]

            coor_ori, y_in_normal, y_out_normal = my_model.forward_ow_complete(val_pt_fea_ten,
                                                                               val_grid_ten,
                                                                               val_batch_size)

            batch = 0

            idx_s = "%06d" % idx[0]

            y_in_normal = torch.argmax(y_in_normal, dim=1)
            y_in_normal = y_in_normal.cpu().detach().numpy()

            y_in_normal = ood_inv_map_np[y_in_normal]

            for count, i_val_grid in enumerate(val_grid):
                hist_list.append(fast_hist_crop(y_in_normal[
                                                    count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                    val_grid[count][:, 2]], val_pt_labs[count],
                                                unique_label))

            count = 0

            point_predict = y_in_normal[count, val_grid[count][:, 0], val_grid[count][:, 1],
                                        val_grid[count][:, 2]].astype(np.int32)

            point_predict.tofile(os.path.join(pred_save_folder, idx_s + '.label'))

            # Anomaly detection
            y_out_normal_pointwise = y_out_normal[batch, :, val_grid[batch][:, 0], val_grid[batch][:, 1],
                                   val_grid[batch][:, 2]].permute(1, 0).cpu().numpy()

            max_values = np.max(y_out_normal_pointwise, axis=1)
            conf_score = np.where(max_values <= 0.65, 1.0, 0.1)

            conf_score = conf_score.astype(np.float32)

            ad_save_file = os.path.join(ad_save_folder, idx_s + ".label")
            conf_score.tofile(ad_save_file)

            pbar.update(1)
    iou = per_class_iu(sum(hist_list))
    print('Validation per class iou: ')
    for class_name, class_iou in zip(unique_label_str, iou):
        print('%s : %.2f%%' % (class_name, class_iou * 100))
    val_miou = np.nanmean(iou) * 100
    del val_vox_label, val_grid, val_pt_fea, val_grid_ten

    print('Current val miou is %.3f.' %
          (val_miou))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='../config/nuScenes_ood_final.yaml')
    parser.add_argument('--save_folder', default='../ow3d_data/nuScenes/')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
