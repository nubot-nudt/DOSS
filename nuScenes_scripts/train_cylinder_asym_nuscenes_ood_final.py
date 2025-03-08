# -*- coding:utf-8 -*-

import os
import time
import argparse
import sys
sys.path.append('..')
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_nuScenes_label_name
from builder import data_builder, model_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint
from torch.utils.tensorboard import SummaryWriter

from utils import loss_functions
import pickle
import yaml
import warnings
import json
import shutil

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
    model_save_path = train_hypers['model_save_path']
    model_latest_path = train_hypers['model_latest_path']

    log_save_path = os.path.dirname(model_latest_path)
    os.makedirs(log_save_path, exist_ok=True)

    # save args and config files
    with open(os.path.join(log_save_path, "args.json"), "w") as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    with open(os.path.join(log_save_path, "argsv.txt"), "w") as f:
        f.write(" ".join(sys.argv))
        f.write("\n")

    shutil.copy(src=config_path, dst=os.path.join(log_save_path, "config.yaml"))

    nuscenes_label_name = get_nuScenes_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(nuscenes_label_name.keys())))[1:] - 1
    unique_label_str = [nuscenes_label_name[x] for x in unique_label + 1]

    with open(dataset_config["label_mapping"], 'r') as stream:
        nuScenesyaml = yaml.safe_load(stream)
    ood_map = nuScenesyaml['ood_map']
    ood_map_tensor = torch.tensor(list(ood_map.values())).to(pytorch_device)

    ood_obj_map = nuScenesyaml['ood_obj_map']
    ood_obj_map_tensor = torch.tensor(list(ood_obj_map.values())).to(pytorch_device)

    # build model
    my_model = model_builder.build(model_config)

    if os.path.exists(model_load_path):
        print(f"Loading pre-trained model {model_load_path}")
        my_model = load_checkpoint(model_load_path, my_model)

    my_model.to(pytorch_device)
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)

    # build loss functions
    class_weighting = np.ones(num_class)

    loss_sem_func = loss_functions.CrossEntropyLoss(weight=class_weighting, device=pytorch_device)

    lovasz_softmax = loss_functions.build_lovasz(lovasz=True, num_class=num_class, ignore_label=ignore_label)

    loss_objectosphere_func = loss_functions.ObjectosphereLoss(sigma=1.0)

    loss_center_func = loss_functions.CenterLoss(n_classes=num_class)

    loss_contrastive_func = loss_functions.ContrastiveLoss(n_classes=num_class)

    # training
    epoch = 0
    best_val_miou = 0
    my_model.train()
    global_iter = 0
    val_global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']
    writer = SummaryWriter(log_save_path)

    while epoch < train_hypers['max_num_epochs']:
        loss_list = []
        pbar = tqdm(total=len(train_dataset_loader))

        mavs = None
        if epoch == 0 and os.path.exists(model_load_path):
            # load mavs and counts of CenterLoss for subsequent training (put mavs_current.pickle and previous_count.pickle in model_load dir)
            mav_load_path = os.path.join(os.path.dirname(model_load_path), "mavs_current.pickle")
            with open(mav_load_path, "rb") as h1:
                print("Storaged mav loaded.")
                mavs = pickle.load(h1)
                mav_tensor = torch.zeros(num_class, num_class).to(pytorch_device)
                for key in mavs.keys():
                    mav_tensor[key] = mavs[key]
                mavs = mav_tensor
            loss_center_func.previous_features = mavs

            count_load_path = os.path.join(os.path.dirname(model_load_path), "previous_count.pickle")
            with open(count_load_path, "rb") as h2:
                print("Storaged previous_count loaded.")
                previous_count = pickle.load(h2)
            loss_center_func.previous_count = previous_count
            
        if epoch and loss_contrastive_func is not None:
            mavs = loss_center_func.read()
            
        total_loss_list = []
        total_sem_loss = []
        total_obj_loss = []
        total_center_loss = []
        total_con_loss = []

        for i_iter, (_, train_vox_label, train_grid, _, train_pt_fea) in enumerate(train_dataset_loader):
            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
            train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]
            voxel_label_tensor = train_vox_label.type(torch.LongTensor).to(pytorch_device)

            voxel_label_tensor_obj = voxel_label_tensor.clone()
            # unknowns are not used for training
            # 0~12
            voxel_label_tensor = ood_map_tensor[voxel_label_tensor]

            # forward + backward + optimize

            # coor_ori: voxel coordinates of each point [pt_num, 4]
            # y_in_normal: output voxel feaures from semantic decoder [B, num_known_class, 480, 360, 32]
            # y_out_normal: output voxel feaures from open-set decoder [B, num_known_class, 480, 360, 32]
            coor_ori, y_in_normal, y_out_normal = my_model.forward_ow_complete(train_pt_fea_ten, train_vox_ten,
                                                                               train_batch_size)

            pt_label_origin = voxel_label_tensor[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)]

            # point-wise features
            y_in_normal_valid = y_in_normal.permute(0, 2, 3, 4, 1)
            y_in_normal_valid = y_in_normal_valid[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)].squeeze()

            y_out_normal_valid = y_out_normal.permute(0, 2, 3, 4, 1)
            y_out_normal_valid = y_out_normal_valid[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)].squeeze()

            loss_sem = loss_sem_func(y_in_normal_valid, pt_label_origin.squeeze())
            loss_lovasz = lovasz_softmax(torch.nn.functional.softmax(y_in_normal), (voxel_label_tensor - 1), ignore=-1)
            loss_sem = sum(loss_sem) + loss_lovasz

            pt_label_origin = pt_label_origin.long().cuda() - 1
            # 0~11, 66
            pt_label_origin[pt_label_origin < 0] = 66

            loss_center = loss_center_func(y_out_normal_valid, pt_label_origin.squeeze(), is_train=True)

            loss_con = loss_contrastive_func(mavs, y_out_normal, voxel_label_tensor, coor_ori, epoch)

            voxel_label_tensor_obj = ood_obj_map_tensor[voxel_label_tensor_obj]
            voxel_label_obj = voxel_label_tensor_obj[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)]
            loss_obj = loss_objectosphere_func(y_out_normal_valid, voxel_label_obj.squeeze())

            loss = torch.tensor(0)
            loss = loss + 1.0 * loss_sem + 0.5 * loss_obj + 0.3 * loss_center + 0.5 * loss_con

            # print(f"epoch={epoch} loss={loss} loss_sem={loss_sem} loss_obj={loss_obj}"
            #       f"loss_center={loss_center} loss_con={loss_con}")

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            
            loss = loss.cpu().detach().numpy()
            loss_sem = loss_sem.cpu().detach().numpy()
            loss_obj = loss_obj.cpu().detach().numpy()
            loss_center = loss_center.cpu().detach().numpy()
            loss_con = loss_con.cpu().detach().numpy()
            
            total_loss_list.append(loss)
            total_sem_loss.append(loss_sem)
            total_obj_loss.append(loss_obj)
            total_center_loss.append(loss_center)
            total_con_loss.append(loss_con)

            if global_iter % 1000 == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')

            optimizer.zero_grad()
            pbar.update(1)
            global_iter += 1
            if global_iter % check_iter == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')

        if loss_center_func is not None:
            mean = loss_center_func.update()
            print("Mean features updated.")
            # save mavs and counts to pickles
            with open(os.path.join(log_save_path, "mavs_current.pickle"), "wb") as h1:
                pickle.dump(mean, h1, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(log_save_path, "previous_count.pickle"), "wb") as h2:
                pickle.dump(loss_center_func.previous_count, h2, protocol=pickle.HIGHEST_PROTOCOL)

        # save the latest model
        torch.save(my_model.state_dict(), model_latest_path)
        
        writer.add_scalar("Loss/total_loss", np.mean(total_loss_list), epoch)
        writer.add_scalar("Loss/sem_loss", np.mean(total_sem_loss), epoch)
        writer.add_scalar("Loss/obj_loss", np.mean(total_obj_loss), epoch)
        writer.add_scalar("Loss/center_loss", np.mean(total_center_loss), epoch)
        writer.add_scalar("Loss/con_loss", np.mean(total_con_loss), epoch)

        pbar.close()
        epoch += 1

        # Evaluation
        print("\n ##########################Evaluating##########################")
        my_model.eval()
        hist_list = []
        val_loss_list = []

        ood_inv_map = nuScenesyaml['ood_inv_map']
        ood_inv_map_np = np.array(list(ood_inv_map.values()))
        
        total_loss_list = []
        total_sem_loss = []
        total_obj_loss = []
        total_center_loss = []
        total_con_loss = []

        if epoch and loss_contrastive_func is not None:
            mavs = loss_center_func.read()
        with torch.no_grad():
            pbar_val = tqdm(total=len(val_dataset_loader))
            for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, idx) in enumerate(
                    val_dataset_loader):

                val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                  val_pt_fea]
                val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
                val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

                val_label_tensor_obj = val_label_tensor.clone()

                val_label_tensor = ood_map_tensor[val_label_tensor]

                coor_ori, y_in_normal, y_out_normal = my_model.forward_ow_complete(val_pt_fea_ten,
                                                                                   val_grid_ten,
                                                                                   val_batch_size)

                pt_label_origin = val_label_tensor[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)]

                y_in_normal_valid = y_in_normal.permute(0, 2, 3, 4, 1)
                y_in_normal_valid = y_in_normal_valid[
                    coor_ori.permute(1, 0).chunk(chunks=4, dim=0)].squeeze()

                y_out_normal_valid = y_out_normal.permute(0, 2, 3, 4, 1)
                y_out_normal_valid = y_out_normal_valid[
                    coor_ori.permute(1, 0).chunk(chunks=4, dim=0)].squeeze()

                val_loss_sem = loss_sem_func(y_in_normal_valid, pt_label_origin.squeeze())
                val_loss_lovasz = lovasz_softmax(torch.nn.functional.softmax(y_in_normal), (voxel_label_tensor - 1),
                                             ignore=-1)
                val_loss_sem = sum(val_loss_sem) + val_loss_lovasz

                pt_label_origin = pt_label_origin.long().cuda() - 1
                pt_label_origin[pt_label_origin < 0] = 66

                val_loss_center = loss_center_func(y_out_normal_valid, pt_label_origin.squeeze(), is_train=False)

                val_loss_con = loss_contrastive_func(mavs, y_out_normal, val_label_tensor, coor_ori, epoch)

                val_label_tensor_obj = ood_obj_map_tensor[val_label_tensor_obj]
                voxel_label_obj = val_label_tensor_obj[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)]
                val_loss_obj = loss_objectosphere_func(y_out_normal_valid, voxel_label_obj.squeeze())

                loss_val = torch.tensor(0)
                loss_val = loss_val + 1.0 * val_loss_sem + 0.5 * val_loss_obj + 0.3 * val_loss_center + 0.5 * val_loss_con

                # print(f"loss_val={loss_val} val_loss_sem={val_loss_sem} val_loss_obj={val_loss_obj}"
                #       f"val_loss_center={val_loss_center} val_loss_con={val_loss_con}")

                y_in_normal = torch.argmax(y_in_normal, dim=1)
                y_in_normal = y_in_normal.cpu().detach().numpy()

                y_in_normal = ood_inv_map_np[y_in_normal]

                for count, i_val_grid in enumerate(val_grid):
                    hist_list.append(fast_hist_crop(y_in_normal[
                                                        count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                        val_grid[count][:, 2]], val_pt_labs[count],
                                                    unique_label))
                val_loss_list.append(loss_val.detach().cpu().numpy())
                pbar_val.update(1)
                val_global_iter += 1

                loss_val = loss_val.cpu().detach().numpy()
                val_loss_sem = val_loss_sem.cpu().detach().numpy()
                val_loss_obj = val_loss_obj.cpu().detach().numpy()
                val_loss_center = val_loss_center.cpu().detach().numpy()
                val_loss_con = val_loss_con.cpu().detach().numpy()
                
                total_loss_list.append(loss_val)
                total_sem_loss.append(val_loss_sem)
                total_obj_loss.append(val_loss_obj)
                total_center_loss.append(val_loss_center)
                total_con_loss.append(val_loss_con)

        my_model.train()
        iou = per_class_iu(sum(hist_list))
        print('Validation per class iou: ')
        for class_name, class_iou in zip(unique_label_str, iou):
            print('%s : %.2f%%' % (class_name, class_iou * 100))
        val_miou = np.nanmean(iou) * 100
        del val_vox_label, val_grid, val_pt_fea, val_grid_ten

        print('Current val miou is %.3f while the best val miou is %.3f' %
              (val_miou, best_val_miou))
        # save model if performance is improved
        if best_val_miou < val_miou:
            print("Better IoU, saving model...")
            best_val_miou = val_miou
            torch.save(my_model.state_dict(), model_save_path)

        writer.add_scalar("Loss_val/total_loss", np.mean(total_loss_list), epoch)
        writer.add_scalar("Loss_val/sem_loss", np.mean(total_sem_loss), epoch)
        writer.add_scalar("Loss_val/obj_loss", np.mean(total_obj_loss), epoch)
        writer.add_scalar("Loss_val/center_loss", np.mean(total_center_loss), epoch)
        writer.add_scalar("Loss_val/con_loss", np.mean(total_con_loss), epoch)
        
        writer.add_scalar('IoU/old_IoU', val_miou.item(), epoch)

        print('Current val loss is %.3f' %
              (np.mean(val_loss_list)))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='../config/nuScenes_ood_final.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
