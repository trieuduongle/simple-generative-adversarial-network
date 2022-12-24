import numpy as np
# import matplotlib.pyplot as plt
import torch
import os
import os.path as osp
import json
import logging
import pickle
import time

# from torchvision.utils import make_grid
from tqdm import tqdm
from PIL import Image as im

from models.generator import Generator
from models.discriminator import Discriminator
from models.losses import GANLoss
from dataloader.dataloader import load_data
from utils import output_namespace, print_log, set_seed, check_dir
from recorder import Recorder
from metrics import metric


class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        self._preparation()
        print_log(output_namespace(self.args))

        self._get_data()
        self._select_optimizer()
        self._select_criterion()
        self._try_resume_trained_model(self.args.resume_path)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(0))
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the model
        self._build_model()

        self.lambda_adv = self.args.lambda_adv
        self.lambda_spatial_adv = self.args.lambda_spatial_adv

    def _build_model(self):
        args = self.args
        self.model = Generator(tuple(args.in_shape), args.hid_S,
                           args.hid_T, args.N_S, args.N_T).to(self.device)
        self.spatial_discriminator = Discriminator(args.image_channels, conv_by='2d').to(self.device)
        self.temporal_discriminator = Discriminator(args.image_channels).to(self.device)
    
    def _try_resume_trained_model(self, path):
        if path:
            if os.path.exists(path):
                print('resuming')
                checkpoint = torch.load(path)
                self.model.load_state_dict(checkpoint['GENERATOR_STATE_DICT'])
                self.generator_optimizer.load_state_dict(checkpoint['GENERATOR_OPTIMIZER_STATE_DICT'])
                self.spatial_discriminator.load_state_dict(checkpoint['SPATIAL_DISCRIMINATOR_STATE_DICT'])
                self.spatial_discriminator_optimizer.load_state_dict(checkpoint['SPATIAL_DISCRIMINATOR_OPTIMIZER_STATE_DICT'])
                self.temporal_discriminator.load_state_dict(checkpoint['TEMPORAL_DISCRIMINATOR_STATE_DICT'])
                self.temporal_discriminator_optimizer.load_state_dict(checkpoint['TEMPORAL_DISCRIMINATOR_OPTIMIZER_STATE_DICT'])
            else:
                raise (ValueError('Resume path does not exist'))

    def _get_data(self):
        config = self.args.__dict__
        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = load_data(**config)
        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader

    def _select_optimizer(self):
        self.generator_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)

        self.spatial_discriminator_optimizer = torch.optim.SGD(
            self.spatial_discriminator.parameters(),
            lr=self.args.lr_D, momentum=0.9
        )

        self.temporal_discriminator_optimizer = torch.optim.SGD(
            self.temporal_discriminator.parameters(),
            lr=self.args.lr_D, momentum=0.9
        )

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.generator_optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        return self.generator_optimizer
    
    def _select_criterion(self):
        self.generator_criterion = torch.nn.MSELoss()
        self.criterion_adv = GANLoss(device=self.device, gan_type='lsgan').to(self.device)

    def _save(self, name=''):
        torch.save({
            'GENERATOR_STATE_DICT': self.model.state_dict(),
            'GENERATOR_OPTIMIZER_STATE_DICT': self.generator_optimizer.state_dict(),
            'SPATIAL_DISCRIMINATOR_STATE_DICT': self.spatial_discriminator.state_dict(),
            'SPATIAL_DISCRIMINATOR_OPTIMIZER_STATE_DICT': self.spatial_discriminator_optimizer.state_dict(),
            'TEMPORAL_DISCRIMINATOR_STATE_DICT': self.temporal_discriminator.state_dict(),
            'TEMPORAL_DISCRIMINATOR_OPTIMIZER_STATE_DICT': self.temporal_discriminator.state_dict()
        }, os.path.join(
            self.checkpoints_path, name + '.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        pickle.dump(state, fw)

    def _predict(self, batch_x):
        if self.args.aft_seq_length == self.args.pre_seq_length:
            pred_y = self.model(batch_x)
        elif self.args.aft_seq_length < self.args.pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :self.args.aft_seq_length]
        elif self.args.aft_seq_length > self.args.pre_seq_length:
            pred_y = []
            d = self.args.aft_seq_length // self.args.pre_seq_length
            m = self.args.aft_seq_length % self.args.pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y

    def train(self, args):
        config = args.__dict__
        recorder = Recorder(verbose=True)

        # Training loop
        for epoch in range(config['epochs']):
            start_time = time.time()
            d_losses = []
            train_losses = []
            non_gan_loss = []
            spatial_discriminator_losses = []
            temporal_discriminator_losses = []

            train_pbar = tqdm(self.train_loader)

            for batch_x, batch_y in train_pbar:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                # Generate images in eval mode
                self.model.eval()
                with torch.no_grad():
                    pred_y = self._predict(batch_x)

                # #===============================
                # # Spatial Discriminator Network Training
                # #===============================

                self.spatial_discriminator.train()
                # Optimizer updates the discriminator parameters
                self.spatial_discriminator_optimizer.zero_grad()

                d_real = self.spatial_discriminator(self.merge_temporal_dim_to_batch_dim(batch_y), transpose=False)
                loss_d_real = self.criterion_adv(d_real, True, is_disc=True) * 0.5
                loss_d_real.backward()

                d_fake = self.spatial_discriminator(self.merge_temporal_dim_to_batch_dim(pred_y.detach()), transpose=False)
                loss_d_fake = self.criterion_adv(d_fake, False, is_disc=True) * 0.5
                loss_d_fake.backward()
                self.spatial_discriminator_optimizer.step()

                d_loss = loss_d_real + loss_d_fake
                spatial_discriminator_losses.append(d_loss.item())

                # #===============================
                # # Temporal Discriminator Network Training
                # #===============================

                self.temporal_discriminator.train()
                d_real = self.temporal_discriminator(batch_y)
                loss_d_real = self.criterion_adv(d_real, True, is_disc=True) * 0.5
                loss_d_real.backward()

                d_fake = self.temporal_discriminator(pred_y.detach())
                loss_d_fake = self.criterion_adv(d_fake, False, is_disc=True) * 0.5
                loss_d_fake.backward()

                # Optimizer updates the discriminator parameters
                self.temporal_discriminator_optimizer.zero_grad()

                d_real = self.temporal_discriminator(batch_y)
                loss_d_real = self.criterion_adv(d_real, True, is_disc=True) * 0.5
                loss_d_real.backward()

                d_fake = self.temporal_discriminator(pred_y.detach())
                loss_d_fake = self.criterion_adv(d_fake, False, is_disc=True) * 0.5
                loss_d_fake.backward()
                self.temporal_discriminator_optimizer.step()

                d_loss = loss_d_real + loss_d_fake
                temporal_discriminator_losses.append(d_loss.item())

                #===============================
                # Generator Network Training
                #===============================

                # Generate images in train mode
                self.model.train()
                self.spatial_discriminator.eval()
                self.temporal_discriminator.eval()
                pred_y = self._predict(batch_x)

                loss = self.generator_criterion(pred_y, batch_y)
                non_gan_loss.append(loss.item())

                adv_loss = self.criterion_adv(
                    self.spatial_discriminator(self.merge_temporal_dim_to_batch_dim(pred_y), transpose=False),
                    True,
                    is_disc=False
                )
                adv_loss = adv_loss * self.lambda_spatial_adv
                loss += adv_loss

                adv_loss = self.criterion_adv(self.temporal_discriminator(pred_y), True, is_disc=False)
                adv_loss = adv_loss * self.lambda_adv
                loss += adv_loss

                # Optimizer updates the generator parameters
                self.generator_optimizer.zero_grad()
                loss.backward()
                self.generator_optimizer.step()

                # Keep losses for logging
                # d_losses.append(d_loss.item())
                train_losses.append(loss.item())

                train_pbar.set_description(
                    f'Train loss: {train_losses[-1]:.9f} - Generator Loss {non_gan_loss[-1]:.9f} - Spatial Loss: {spatial_discriminator_losses[-1]:.9f} - Temporal Loss {temporal_discriminator_losses[-1]:.9f}'
                )

            train_loss_average = np.average(train_losses)
            spatial_loss_average = np.average(spatial_discriminator_losses)
            temporal_discriminator_average = np.average(temporal_discriminator_losses)
            generator_loss_average = np.average(non_gan_loss)
            if epoch % args.log_step == 0:
                with torch.no_grad():
                    vali_loss = self.vali(self.vali_loader, epoch)
                    self.interpolate(epoch + 1)
                print_log(f"Epoch: {epoch + 1} | Train Loss: {train_loss_average:.4f} - Generator Loss {generator_loss_average:.9f} - Spatial Loss: {spatial_loss_average:.9f} - Temporal Loss {temporal_discriminator_average:.9f} - Vali Loss: {vali_loss:.4f} | Take {(time.time() - start_time):.4f} seconds\n")

                recorder(vali_loss, self.model, self.path)

            if epoch % args.save_epoch_freq == 0:
                self._save(name=str(epoch + 1))

    def vali(self, vali_loader):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            if i * batch_x.shape[0] > 1000:
                break

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self.model(batch_x)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 pred_y, batch_y], [preds_lst, trues_lst]))

            loss = self.generator_criterion(pred_y, batch_y)
            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)

        mse, mae, ssim, psnr = metric(preds, trues, vali_loader.dataset.mean, vali_loader.dataset.std, True)
        print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))
        self.model.train()
        return total_loss
    
    def interpolate(self, sub_folder):
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        for batch_x, batch_y in self.test_loader:
            pred_y = self.model(batch_x.to(self.device))
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))
            break
        
        inputs, trues, preds = map(lambda data: np.concatenate(
            data, axis=0), [inputs_lst, trues_lst, preds_lst])
        print(preds.shape)

        folder_path = self.path+f'/interpolation/{sub_folder}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for index,pred in enumerate(preds[0]):
            data = im.fromarray(np.uint8(np.squeeze(np.array(pred).transpose(1,2,0)) * 255))
            data.save(os.path.join(folder_path,'pred_'+ str(index) + '.png'))

        for index,pred in enumerate(inputs[0]):
            data = im.fromarray(np.uint8(np.squeeze(np.array(pred).transpose(1,2,0)) * 255))
            
            data.save(os.path.join(folder_path,'input_'+ str(index) + '.png'))

        for index,pred in enumerate(trues[0]):
            data = im.fromarray(np.uint8(np.squeeze(np.array(pred).transpose(1,2,0)) * 255))
            data.save(os.path.join(folder_path,'trues_'+ str(index) + '.png'))

    def merge_temporal_dim_to_batch_dim(self, inputs):
        in_shape = list(inputs.shape)
        return inputs.view([in_shape[0] * in_shape[1]] + in_shape[2:])