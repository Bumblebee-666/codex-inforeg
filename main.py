import argparse
# 666777
import os
import copy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pdb
import torch.nn.functional as F
from dataset.CramedDataset import CramedDataset
from utils.cmob import CMoB
from models.focal_loss import FocalLoss
from dataset.dataset import AVDataset
from models.basic_model import *
from utils.utils import setup_seed, weight_init
import torchvision
import os
import pickle
import csv
from datetime import datetime
from KSDataset import *
import matplotlib.pyplot as plt
import numpy as np
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--modulation', default='Ours', type=str,

                        choices=['Normal', 'OGM', 'OGM_GE',"Ours"])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=3, type=int)
    parser.add_argument('--use_video_frames', default=3, type=int)
    parser.add_argument('--audio_path', default='/data/Lab105/Datasets/CREMA-D/CREMAD/AudioWAV/', type=str)
    parser.add_argument('--visual_path', default='/data/Lab105/Datasets/CREMA-D/CREMAD/', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.002, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=30, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')
    parser.add_argument('--alpha', default=0.8, type=float, help='alpha in OGM-GE')
    
    parser.add_argument('--ckpt_path', default= r'./ckpt', type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=True, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', default='./logs',type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='1', type=str, help='GPU ids')
    parser.add_argument('--audio_fim_path',default='./audio_fim_folder',type=str,help='path to store the audio fim')
    parser.add_argument('--visual_fim_path',default='./visual_fim_folder',type=str,help='path to store the visual fim')
    parser.add_argument('--mm_fim_path',default='./mm_fim_folder',type=str,help='path to store the mm fim')
    parser.add_argument('--accuracy_path',default='./accuracy_folder',type=str,help='path to store accuracy csv file')
    parser.add_argument('--use_cmob', action='store_true', help='Use Causal Modality Valuation for balancing')
    parser.add_argument('--use_sample_weighting', action='store_true', help='Use Sample-Level Weighting (Focal Loss)')
    parser.add_argument('--use_cross_gate', action='store_true', help='Use Cross-Gate modulation module')
    parser.add_argument('--cross_gate_strength', default=1.0, type=float, help='Strength of Cross-Gate modulation')
    parser.add_argument('--cross_gate_dropout', default=0.0, type=float, help='Dropout rate inside Cross-Gate module')
    parser.add_argument('--cross_gate_strength_audio', default=1.0, type=float, help='Cross-Gate update strength for audio branch')
    parser.add_argument('--cross_gate_strength_visual', default=1.15, type=float, help='Cross-Gate update strength for visual branch')
    parser.add_argument('--cross_gate_temp', default=1.0, type=float, help='Gate temperature in Cross-Gate module')
    parser.add_argument('--focal_gamma', default=2.0, type=float, help='Gamma for Focal Loss (deprecated, use start/end)')
    parser.add_argument('--focal_gamma_start', default=0.0, type=float, help='Starting gamma for Focal Loss')
    parser.add_argument('--focal_gamma_end', default=2.0, type=float, help='Ending gamma for Focal Loss')
    parser.add_argument('--sw_mode', default='epoch_range', type=str, choices=['prime_window', 'always', 'epoch_range'])
    parser.add_argument('--sw_k_threshold', default=0.05, type=float)
    parser.add_argument('--sw_start_epoch', default=5, type=int)
    parser.add_argument('--sw_end_epoch', default=30, type=int)
    parser.add_argument('--lambda_audio', default=1.0, type=float, help='Weight for audio branch loss')
    parser.add_argument('--lambda_visual', default=1.25, type=float, help='Weight for visual branch loss')
    parser.add_argument('--cmob_beta_cap', default=1.5, type=float, help='Upper bound for CMoB beta scaling')
    parser.add_argument('--cmob_beta_warmup_epochs', default=5, type=int, help='Warmup epochs before full CMoB effect')
    parser.add_argument('--cmob_ema_momentum', default=0.9, type=float, help='EMA momentum for CMoB gap smoothing')
    parser.add_argument('--cmob_visual_boost', default=0.2, type=float, help='Extra visual branch support when audio dominates')
    parser.add_argument('--cmob_gap_scale', default=1.0, type=float, help='Scale factor applied to CMoB gap')
    parser.add_argument('--cmob_audio_grad_scale', default=1.0, type=float, help='Gradient scale for audio when visual dominates (>1 boosts audio learning)')
    parser.add_argument('--best_target_acc', default=0.76, type=float, help='Target total accuracy threshold for joint-best checkpointing')
    parser.add_argument('--best_target_audio_acc', default=0.615, type=float, help='Target audio accuracy threshold for joint-best checkpointing')
    parser.add_argument('--best_target_visual_acc', default=0.615, type=float, help='Target visual accuracy threshold for joint-best checkpointing')

    return parser.parse_args()


def compute_trace_k(trace_list, window=10):
    if trace_list is None or len(trace_list) < window + 1:
        return 0.0

    current_window = trace_list[-window:]
    previous_window = trace_list[-(window + 1):-1]
    current_mean = float(np.mean(current_window))
    previous_mean = float(np.mean(previous_window))
    denominator = max(abs(current_mean), 1e-8)
    return (current_mean - previous_mean) / denominator


def save_training_checkpoint(args, model, optimizer, scheduler, epoch, acc, acc_a, acc_v, tag='best_model'):
    os.makedirs(args.ckpt_path, exist_ok=True)
    acc_str = '{:.4f}'.format(acc)
    acc_a_str = '{:.4f}'.format(acc_a)
    acc_v_str = '{:.4f}'.format(acc_v)

    model_name = '{}_of_dataset_{}_{}_alpha_{}_' \
                 'optimizer_{}_training_epochs_{}_' \
                 'epoch_{}_acc_{}_acca_{}_accv_{}.pth'.format(tag,
                                                              args.dataset,
                                                              args.modulation,
                                                              args.alpha,
                                                              args.optimizer,
                                                              args.modulation_starts,
                                                              epoch,
                                                              acc_str,
                                                              acc_a_str,
                                                              acc_v_str)

    saved_dict = {
        'saved_epoch': epoch,
        'modulation': args.modulation,
        'alpha': args.alpha,
        'fusion': args.fusion_method,
        'acc': acc,
        'acc_a': acc_a,
        'acc_v': acc_v,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }

    save_dir = os.path.join(args.ckpt_path, model_name)
    torch.save(saved_dict, save_dir)
    return save_dir



def freeze_audio_net(model):
    if isinstance(model, nn.DataParallel):
        model = model.module

    for param in model.audio_net.parameters():
        param.requires_grad = False

def freeze_head(model):
    if isinstance(model, nn.DataParallel):
        model = model.module

    for param in model.head.parameters():
        param.requires_grad = False

def open_head(model):
    if isinstance(model, nn.DataParallel):
        model = model.module

    for param in model.head.parameters():
        param.requires_grad = True




def open_audio_net(model):
    if isinstance(model, nn.DataParallel):
        model = model.module

    for param in model.audio_net.parameters():
        param.requires_grad = True

def freeze_visual_net(model):
    if isinstance(model, nn.DataParallel):
        model = model.module

    for param in model.visual_net.parameters():
        param.requires_grad = False
        

def open_visual_net(model):
    if isinstance(model, nn.DataParallel):
        model = model.module

    for param in model.visual_net.parameters():
        param.requires_grad = True




def calculate_weight_mean(network):
    total_sum = 0
    total_count = 0

    for param in network.parameters():
        if param.requires_grad:
            total_sum += param.data.sum().item()
            total_count += param.data.numel()

    # 计算并返回权重的平均值
    weight_mean = total_sum / total_count if total_count > 0 else 0
    return weight_mean

def calculate_audio_visual_weight_mean(model):
    # 如果使用 DataParallel，获取原始模型
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # 计算 audio_net 的权重平均值
    audio_mean = calculate_weight_mean(model.audio_net)

    # 计算 visual_net 的权重平均值
    visual_mean = calculate_weight_mean(model.visual_net)

    return audio_mean, visual_mean


def calculate_proximal_term(model, global_model , epoch):
    proximal_term = 0.0

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if isinstance(global_model, torch.nn.DataParallel):
        global_model = global_model.module

    for w, w_t in zip(model.audio_net.parameters(), global_model.audio_net.parameters()):
        proximal_term += (w - w_t).norm(2)  
        if(epoch < 10):
            return (1/2) * proximal_term
        elif(epoch < 20):
            return (1/2) * proximal_term
        elif(epoch < 30):
            return (1/2) * proximal_term
        else:
            return (1/2) * proximal_term


def calculate_proximal_term2(model, global_model , epoch):
    proximal_term = 0.0

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if isinstance(global_model, torch.nn.DataParallel):
        global_model = global_model.module
    for w, w_t in zip(model.visual_net.parameters(), global_model.visual_net.parameters()):
        proximal_term += (w_t - w).norm(2) 
        if(epoch < 10):
            return (5/2) * proximal_term
        elif(epoch < 20):
            return (5/2) * proximal_term
        elif(epoch < 30):
            return (1/2) * proximal_term
        else:
            return (1/2) * proximal_term



def log_images_to_tensorboard(writer, dataloader, device, phase='train', num_images=16):
    # Get a batch of images and labels
    spec, images, labels = next(iter(dataloader))
    
    # Move images to the device (if using GPU)
    images = images.to(device)
    # Squeeze the extra dimension (batch_size, 3, 1, 224, 224) -> (batch_size, 3, 224, 224)
    images = images.squeeze(2)
    
    # Select the first `num_images` images and labels to log
    img_grid = torchvision.utils.make_grid(images[:num_images])
    
    # Add images to TensorBoard
    writer.add_image(f'{phase}_images', img_grid)
    
    # If you want to add labels, make sure they are in the proper format (e.g., string)
    # Here we'll just log labels as scalars (for simplicity)
    for i in range(min(num_images, len(labels))):
        writer.add_text(f'{phase}_label_{i}', str(labels[i].item()), global_step=0)


def log_images_to_tensorboard(writer, dataloader, device, phase='train'):
    for step, (spec, images, labels) in enumerate(dataloader):
        # Move images to the device (if using GPU)
        images = images.to(device)
        
        # Squeeze the extra dimension (batch_size, 3, 1, 224, 224) -> (batch_size, 3, 224, 224)
        images = images.squeeze(2)
        
        # Log each image and its label separately
        for i in range(len(images)):
            # Add each individual image to TensorBoard
            writer.add_image(f'{phase}_image_{step}_{i}', images[i], global_step=step)
            
            # Add the corresponding label as text
            writer.add_text(f'{phase}_label_{step}_{i}', str(labels[i].item()), global_step=step)
        
        # Optionally, break the loop after a few steps if you don't want to log the whole dataset
        if step == 10:
            break







def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, writer=None, visual_trace_list=None, audio_trace_list=None,epoch_data_lists=None):
    cmob_tool = CMoB(device) if args.use_cmob else None
    audio_trace_list = audio_trace_list if audio_trace_list is not None else []
    visual_trace_list = visual_trace_list if visual_trace_list is not None else []
    total_audio_grad_sum = 0
    total_visual_grad_sum = 0
    total_audio_count = 0
    total_visual_count = 0
    alpha1 = 0 
    k = compute_trace_k(audio_trace_list)
        


    
    
    if epoch < 0:
        mu_v = 1
        mu_a = 2.5
    else:
        mu_a =0
    
    ce_criterion = nn.CrossEntropyLoss()
    focal_criterion = None
    current_gamma = 0.0
    sw_enabled = False
    if args.use_sample_weighting:
        if args.sw_mode == 'always':
            sw_enabled = True
        elif args.sw_mode == 'epoch_range':
            sw_enabled = (epoch >= args.sw_start_epoch) and (epoch <= args.sw_end_epoch)
        else:
            sw_enabled = (len(audio_trace_list) >= 11) and (k > args.sw_k_threshold)

        if sw_enabled:
            if args.sw_mode == 'epoch_range':
                progress = min(1.0, max(0.0, (epoch - args.sw_start_epoch) / max(1, args.sw_end_epoch - args.sw_start_epoch)))
            else:
                progress = min(1.0, epoch / max(1, args.lr_decay_step))
            current_gamma = args.focal_gamma_start + (args.focal_gamma_end - args.focal_gamma_start) * progress
            focal_criterion = FocalLoss(gamma=current_gamma)

    if writer is not None:
        writer.add_scalar('training/trace_k', k, epoch)
        writer.add_scalar('training/sw_enabled', float(sw_enabled), epoch)
        writer.add_scalar('training/focal_gamma', current_gamma, epoch)

    def branch_criterion_fn(logits, targets):
        if args.use_sample_weighting and sw_enabled and focal_criterion is not None:
            return focal_criterion(logits, targets)
        return ce_criterion(logits, targets)
    
    
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()
    global_model = cp.deepcopy(model)
    visual_cos_similarity = 0
    audio_cos_similarity = 0
    mm_cos_similarity = 0
    fisher_matrix = None
    ema_gap = None
    # optimizer_audio = optim.SGD(model.module.audio_net.parameters(), lr=args.learning_rate, momentum=0.9)
    
    # scheduler_audio = optim.lr_scheduler.StepLR(optimizer_audio, args.lr_decay_step, args.lr_decay_ratio)

    model.train()
    print("Start training ... ")

    record_names_audio = []
    record_names_visual = []
    for name, param in model.named_parameters():
        if 'head' in name: 
            continue
        if ('audio' in name):
            record_names_audio.append((name, param))
            continue
        if ('visual' in name):
            record_names_visual.append((name, param))
            continue

    _loss_a = 0
    _loss_v = 0
    

    _loss = 0



    


    
    fim = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            if module.weight.requires_grad:
                fim[name] = torch.zeros_like(module.weight)
    fim_audio = {}
    fim_visual = {}
    fim_audio_head = {}
    fim_visual_head = {}
    fim_mm_head = {}


    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if 'audio_net' in name:
                if module.weight.requires_grad:
                    fim_audio[name] = torch.zeros_like(module.weight)
            elif 'visual_net' in name:
                if module.weight.requires_grad:
                    fim_visual[name] = torch.zeros_like(module.weight)
            elif 'head_audio' in name:
                if module.weight.requires_grad:
                    fim_audio_head[name] = torch.zeros_like(module.weight)
            elif 'head_video' in name:
                if module.weight.requires_grad:
                    fim_visual_head[name] = torch.zeros_like(module.weight)
            elif 'head' in name:
                if module.weight.requires_grad:
                    fim_mm_head[name] = torch.zeros_like(module.weight)
            

    
    w_a,w_v = calculate_audio_visual_weight_mean(model)
    
    writer.add_scalar('w_a',w_a,epoch)
    writer.add_scalar('w_v',w_v,epoch)

    weights_head = model.module.head.weight.data
    weights_audio_head = model.module.head_audio.weight.data
    weights_video_head = model.module.head_video.weight.data
    cos_similarity_audio_head = F.cosine_similarity(
        weights_audio_head, weights_head[:, :512], dim=1).mean().item()

    cos_similarity_video_head = F.cosine_similarity(
        weights_video_head, weights_head[:, 512:], dim=1).mean().item()
    cos_similarity_head = F.cosine_similarity(
        weights_head[:, :512], weights_head[:, 512:], dim=1).mean().item()

    writer.add_scalar('cos_similarity_audio_head',cos_similarity_audio_head,epoch)
    writer.add_scalar('cos_similarity_video_head',cos_similarity_video_head,epoch)
    writer.add_scalar('cos_similarity_head',cos_similarity_head,epoch)
    

    # fim = compute_fim(model, criterion, dataloader, device)
    # log_images_to_tensorboard(writer, dataloader, device, phase='train', num_images=16)


    for step, (spec, image, label) in enumerate(dataloader):

        #pdb.set_trace()
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        lr = optimizer.param_groups[0]['lr']

        # TODO: make it simpler and easier to extend
        outputs = model(spec.unsqueeze(1).float(), image.float())
        a, v, out_a, out_v, out = outputs[:5]

        # if args.fusion_method == 'sum':
        #     out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
        #              model.module.fusion_module.fc_y.bias)
        #     out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
        #              model.module.fusion_module.fc_x.bias)
        # else:
        #     weight_size = model.module.fusion_module.fc_out.weight.size(1)
        #     out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
        #              + model.module.fusion_module.fc_out.bias / 2)

        #     out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
        #              + model.module.fusion_module.fc_out.bias / 2)

        loss = ce_criterion(out, label)
        loss_out_v = branch_criterion_fn(out_v, label)
        loss_out_a = branch_criterion_fn(out_a, label)
        weighted_loss_out_a = args.lambda_audio * loss_out_a
        weighted_loss_out_v = args.lambda_visual * loss_out_v

        if epoch_data_lists is not None:
            with torch.no_grad():
                softmax_func = torch.nn.Softmax(dim=1)
            
                softmax_v = softmax_func(out_v)
                avg_conf_v = torch.mean(softmax_v.gather(1, label.view(-1, 1))).item()
                _, predicted_labels_v = torch.max(softmax_v, dim=1)
                batch_acc_v = torch.mean((predicted_labels_v == label).float()).item()
                epoch_data_lists['conf_v'].append(avg_conf_v)
                epoch_data_lists['acc_v'].append(batch_acc_v)

                softmax_a = softmax_func(out_a)
                avg_conf_a = torch.mean(softmax_a.gather(1, label.view(-1, 1))).item()
                _, predicted_labels_a = torch.max(softmax_a, dim=1)
                batch_acc_a = torch.mean((predicted_labels_a == label).float()).item()
                epoch_data_lists['conf_a'].append(avg_conf_a)
                epoch_data_lists['acc_a'].append(batch_acc_a)


        losses=[loss, weighted_loss_out_a, weighted_loss_out_v]
        all_loss = ['both', 'audio', 'visual']
        grads_audio = {}
        grads_visual={}

        for idx, loss_type in enumerate(all_loss):
            loss_tem = losses[idx]
            loss_tem.backward(retain_graph=True)
            if(loss_type=='visual'):
                for tensor_name, param in record_names_visual:
                    if loss_type not in grads_visual.keys():
                        grads_visual[loss_type] = {}
                    if param.grad is not None:
                        grads_visual[loss_type][tensor_name] = param.grad.data.clone() 
                    else:
                        grads_visual[loss_type][tensor_name] = torch.zeros_like(param.data)
                grads_visual[loss_type]["concat"] = torch.cat([grads_visual[loss_type][tensor_name].flatten()  for tensor_name, _ in record_names_visual])           
                average_grad_visual = torch.mean(grads_visual[loss_type]["concat"]).item()
                writer.add_scalar('average_grad_visual',average_grad_visual,epoch)
            elif(loss_type=='audio'):
                for tensor_name, param in record_names_audio:
                    if loss_type not in grads_audio.keys():
                        grads_audio[loss_type] = {}
                    if param.grad is not None:
                        grads_audio[loss_type][tensor_name] = param.grad.data.clone() 
                    else:
                        grads_audio[loss_type][tensor_name] = torch.zeros_like(param.data)
                grads_audio[loss_type]["concat"] = torch.cat([grads_audio[loss_type][tensor_name].flatten()  for tensor_name, _ in record_names_audio])
                average_grad_audio = torch.mean(grads_audio[loss_type]["concat"]).item()
                writer.add_scalar('average_grad_audio',average_grad_audio,epoch)
            else:
                for tensor_name, param in record_names_audio:
                    if loss_type not in grads_audio.keys():
                        grads_audio[loss_type] = {}
                    if param.grad is not None:
                        grads_audio[loss_type][tensor_name] = param.grad.data.clone() 
                    else:
                        grads_audio[loss_type][tensor_name] = torch.zeros_like(param.data)
                grads_audio[loss_type]["concat"] = torch.cat([grads_audio[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_audio])
                average_grad_audio_mm = torch.mean(grads_audio[loss_type]["concat"]).item()
                writer.add_scalar('average_grad_audio_mm',average_grad_audio_mm,epoch)
                for tensor_name, param in record_names_visual:
                    if loss_type not in grads_visual.keys():
                        grads_visual[loss_type] = {}
                    if param.grad is not None:
                        grads_visual[loss_type][tensor_name] = param.grad.data.clone()
                    else:
                        grads_visual[loss_type][tensor_name] = torch.zeros_like(param.data)
                grads_visual[loss_type]["concat"] = torch.cat([grads_visual[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_visual])
                average_grad_visual_mm = torch.mean(grads_visual[loss_type]["concat"]).item()
                writer.add_scalar('average_grad_visual_mm',average_grad_visual_mm,epoch)
            
            optimizer.zero_grad()


        weighted_loss_out_v.backward(retain_graph=True)
        weighted_loss_out_a.backward(retain_graph=True)
        loss.backward()

        # 初始化 beta
        beta = torch.tensor(0.0, device=device)
        beta2 = torch.tensor(0.0, device=device)
        model.usegate = False
        warmup = min(1.0, float(epoch + 1) / max(1, args.cmob_beta_warmup_epochs))
        k_threshold = args.sw_k_threshold

        if args.use_cmob:
            # CMoB: gap-aware modulation with EMA smoothing and capped beta
            gap, te_a, te_v = cmob_tool.calculate_causal_gap(model, spec, image, label, ce_criterion)
            scaled_gap = gap * args.cmob_gap_scale
            if ema_gap is None:
                ema_gap = scaled_gap.detach()
            else:
                ema_gap = args.cmob_ema_momentum * ema_gap + (1 - args.cmob_ema_momentum) * scaled_gap.detach()
            smoothed_gap = ema_gap

            if smoothed_gap > 0 and k > k_threshold:
                # audio dominates: penalize audio (exp-based, same as original), lightly support visual
                beta = warmup * torch.clamp(0.95 * torch.exp(smoothed_gap), max=args.cmob_beta_cap)
                beta2 = warmup * args.cmob_visual_boost * torch.tanh(torch.abs(smoothed_gap))
            elif smoothed_gap < 0 and k > k_threshold:
                # visual dominates: penalize visual proportionally (tanh-based, avoids always-capped issue)
                beta2 = warmup * torch.clamp(args.cmob_beta_cap * torch.tanh(torch.abs(smoothed_gap)), max=args.cmob_beta_cap)

        else:
            # Original score-based branch, aligned with the same threshold/caps
            score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
            score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])

            if (score_a > score_v) and k > k_threshold:
                gap = tanh(score_a - score_v)
                beta = warmup * torch.clamp(0.95 * torch.exp(gap), max=args.cmob_beta_cap)
                beta2 = warmup * args.cmob_visual_boost * torch.tanh(torch.abs(gap))
            elif (score_a < score_v) and k > k_threshold:
                gap = tanh(score_v - score_a)
                beta2 = warmup * torch.clamp(0.1 * torch.exp(gap), max=args.cmob_beta_cap)
        for model_param, global_model_param in zip(model.parameters(),global_model.parameters()):

            if model_param.requires_grad and any(model_param is param for param in model.module.audio_net.parameters()):
                model_param.grad += beta * (model_param - global_model_param)

            if model_param.requires_grad and any(model_param is param for param in model.module.visual_net.parameters()):
                model_param.grad += beta2 * (model_param - global_model_param)

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                # 修改：增加 and module.weight.grad is not None
                if module.weight.requires_grad and module.weight.grad is not None:
                    fim[name] += (module.weight.grad * module.weight.grad)
                    fim[name].detach_()

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 修改点：每个分支都必须加 and module.weight.grad is not None
                if 'audio_net' in name and module.weight.requires_grad and module.weight.grad is not None:
                    fim_audio[name] += (module.weight.grad * module.weight.grad)
                    fim_audio[name].detach_()
                elif 'visual_net' in name and module.weight.requires_grad and module.weight.grad is not None:
                    fim_visual[name] += (module.weight.grad * module.weight.grad)
                    fim_visual[name].detach_()
                elif 'head_audio' in name and module.weight.requires_grad and module.weight.grad is not None:
                    fim_audio_head[name] += (module.weight.grad * module.weight.grad)
                    fim_audio_head[name].detach_()
                elif 'head_video' in name and module.weight.requires_grad and module.weight.grad is not None:
                    fim_visual_head[name] += (module.weight.grad * module.weight.grad)
                    fim_visual_head[name].detach_()
                elif 'head' in name and module.weight.requires_grad and module.weight.grad is not None:
                    fim_mm_head[name] += (module.weight.grad * module.weight.grad)
                    fim_mm_head[name].detach_()

        for name, parms in model.named_parameters():
            if parms.grad is not None: 
                layer = str(name).split('.')[1]

                if 'audio' in layer and len(parms.grad.size()) == 4:

                    audio_L2_norm_square = torch.sum(parms.grad ** 2).item()
                    total_audio_grad_sum += lr * audio_L2_norm_square
                    total_audio_count += 1


                if 'visual' in layer and len(parms.grad.size()) == 4:

                    visual_L2_norm_square = torch.sum(parms.grad ** 2).item()
                    total_visual_grad_sum += lr * visual_L2_norm_square
                    total_visual_count += 1


        optimizer.step()

        _loss += loss.item()
        _loss_a += loss_out_a.item()
        _loss_v += loss_out_v.item()

    if total_audio_count > 0:
        epoch_audio_L2_norm_mean = total_audio_grad_sum / total_audio_count
    else:
        epoch_audio_L2_norm_mean = 0

    if total_visual_count > 0:
        epoch_visual_L2_norm_mean = total_visual_grad_sum / total_visual_count
    else:
        epoch_visual_L2_norm_mean = 0


    fim_trace = 0
    for name in fim:
        fim[name] = fim[name].mean().item()
        fim_trace += fim[name]

    fim_trace_audio = 0
    for name in fim_audio:
        fim_audio[name] = fim_audio[name].mean().item()
        fim_trace_audio += fim_audio[name]

    fim_trace_visual = 0
    for name in fim_visual:
        fim_visual[name] = fim_visual[name].mean().item()
        fim_trace_visual += fim_visual[name]

    fim_trace_visual_head = 0
    for name in fim_visual_head:
        fim_visual_head[name] = fim_visual_head[name].mean().item()
        fim_trace_visual_head += fim_visual_head[name]

    fim_trace_audio_head = 0
    for name in fim_audio_head:
        fim_audio_head[name] = fim_audio_head[name].mean().item()
        fim_trace_audio_head += fim_audio_head[name]

    fim_trace_mm_head = 0
    for name in fim_mm_head:
        fim_mm_head[name] = fim_mm_head[name].mean().item()
        fim_trace_mm_head += fim_mm_head[name]
    
    visual_trace_list.append(fim_trace_visual)
    audio_trace_list.append(fim_trace_audio)
    print(audio_trace_list)
    writer.add_scalar('fim_trace_visual_head',fim_trace_visual_head,epoch)
    writer.add_scalar('fim_trace_audio_head',fim_trace_audio_head,epoch)
    writer.add_scalar('fim_trace_mm_head',fim_trace_mm_head,epoch)

    scheduler.step()


    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader), epoch_audio_L2_norm_mean, epoch_visual_L2_norm_mean,fim_trace,fim_trace_audio,fim_trace_visual, average_grad_visual, average_grad_audio, average_grad_visual_mm, average_grad_audio_mm


def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label) in enumerate(dataloader):

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            outputs = model(spec.unsqueeze(1).float(), image.float())
            a, v, a_output, v_output, out = outputs[:5]

            out_v = v_output
            out_a = a_output

            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0

                #pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def main():
    args = get_arguments()
    print(args)
    loss_list = []
    loss_a_list = []
    loss_v_list = []

    audio_NormList = [0,0,0,0,0,0,0,0,0,0]
    visual_NormList = [0,0,0,0,0,0,0,0,0,0]
    audio_FGNList = []
    visual_FGNList = []
    audio_GNorm_list = []
    visual_GNorm_list = []
    fim_list = []
    audio_fim_list = []
    visual_fim_list = []
    accuracy_list = []
    accuracy_list_audio = []
    accuracy_list_visual = []
    avarage_gradient_visual_mm_list = []
    avarage_gradient_audio_mm_list = []
    avarage_gradient_visual_list = []
    avarage_gradient_audio_list = []
    
    visual_trace_list = []
    audio_trace_list = []



    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    model = AVClassifier(args)

    model.apply(weight_init)
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.cuda()


    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay = 1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    if args.dataset == 'KineticSound':
        train_dataset = KS_dataset(args=None, mode='train', select_ratio=1, v_norm=True, a_norm=False, name="KS")
        test_dataset = KS_dataset(args=None, mode='test', select_ratio=1, v_norm=True, a_norm=False, name="KS")
    elif args.dataset == 'CREMAD':
        train_dataset = CramedDataset(args, mode='train')
        test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == 'AVE':
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    print(f"Number of samples in training dataset: {len(train_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=32, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True)
    # writer_path = os.path.join(args.tensorboard_path, args.dataset)
    # if not os.path.exists(writer_path):
    #     os.mkdir(writer_path)
    # log_name = '{}_{}'.format(args.fusion_method, args.modulation)
    # writer = SummaryWriter(os.path.join(writer_path, log_name))
    # log_images_to_tensorboard(writer, train_dataloader, device, phase='train')

    if args.train:

        best_acc = 0.0
        best_joint_acc = 0.0

        for epoch in range(args.epochs):
            total_audio_grad_sum = 0
            total_visual_grad_sum = 0
            total_audio_count = 0
            total_visual_count = 0

            epoch_data_lists = {
                'conf_v': [], 'acc_v': [],
                'conf_a': [], 'acc_a': [],
            }

            print('Epoch: {}: '.format(epoch))

            if args.use_tensorboard:

                writer_path = os.path.join(args.tensorboard_path, args.dataset)
                if not os.path.exists(writer_path):
                    os.mkdir(writer_path)
                log_name = '{}_{}'.format(args.fusion_method, args.modulation)
                writer = SummaryWriter(os.path.join(writer_path, log_name))
                
                
                batch_loss, batch_loss_a, batch_loss_v, epoch_audio_L2_norm_mean, epoch_visual_L2_norm_mean,fim_trace ,fim_trace_audio,fim_trace_visual,average_grad_visual,average_grad_audio,average_grad_visual_mm,average_grad_audio_mm = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler,writer,visual_trace_list,audio_trace_list,epoch_data_lists)
                fim_list.append(fim_trace)
                loss_list.append(batch_loss)
                loss_a_list.append(batch_loss_a)
                loss_v_list.append(batch_loss_v)
                avarage_gradient_visual_list.append(average_grad_visual)
                avarage_gradient_audio_list.append(average_grad_audio)
                avarage_gradient_visual_mm_list.append(average_grad_visual_mm)
                avarage_gradient_audio_mm_list.append(average_grad_audio_mm)
                audio_fim_list.append(fim_trace_audio)
                visual_fim_list.append(fim_trace_visual)
                # visual_trace_list.append(fim_trace_visual)
                # audio_trace_list.append(fim_trace_audio)
                audio_GNorm_list.append(epoch_audio_L2_norm_mean)
                visual_GNorm_list.append(epoch_visual_L2_norm_mean)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
                accuracy_list.append(acc)
                accuracy_list_visual.append(acc_v)
                accuracy_list_audio.append(acc_a)
                audio_NormList.append(epoch_audio_L2_norm_mean)
                audio_OldNorm = max([np.mean(audio_NormList[-11:-1]), 0.0000001])
                audio_NewNorm = np.mean(audio_NormList[-11:])
                audio_FGNList.append((audio_NewNorm - audio_OldNorm) / audio_NewNorm)
                print("audio_FGN:", (audio_NewNorm - audio_OldNorm) / audio_OldNorm)
                visual_NormList.append(epoch_visual_L2_norm_mean)
                visual_OldNorm = max([np.mean(visual_NormList[-11:-1]), 0.0000001])
                visual_NewNorm = np.mean(visual_NormList[-11:])
                visual_FGNList.append((visual_NewNorm - visual_OldNorm) / visual_NewNorm)
                print("visual_FGN:", (visual_NewNorm - visual_OldNorm) / visual_OldNorm)

                writer.add_scalars('Loss', {'Total Loss': batch_loss,
                                            'Audio Loss': batch_loss_a,
                                            'Visual Loss': batch_loss_v}, epoch)

                writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                                  'Audio Accuracy': acc_a,
                                                  'Visual Accuracy': acc_v}, epoch)

                writer.add_scalar('FIM_Trace', fim_trace,epoch)
                writer.add_scalar('FIM_Trace_audio', fim_trace_audio,epoch)
                writer.add_scalar('Fim_Trace_visual',fim_trace_visual,epoch)
                # ==================== 最终版本的绘图代码 ===============================
                print(f"Epoch {epoch} 完成, 开始绘制 Batch 级置信度-准确率散点图...")
                
                plt.figure(figsize=(12, 12))
                
                # 绘制 y=x 对角线作为参考
                plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x (Reference Line)')
                
                # 从 epoch_data_lists 字典中取数据绘图
                plt.scatter(epoch_data_lists['conf_v'], epoch_data_lists['acc_v'], color='blue', alpha=0.6, s=30, label='Visual Modality (Batch Points)')
                plt.scatter(epoch_data_lists['conf_a'], epoch_data_lists['acc_a'], color='red', alpha=0.6, s=30, label='Audio Modality (Batch Points)')

                plt.title(f'Batch-Level Confidence vs. Accuracy - Epoch {epoch}', fontsize=16)
                plt.xlabel('Average Confidence per Batch', fontsize=12)
                plt.ylabel('Accuracy per Batch', fontsize=12)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.gca().set_aspect('equal', adjustable='box')

                plot_save_path = os.path.join('results', 'batch_scatter_plots')
                os.makedirs(plot_save_path, exist_ok=True)
                
                plt.savefig(os.path.join(plot_save_path, f'batch_scatter_epoch_{epoch}.png'))
                print(f"Batch级散点图已保存到: {os.path.join(plot_save_path, f'batch_scatter_epoch_{epoch}.png')}")
                plt.close()
                # ==================== 绘图代码结束 ========================================

                                            

            else:
                batch_loss, batch_loss_a, batch_loss_v, epoch_audio_L2_norm_mean, epoch_visual_L2_norm_mean,fim_trace,fim_trace_audio,fim_trace_visual,average_grad_visual,average_grad_audio,average_grad_visual_mm,average_grad_audio_mm = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler,epoch_data_lists=epoch_data_lists)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
                audio_GNorm_list.append(epoch_audio_L2_norm_mean)
                visual_GNorm_list.append(epoch_visual_L2_norm_mean)
                audio_NormList.append(epoch_audio_L2_norm_mean)
                audio_OldNorm = max([np.mean(audio_NormList[-2:-1]), 0.0000001])
                audio_NewNorm = np.mean(audio_NormList[-2:])
                audio_FGNList.append((audio_NewNorm - audio_OldNorm) / audio_NewNorm)
                print("audio_FGN:", (audio_NewNorm - audio_OldNorm) / audio_OldNorm)
                visual_NormList.append(epoch_visual_L2_norm_mean)
                visual_OldNorm = max([np.mean(visual_NormList[-2:-1]), 0.0000001])
                visual_NewNorm = np.mean(visual_NormList[-2:])
                visual_FGNList.append((visual_NewNorm - visual_OldNorm) / visual_NewNorm)
                print("visual_FGN:", (visual_NewNorm - visual_OldNorm) / visual_OldNorm)

            joint_target_met = acc >= args.best_target_acc and acc_a >= args.best_target_audio_acc and acc_v >= args.best_target_visual_acc

            print('Epoch:  {}   Loss: {:.3f}  Acc: {:.3f}  Audio Acc: {:.3f}  Visual Acc: {:.3f}  Best Acc: {:.3f}  Best Joint Acc: {:.3f}'.format(
                epoch, batch_loss, acc, acc_a, acc_v, max(best_acc, acc), best_joint_acc
            ))

            if acc > best_acc:
                best_acc = float(acc)

                save_dir = save_training_checkpoint(args, model, optimizer, scheduler, epoch, acc, acc_a, acc_v, tag='best_model')
                print('The best model has been saved at {}.'.format(save_dir))
                print("Loss: {:.3f}, Acc: {:.3f}".format(batch_loss, acc))
                print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))
            else:
                print("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))
                print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))

            if joint_target_met and acc > best_joint_acc:
                best_joint_acc = float(acc)
                save_dir = save_training_checkpoint(args, model, optimizer, scheduler, epoch, acc, acc_a, acc_v, tag='best_joint_target_model')
                print('The best joint-target model has been saved at {}.'.format(save_dir))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = f"experiments_{timestamp}_{args.modulation}"
        results_path = os.path.join(os.getcwd(), "results", experiment_folder)
        os.makedirs(results_path, exist_ok=True)

        data_to_save = {
            "fim_list": fim_list,
            "audio_fim_list": audio_fim_list,
            "visual_fim_list": visual_fim_list,
            "accuracy_list": accuracy_list,
            "accuracy_list_audio": accuracy_list_audio,
            "accuracy_list_visual": accuracy_list_visual,
            "avarage_gradient_visual_mm_list": avarage_gradient_visual_mm_list,
            "avarage_gradient_audio_mm_list": avarage_gradient_audio_mm_list,
            "avarage_gradient_visual_list": avarage_gradient_visual_list,
            "avarage_gradient_audio_list": avarage_gradient_audio_list,
            "loss_list":loss_list,
            "loss_a_list":loss_a_list,
            "loss_v_list":loss_v_list
            
        }

        for name, data in data_to_save.items():
            pkl_path = os.path.join(results_path, f"{name}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(data, f)
            csv_path = os.path.join(results_path, f"{name}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                if isinstance(data[0], (list, tuple)):
                    writer.writerows(data)
                else:
                    writer.writerow(data)


    else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path)
        # epoch = loaded_dict['saved_epoch']
        modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']

        assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

        model = model.load_state_dict(state_dict)
        print('Trained model loaded!')

        acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


if __name__ == "__main__":
    main()
