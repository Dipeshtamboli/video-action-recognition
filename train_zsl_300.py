import requests
import pdb
import numpy as np
import timeit
from datetime import datetime
import socket
import os
import time
import glob
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from sklearn.preprocessing import normalize
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data_preprocessing_16_frames import VideoDataset
import model_300_zsl
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# torch.backends.cudnn.benchmark = False
# special_info_for_run = "  300_feats_gzsl"

def send_message(send_string):
    headers = {
        'Content-type': 'application/json',
    }
    data = '{}{}{}'.format("{\"text\":\"",send_string,"\"}")
    response = requests.post('https://hooks.slack.com/services/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', headers=headers, data=data)

send_message("code started")
std_start_time = time.time()
# Use GPU if available else revert to CPU
gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# device = torch.device("cuda:0")
print("Device being used:", device, "| gpu_id: ", gpu_id)

nEpochs = 100  #100 Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True # See evolution of the test set when training
nTestInterval = 1 #20 Run on test set every nTestInterval epochs
snapshot = 50 # Store a model every snapshot epochs
# lr = 1e-5 # Learning rate
lr = 1e-5 # Learning rate
mse_300_lambada = 10.0
def to_np(z):
       return z.data.cpu().numpy()



dataset = 'ucf101' # Options: hmdb51 or ucf101


if dataset == 'hmdb51':
    num_classes=51
elif dataset == 'ucf101':
    num_classes = 101
else:
    print('We only implemented hmdb and ucf datasets.')
    raise NotImplementedError

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

print("save_dir_root:",save_dir_root)

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-3]) + 1 if runs else 0

# print("runs:",runs)
# print("run_id:",run_id)
save_dir = os.path.join(save_dir_root, 'run',datetime.now().strftime('%b%d_%H-%M') +'__'+ special_info_for_run)
modelName = 'C3D' # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset

print("save_dir:",save_dir)

# exit()



def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    if modelName == 'C3D':
        model = model_300_zsl.C3D(num_classes=num_classes, pretrained=False)
        train_params = [{'params': model_300_zsl.get_1x_lr_params(model), 'lr': lr},
                        {'params': model_300_zsl.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = model.parameters()
    else:
        print('We only implemented C3D and R2Plus1D models.')
        raise NotImplementedError
    criterion_CEL = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    # criterion_MSE = nn.MSELoss()  # standard crossentropy loss for classification
    # optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-2)
    optimizer = optim.Adam(train_params, lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion_CEL.to(device)
    # criterion_MSE.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    # 16_frames_split
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train',clip_len=16), batch_size=10, shuffle=True, num_workers=4)
    test_seen_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test_seen', clip_len=16), batch_size=20, num_workers=4)
    test_unseen_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test_unseen', clip_len=16), batch_size=20, num_workers=4)

    # val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val',  clip_len=16), batch_size=20, num_workers=4)
    # val_dataloader = train_dataloader
    # test_dataloader = train_dataloader

    # attributes = np.load("../npy_attributes/attributes_101_300.npy") #shape -> (300,101)
    # seen_attributes = np.load("../npy_attributes/seen_attributes_300_51.npy") #shape -> (300,101)
    # unseen_attributes = np.load("../npy_attributes/unseen_attributes_300_50.npy") #shape -> (300,101)
    # print("attributes.shape",attributes.shape)    
    # exit()
    trainval_loaders = {'train': train_dataloader, 'test_seen': test_seen_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'test_seen']}
    test_seen_size = len(test_seen_dataloader.dataset)

    # seen_attributes = np.transpose(seen_attributes)
    # seen_attributes = normalize(seen_attributes, axis=1, norm='l2')
    # seen_attributes = torch.from_numpy(seen_attributes).float().to(device)
    # seen_attributes = Variable(seen_attributes, requires_grad=True)

    # unseen_attributes = np.transpose(unseen_attributes)
    # unseen_attributes = normalize(unseen_attributes, axis=1, norm='l2')
    # unseen_attributes = torch.from_numpy(unseen_attributes).float().to(device)
    # unseen_attributes = Variable(unseen_attributes, requires_grad=True)

    # print(unseen_attributes.shape)
    # print(seen_attributes.shape)
    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        print("training ...")
        for phase in ['train', 'test_seen']:
            pred_labels_train_all = []
            gt_labels_train_all = []
        # for phase in [ 'test_seen']:
            start_time = timeit.default_timer()
            # break
            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0
            running_zsl_corrects = 0.0
            model.volatile = True

            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                model.train()
                # scheduler.step()
            else:
                model.eval()
            first_time = 1
            for inputs, labels in (trainval_loaders[phase]): #tqdm
                # move inputs and labels to the device the training is taking place on
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                seen_att_labels = seen_attributes[labels]
                ground_truth_att_all_300 = seen_attributes
                optimizer.zero_grad()

                if phase == 'train':
                    # logits_51,vid_mse_300 = model(inputs)
                    logits_51= model(inputs)
                else:
                    with torch.no_grad():
                        logits_51,vid_mse_300 = model(inputs)

                # att_100_mse = att_100_mse_all[labels]
                # att_300_mse = att_300_mse_all[labels]
                if first_time==1:
                    # print("don't worry, code is working, don't think to add tqdm here.")
                    first_time=0
                    projected_att_300dim=vid_mse_300
                    ground_truth_labels = labels


                projected_att_300dim = torch.cat((projected_att_300dim, vid_mse_300),dim=0)
                ground_truth_labels = torch.cat((ground_truth_labels, labels),dim=0)

                loss_CEL_51 = criterion_CEL(logits_51, labels)
                probs = nn.Softmax(dim=1)(logits_51)
                preds = torch.max(probs, 1)[1]
                MSE_loss = criterion_MSE(vid_mse_300, seen_att_labels)
                loss = loss_CEL_51 + mse_300_lambada * MSE_loss

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                cel_loss_running = loss_CEL_51.item() * inputs.size(0)
                mse_loss_running = MSE_loss.item() * inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
           
            # pred_labels_train_all = np.hstack((pred_labels_train_all))
            # gt_labels_train_all = np.hstack((gt_labels_train_all))
            
            # zsl_acc_train = get_per_class_accuracy_skl(pred_labels_train_all, gt_labels_train_all)

            epoch_loss = running_loss / trainval_sizes[phase]
            cel_loss = cel_loss_running / trainval_sizes[phase]
            mse_loss = mse_loss_running / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            gzsl_nearest_neighbour_acc = get_nearest_neighbour_acc(projected_att_300dim, ground_truth_att_all_300, ground_truth_labels)

            if phase == 'train':
                writer.add_scalar('data/train_gzsl_acc', gzsl_nearest_neighbour_acc, epoch)
                writer.add_scalar('data/train_total_loss', epoch_loss, epoch)
                writer.add_scalar('data/train_loss_CEL', cel_loss, epoch)
                writer.add_scalar('data/train_loss_MSE_300', mse_loss, epoch)
                writer.add_scalar('data/train_CE_classificatn_acc', epoch_acc, epoch)

            else:
                writer.add_scalar('data/test_seen_gzsl_acc', gzsl_nearest_neighbour_acc, epoch)
                writer.add_scalar('data/test_seen_total_loss', epoch_loss, epoch)
                writer.add_scalar('data/test_seen_loss_CEL', cel_loss, epoch)
                writer.add_scalar('data/test_seen_loss_MSE_300', mse_loss, epoch)
                writer.add_scalar('data/test_seen_classificatn_acc', epoch_acc, epoch)

            print_satement = "[{}] Epoch: {}/{} Loss: {} epoch_Acc: {} gzsl_Acc: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc, gzsl_nearest_neighbour_acc)
            msg_satement = "[{}] Epoch: {}/{} epoch_Acc: {} gzsl_Acc: {}".format(phase, epoch+1, nEpochs, str(epoch_acc.item())[:4], str(gzsl_nearest_neighbour_acc)[:5])
            print(print_satement)
            send_message(msg_satement)
            stop_time = timeit.default_timer()
            # print("Execution time: " + str(stop_time - start_time))
            print("time taken till now(hr:min)->", int((time.time() - std_start_time)/3600),':',int((time.time() - std_start_time)/60%60))


        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()
            print("testing ...")

            running_loss = 0.0
            running_corrects = 0.0
            running_zsl_corrects = 0.0
            pred_labels_all = []
            gt_labels_all = []
            first_time =1
            for inputs, labels in (test_unseen_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                unseen_att_labels = unseen_attributes[labels]
                ground_truth_att_all_300 = unseen_attributes                
                with torch.no_grad():
                    logits_51,vid_mse_300 = model(inputs)

                # att_100_mse = att_100_mse_all[labels]
                # att_300_mse = att_300_mse_all[labels]
                if first_time==1:
                    first_time=0
                    projected_att_300dim=vid_mse_300
                    ground_truth_labels = labels

                projected_att_300dim = torch.cat((projected_att_300dim, vid_mse_300),dim=0)
                ground_truth_labels = torch.cat((ground_truth_labels, labels),dim=0)

                loss_CEL_51 = criterion_CEL(logits_51, labels)
                probs = nn.Softmax(dim=1)(logits_51)
                preds = torch.max(probs, 1)[1]
                MSE_loss = criterion_MSE(vid_mse_300, unseen_att_labels)

                loss = loss_CEL_51 + mse_300_lambada * MSE_loss

                #latent_unseen_attribute < pass unseen att throu nw
                #running_zsl_corrects += zsl_acc
                    
                cel_loss_running = loss_CEL_51.item() * inputs.size(0)
                mse_loss_running = MSE_loss.item() * inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / trainval_sizes[phase]
            cel_loss = cel_loss_running / trainval_sizes[phase]
            mse_loss = mse_loss_running / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            gzsl_nearest_neighbour_acc = get_nearest_neighbour_acc(projected_att_300dim, ground_truth_att_all_300, ground_truth_labels)
            writer.add_scalar('data/test_unseen_gzsl_acc', gzsl_nearest_neighbour_acc, epoch)
            writer.add_scalar('data/test_unseen_total_loss', epoch_loss, epoch)
            # writer.add_scalar('data/test_unseen_loss_CEL', cel_loss, epoch)
            writer.add_scalar('data/test_unseen_loss_MSE_300', mse_loss, epoch)
            # writer.add_scalar('data/test_unseen_CE_classificatn_acc', epoch_acc, epoch)
            
            print_msg = "[test_unseen] Epoch: {}/{} gzsl_Acc: {}".format(epoch+1, nEpochs, str(gzsl_nearest_neighbour_acc)[:5])
            send_message(print_msg)
            print(print_msg)
            stop_time = timeit.default_timer()
            # print("Execution time: " + str(stop_time - start_time))
            print("time taken till now(hr:min)->", int((time.time() - std_start_time)/3600),':',int((time.time() - std_start_time)/60%60))
            print("---------------------------------------------------------------------")

    writer.close()

def get_nearest_neighbour_acc(projected_att_300dim, ground_truth_att_all_300, ground_truth_labels):
    # pred_labels = []
    pred_labels = torch.LongTensor(projected_att_300dim.shape[0], ).cuda()
    for i in range(projected_att_300dim.shape[0]):
                proj_att_rep = projected_att_300dim[i, :].repeat(ground_truth_att_all_300.shape[0], 1)
                #dist = torch.sum((pred_att_rep_test_seen - latent_att_test_seen)**2, 1)

                dist = get_distance(proj_att_rep, ground_truth_att_all_300)
                pred_labels[i] = torch.argmin(dist)
                # pred_labels.append(torch.argmin(dist))
                # pdb.set_trace()
                acc = get_per_class_accuracy_skl(to_np(pred_labels), to_np(ground_truth_labels))
    return acc

# def get_test_accuracy_gzsl(latent_att_unseen,latent_vis,num_classes,label):
#         st = time.time()
#         pred_labels = torch.LongTensor(latent_vis.size(0), ).cuda()
#         for i in range(latent_vis.shape[0]):
#                 latent_rep = latent_vis[i, :].repeat(num_classes, 1)
#                 #dist = torch.sum((pred_att_rep - latent_att_unseen)**2, 1)
#                 dist = get_distance(latent_rep, latent_att_unseen)
#                 pred_labels[i] = torch.argmin(dist)
#         return to_np(pred_labels.cpu())

def get_distance(mat1, mat2,P_NORM=2):
    # print("mat1.shape,mat2.shape",mat1.shape,mat2.shape)
    # pdb.set_trace()
    return torch.sum(torch.abs(mat1 - mat2).pow(P_NORM), 1).pow(1.0/P_NORM)
    
def get_per_class_accuracy_skl(predicted_labels, gt_labels):
        acc = 0.
        predicted_labels = (predicted_labels)
        gt_labels = (gt_labels)
        unique_labels = np.unique(gt_labels)
        for l in unique_labels:
                idx = np.nonzero(gt_labels == l)[0]
                # pdb.set_trace()
                acc += np.sum(gt_labels[idx] == predicted_labels[idx])/len(predicted_labels)
                # acc += accuracy_score(gt_labels[idx], predicted_labels[idx])
        acc = acc / unique_labels.shape[0]
        return acc


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label

if __name__ == "__main__":
    train_model()   
    print("total_time_taken:",int(-(std_start_time - time.time())/3600)," hrs  ", int(-(std_start_time - time.time())/60%60), " mins")
