import argparse
import pdb
import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
import time
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import requests
import model_reverse as C3D_model
from old_16_frames import VideoDataset

def send_dipesh(send_string):
#    chatwork.send_message(log_id, send_string)
    headers = {
        'Content-type': 'application/json',
    }
    data = '{}{}{}'.format("{\"text\":\"",send_string,"\"}")
    response = requests.post('https://hooks.slack.com/services/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', headers=headers, data=data)

send_dipesh("test timestamp: {}".format(datetime.now()))
send_dipesh("python file name: "+os.path.abspath(__file__))
send_dipesh("--- UCF code started ---")
# Use GPU if available else revert to CPU

parser = argparse.ArgumentParser(description='Video action recogniton training')
parser.add_argument('--logfile_name', type=str, default="fwd_rvs_test",
                    help='file name for storing the log file')
parser.add_argument('--gpu', type=int, default=1,
                    help='GPU ID, start from 0')
args = parser.parse_args()

gpu_id = str(args.gpu)
log_name = args.logfile_name
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device, "| gpu_id: ", gpu_id)
std_start_time = time.time()

nEpochs = 100  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True # See evolution of the test set when training
nTestInterval = 20 # Run on test set every nTestInterval epochs
snapshot = 50 # Store a model every snapshot epochs
lr = 1e-3 # Learning rate

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

# if resume_epoch != 0:
#     runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
#     run_id = int(runs[-1].split('_')[-1]) if runs else 0
# else:
#     runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
#     run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

# save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
save_dir = os.path.join(save_dir_root, 'run',log_name)
modelName = 'C3D' # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset

def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
        # train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
        #                 {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]

    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.SGD(list(model.parameters()), lr=lr, momentum=0.9, weight_decay=5e-4)
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
    criterion.to(device)

    log_dir = os.path.join(save_dir, 'models',log_name ,datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train',clip_len=16), batch_size=20, shuffle=True, num_workers=4)
    val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val',  clip_len=16), batch_size=20, num_workers=4)
    test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16), batch_size=20, num_workers=4)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                model.train()
            else:
                model.eval()

            for inputs, labels in (trainval_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    inputs_rev = [inputs[:,:,15-i,:,:] for i in range(16)]
                    inputs_rev = torch.stack(inputs_rev)
                    inputs_rev = inputs_rev.permute(1,2,0,3,4)
                    outputs = model(inputs, inputs_rev)
                else:
                    with torch.no_grad():
                        # outputs = model(inputs,inputs)
                        inputs_rev = [inputs[:,:,15-i,:,:] for i in range(16)]
                        inputs_rev = torch.stack(inputs_rev)
                        inputs_rev = inputs_rev.permute(1,2,0,3,4)
                        outputs = model(inputs, inputs_rev)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

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

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in (test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs, inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

    writer.close()


if __name__ == "__main__":
    train_model()
    print("total_time_taken:",int(-(std_start_time - time.time())/3600)," hrs  ", int(-(std_start_time - time.time())/60%60), " mins")
    send_dipesh("--- UCF code ENDED ---")
    
