import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
# from google.colab import drive  # include this on Google Colab
import torchvision.models as models
import sys
sys.path.insert(0, 'Ranger-Deep-Learning-Optimizer/ranger')  # chage the path to your ranger optimizer's path
from ranger import Ranger


batch_size = 64

train_loader = torch.utils.data.DataLoader(datasets.ImageFolder('data/imagenette2/train',
                                                                transform=transforms.Compose([
                                                                    transforms.RandomResizedCrop(256),
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                         std=[0.229, 0.224, 0.225])
                                                                    
                                                                ])),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.ImageFolder('data/imagenette2/val',
                                                               transform=transforms.Compose([
                                                                   transforms.ToTensor(),
                                                                   transforms.Resize([500, 500]),
                                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                        std=[0.229, 0.224, 0.225])
                                                               ])),
                                          batch_size=batch_size, shuffle=True)
# change the imagefolder path to your datasets' path
classes = ('tench', 'springer', 'casette_player', 'chain_saw',
           'church', 'French_horn', 'garbage_truck', 'gas_pump', 'golf_ball', 'parachute')


model = models.resnet18(pretrained=False)
fc_features = model.fc.in_features
model.fc = nn.Linear(fc_features, 10)
model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = Ranger(model.parameters(), lr=8e-03, eps=1e-06)
# for resuming checkpoints:
# checkpoint = torch.load('drive/MyDrive/Imagenette Classification/ckpt_211.pth', map_location='cuda:0')
# model.load_state_dict(checkpoint['net'])
# optimizer.load_state_dict(checkpoint['optimizer'])


for epoch in range(600):
    model.train()
    for image, label in train_loader:
        image, label = image.cuda(), label.cuda()
        y = model(image)
        loss = criterion(y, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    correct = 0
    total = 0
    for image, label in test_loader:
        image, label = image.cuda(), label.cuda()
        y = model(image)
        pred = torch.argmax(y, dim=1)
        correct += torch.sum((label == pred).long()).cpu().numpy()
        total += image.size(0)
    print('epoch %d: %4f' % (epoch, correct / total))
    checkpoint = {
        "net": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, 'drive/MyDrive/Imagenette Classification/Checkpoints/ckpt_%s.pth' % (str(epoch)))
    # change the path to your checkpoints' saving path
