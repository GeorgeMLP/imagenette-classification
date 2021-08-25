import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models


batch_size = 1

train_loader = torch.utils.data.DataLoader(datasets.ImageFolder('imagenette2/train',
                                                                transform=transforms.Compose([
                                                                    transforms.RandomResizedCrop(256),
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                         std=[0.229, 0.224, 0.225])
                                                                ])),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.ImageFolder('imagenette2/val',
                                                               transform=transforms.Compose([
                                                                   transforms.ToTensor(),
                                                                   transforms.Resize([500, 500]),
                                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                        std=[0.229, 0.224, 0.225])
                                                               ])),
                                          batch_size=batch_size, shuffle=True)
# change the path to your datasets' path
classes = ('tench', 'springer', 'casette_player', 'chain_saw',
           'church', 'French_horn', 'garbage_truck', 'gas_pump', 'golf_ball', 'parachute')


net = models.resnet18(pretrained=False)
fc_features = net.fc.in_features
net.fc = nn.Linear(fc_features, 10)
path_checkpoint = "ResNet18 0.945223.pth"
net = net.cuda()
checkpoint = torch.load(path_checkpoint, map_location='cuda:0')
net.load_state_dict(checkpoint['net'])
net.eval()
correct = 0
total = 0
for image, label in test_loader:
    image, label = image.cuda(), label.cuda()
    y = net(image)
    pred = torch.argmax(y, dim=1)
    correct += torch.sum((label == pred).long()).cpu().numpy()
    total += image.size(0)
print('%4f' % (correct / total))
