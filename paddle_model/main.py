# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 17:28:11 2021

@author: wjg
"""


from torchvision import transforms
import torchvision
import torch
import numpy as np
#预处理
transform = transforms.Compose([transforms.Resize([224, 224]),transforms.ToTensor()])# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#读取数据集
trainset = torchvision.datasets.ImageFolder(root='D:/MLP/val', transform=transform)
#打包成DataLoader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
 
#同上
testset = torchvision.datasets.ImageFolder(root='D:/MLP/val', transform=transform)
testloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=0)


from model.mlp_mixer import mixer_b16_224
model = mixer_b16_224(pretrained=True,num_classes=1000)

def validate(val_loader, model, criterion):
    model.eval()
    acc1 = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input
            target = target
            
            output = model(input)
            loss = criterion(output, target)
            acc = (output.argmax(1) == target).float().mean().item()
            acc1.append(acc)
            print('loss:',loss, 'Acc:',acc)

        print(' * Val Acc@1 {0}'.format(np.mean(acc1)))
        return np.mean(acc1)
    
criterion = torch.nn.CrossEntropyLoss()
val_acc = validate(testloader, model, criterion)