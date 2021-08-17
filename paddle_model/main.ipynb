import paddle
import paddle.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
import paddle.vision.transforms as T


BATCH_SIZE = 32
transform_train = T.Compose([T.Resize(size=224), 
                             T.RandomHorizontalFlip(0.5),
                             T.ToTensor(),
                             T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_test = T.Compose([T.Resize(size=224), 
                            T.ToTensor(),
                            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar10_train = paddle.vision.datasets.Cifar10(mode='train', transform=transform_train)
cifar10_test = paddle.vision.datasets.Cifar10(mode='test', transform=transform_test)
train_loader = paddle.io.DataLoader(cifar10_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = paddle.io.DataLoader(cifar10_test, batch_size=BATCH_SIZE, shuffle=False)

from model.mlp_mixer import mixer_b16_224,mixer_b16_224_in21k
model = mixer_b16_224(pretrained=True,num_classes=10)
model_in21k = mixer_b16_224_in21k(pretrained=True,num_classes=10)

model = paddle.Model(model)
model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=1e-5,parameters=model.parameters()),
              loss=paddle.nn.CrossEntropyLoss(),
              metrics=paddle.metric.Accuracy(topk=(1, 5)))
model.fit(train_data=train_loader,
          eval_data=test_loader,
          epochs=2,
          eval_freq=1,
          save_dir='/home/aistudio/mixer_b16_224_checkpoints',
          save_freq=1,
          verbose=1)

model = paddle.Model(model_in21k)
model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=1e-5,parameters=model.parameters()),
              loss=paddle.nn.CrossEntropyLoss(),
              metrics=paddle.metric.Accuracy(topk=(1, 5)))
model.fit(train_data=train_loader,
          eval_data=test_loader,
          epochs=2,
          eval_freq=1,
          save_dir='/home/aistudio/mixer_b16_224_in21k_checkpoints',
          save_freq=1,
          verbose=1)
