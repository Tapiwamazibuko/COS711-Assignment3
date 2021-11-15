#!/usr/bin/env python
# coding: utf-8

# In[207]:


from torchvision import transforms, utils
from torch.utils.data import random_split
from fastai.vision.all import *
import pandas as pd
import gc
import cv2
import albumentations
from typing import Any
from PIL import Image
import numpy as np
import colorsys
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore')


# In[2]:


class AlbumentationsTransform(RandTransform):
    split_idx,order=None,2
    def __init__(self, train_aug, valid_aug): store_attr()
    
    def before_call(self, b, split_idx):
        self.idx = split_idx
    
    def encodes(self, img: PILImage):
        if self.idx == 0:
            aug_img = self.train_aug(image=np.array(img))['image']
        else:
            aug_img = self.valid_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)


# In[192]:


def get_train_aug(): return albumentations.Compose([
            albumentations.HueSaturationValue(
                hue_shift_limit=30, 
                sat_shift_limit=20, 
                val_shift_limit=10, 
                p=1., always_apply=False
            ),
            albumentations.CenterCrop (
                height=205, width=205, always_apply=False, p=1.0)

])

def get_val_aug(): return albumentations.Compose([
            albumentations.HueSaturationValue(
                hue_shift_limit=30, 
                sat_shift_limit=20, 
                val_shift_limit=10, 
                p=1., always_apply=False
            ),
            albumentations.CenterCrop (
                height=205, width=205, always_apply=False, p=1.0) 
])


# In[193]:


item_tfms = [Resize(224), AlbumentationsTransform(get_train_aug(), get_val_aug())]


# In[5]:


train = pd.read_csv('Train.csv')
print(train.shape)
train.head(2)


# In[6]:


number_of_rows = len(train.index) + 1
number_of_rows_file1 = int(number_of_rows * 0.2)
skip_rows = 1
rows = number_of_rows - number_of_rows_file1


# In[164]:


df_val = pd.read_csv('Train.csv', header=None, nrows = number_of_rows_file1,skiprows = skip_rows)


# In[165]:


col_names = ['Image_ID', 'class', 'xmin', 'ymin', 'width', 'height']


# In[9]:


df_train = pd.read_csv('Train.csv', header=None, nrows = rows,skiprows = number_of_rows_file1)


# In[166]:


df_val = df_val.set_axis(col_names, axis=1)
df_train = df_train.set_axis(col_names, axis=1)


# In[194]:


# Create the dataloaders
dls = ImageDataLoaders.from_df(df_train, path="Train_Images/Train_Images", 
                               fn_col='Image_ID', 
                               label_col='class', 
                               suff='.jpg', item_tfms=item_tfms, num_workers=0, valid_pct=0.2, bs=64, 
                               splitter=RandomSplitter(valid_pct=0.2, seed=42), shuffle=True)
dls.show_batch()


# In[136]:


dls_test = ImageDataLoaders.from_df(df_val, path="Train_Images\Test", 
                               fn_col='Image_ID', 
                               label_col='class', 
                               suff='.jpg', item_tfms=Resize(224), num_workers=0)
dls_test.show_batch()


# In[93]:


class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )


# In[120]:


def nn_model(version: str,  progress: bool, pretrained: bool, **kwargs: Any):
    pytorch_net = nn.Sequential(
    nn.Conv2d(3, 96, kernel_size=7, stride=2),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
    Fire(96, 16, 64, 64),
    Fire(128, 16, 64, 64),
    Fire(128, 32, 128, 128),
    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
    Fire(256, 32, 128, 128),
    Fire(256, 48, 192, 192),
    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
    Fire(384, 48, 192, 192),
    nn.Dropout(p=0.5),
    nn.Conv2d(384, 3, kernel_size=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.LogSoftmax(dim=1))
    
    return pytorch_net


# In[90]:


def model(version: str = "1_0", pretrained: bool = False, progress: bool = True, **kwargs: Any):
    #
    return nn_model("1_0", pretrained, progress, **kwargs)


# In[174]:


learn = cnn_learner(dls, model, metrics=accuracy, normalize=True, pretrained=False)


# In[210]:


learn = cnn_learner(dls, squeezenet1_0, metrics=accuracy, normalize=True, loss_func=CrossEntropyLossFlat(), opt_func=Adam)


# In[ ]:


learn.lr_find()


# In[27]:


learn.fine_tune(3, base_lr=3e-3)


# In[200]:


learn.fit(15, cbs=EarlyStoppingCallback(monitor='valid_loss', patience=2))


# In[209]:


learn.show_results()
plt.show()


# In[201]:


preds = learn.get_preds(dl=dls_test.test_dl(df_val)) # Getting the predicted probabilities


# In[208]:


f1_score(y_true, y_pred, average='macro')


# In[84]:


learn.dls.vocab 


# In[204]:


y_pred = [learn.dls.vocab[p[2]] for p in np.argsort(preds[0])]


# In[195]:


y_true = df_val['class']


# In[202]:


# Need to make sure order is right - note staring with the highest prob
df_val['prediction'] = [learn.dls.vocab[p[2]] for p in np.argsort(preds[0])]
df_val.head()


# In[203]:


correct_pred = 0
for index, row in df_val.iterrows():
    if(row["class"] == row["prediction"]):
        correct_pred += 1

print(correct_pred/781)

