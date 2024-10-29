# utils file
import os
import glob
from PIL import Image
import torchvision.transforms as trans
from tqdm import tqdm
import torch
import numpy as np
from keras.utils import to_categorical
import tensorflow as tf



os.chdir("/content/drive/MyDrive/GEOMODEL/IMAGES/DATASET")


################################################################################
# Transform Image to Tensor:                                                    #
################################################################################
def image_to_tensor_resize(name, pixels):

  # Read a PIL image
  image = Image.open(name)

  # image to a Torch tensor
  transform = trans.transforms.Compose([
    trans.Resize((pixels,pixels)),
    trans.ToTensor(),
    trans.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])
    
  # Convert the PIL image to Torch tensor
  img_tensor = transform(image)
  
  return img_tensor

################################################################################
# Resize Image:                                                                #
################################################################################
def resize_image(name, pixels):

  img_tensor = image_to_tensor_resize(name, pixels)
  
  # define a transform to convert a tensor to PIL image
  transform = trans.ToPILImage()

  # convert the tensor to PIL image using above transform
  img = transform(img_tensor)

  img = img.convert("RGB")

  os.chdir("/content/drive/MyDrive/GEOMODEL/resize_GAME")

  img.save(name)

  return 



################################################################################
# Resize Images:                                                               #
################################################################################

def resize_images(pixels):

  files = sorted(glob.glob(f"/content/drive/MyDrive/GEOMODEL/GAME/*jpg"))

  for file in tqdm(files):
    print(file)
    img = os.path.split(file)[1]
    
    resize_image(img, pixels)
    os.chdir("/content/drive/MyDrive/GEOMODEL/GAME")
  
  return 


################################################################################
# Transform Image to Tensor:                                                    #
################################################################################
def image_to_tensor(name):

  # Read a PIL image
  image = Image.open(name)

  # image to a Torch tensor
  transform = trans.transforms.Compose([
    trans.ToTensor()
])
    
  # Convert the PIL image to Torch tensor
  img_tensor = transform(image)
  
  return img_tensor


################################################################################
# Load Data:                                                                   #
################################################################################


def load_data(data, labels, join, pixels):

  files = sorted(glob.glob(f"/content/drive/MyDrive/GEOMODEL/resizePERS/*jpg"))

  mapping = {"Albania": 0, "Austria": 1, "Belgium": 2, "Bosnia Herzegovina": 3, "Bulgaria": 4, "Croatia": 5, "Czech Republic": 6, "Denmark": 7, "Estonia": 8, "Finland": 9, "France": 10, "North Macedonia": 11, "Germany": 12, "Great Britain": 13, "Greece": 14, "Hungary": 15, "Italy": 16, "Latvia": 17, "Lithuania": 18, "Luxembourg": 19, "Montenegro": 20, "Netherlands": 21,  "Poland": 22, "Portugal": 23, "Romania": 24, "Serbia": 25, "Slovakia": 26, "Slovenia": 27, "Spain": 28, "Sweden": 29, "Switzerland": 30, "Turkey": 31} 
  counter = {"Albania": 0, "Austria": 0, "Belgium": 0, "Bosnia Herzegovina": 0, "Bulgaria": 0, "Croatia": 0, "Czech Republic": 0, "Denmark": 0, "Estonia": 0, "Finland": 0, "France": 0, "North Macedonia": 0, "Germany": 0, "Great Britain": 0, "Greece": 0, "Hungary": 0, "Italy": 0, "Latvia": 0, "Lithuania": 0, "Luxembourg": 0, "Montenegro": 0, "Netherlands": 0,  "Poland": 0, "Portugal": 0, "Romania": 0, "Serbia": 0, "Slovakia": 0, "Slovenia": 0, "Spain": 0, "Sweden": 0, "Switzerland": 0, "Turkey": 0} 

  i = 0
  for file in tqdm(files): 
    img = os.path.split(file)[1]

    tensor = image_to_tensor_resize(img, pixels)
    country = img.split("_")[0]
    
    data[i] = tensor
    labels[i] = mapping[country]
    tup = (tensor, mapping[country])
    join[i] = tup
    i += 1
    

  return np.array(data), torch.tensor(labels), np.array(join)

################################################################################
# Save Tensors:                                                                #
################################################################################
def save_tensors(pixels):

  files = sorted(glob.glob(f"/content/drive/MyDrive/GEOMODEL/resizeGAME/*jpg"))

  for file in tqdm(files[8845:]):
    img = os.path.split(file)[1]
    country = img.split("_")[0]
    num = img.split("_")[1]
    pers = img.split("_")[2]
    pers = pers[0]
    tensor = image_to_tensor_resize(img, pixels)
    torch.save(tensor, "/content/drive/MyDrive/GEOMODEL/TENSORS_GAME/{}_{}_{}.t".format(country, num, pers))

  return 
  
################################################################################
# Load Tensors:                                                                #
################################################################################
def load_tensors(data, labels, join):

  files = sorted(glob.glob(f"/content/drive/MyDrive/GEOMODEL/ALL_TENSORS/*t"))
  mapping = {"Albania": 0, "Austria": 1, "Belgium": 2, "Bosnia Herzegovina": 3, "Bulgaria": 4, "Croatia": 5, "Czech Republic": 6, "Denmark": 7, "Estonia": 8, "Finland": 9, "France": 10, "North Macedonia": 11, "Germany": 12, "Great Britain": 13, "Greece": 14, "Hungary": 15, "Italy": 16, "Latvia": 17, "Lithuania": 18, "Luxembourg": 19, "Montenegro": 20, "Netherlands": 21,  "Poland": 22, "Portugal": 23, "Romania": 24, "Serbia": 25, "Slovakia": 26, "Slovenia": 27, "Spain": 28, "Sweden": 29, "Switzerland": 30, "Turkey": 31} 

  i = 0
  for file in tqdm(files): 
    img = os.path.split(file)[1]
    country = img.split("_")[0]

    tensor = torch.load("/content/drive/MyDrive/GEOMODEL/ALL_TENSORS/{}".format(img))
    data[i] = tensor
    labels[i] = mapping[country]
    tup = (tensor, mapping[country])
    join[i] = tup
    i += 1


  return np.array(data), torch.tensor(labels), np.array(join)


################################################################################
# Seed:                                                                   #
################################################################################
import random

random.seed(1)

################################################################################
# Divide images taking only 1 perspective:                                                      #
################################################################################
def split_one_perspective(data, idx):

  n = len(data)

  return idx[0:n:3], data[0:n:3]

################################################################################
# Divide images taking 2 perspectives:                                                      #
################################################################################
def split_two_perspective(data, idx):

  n = len(data)
  to_exclude = range(2, n, 3)
  data, idx = np.array(data), np.array(idx)
  data, idx = np.delete(data, to_exclude), np.delete(idx, to_exclude)
  
  return idx, data

################################################################################
# Divide images taking 3 perspectives:                                          #
################################################################################
def split_three_perspective(data, idx):

  return idx, data


################################################################################
# Divide images into Training:                                                 #
################################################################################
def split_train_test(indices, data, labels, num_perspectives, proportion): # data is a a self.countries_dict
  
  if num_perspectives == "1":
    idx, images = split_one_perspective(data, indices)

  if num_perspectives == "2":
    idx, images = split_two_perspective(data, indices)
  
  if num_perspectives == "3":
    idx, images = split_three_perspective(data, indices)


  train_size = int(proportion * len(idx))
  test_size = len(idx) - train_size
  train_idx, test_idx = torch.utils.data.random_split(idx, [train_size, test_size])

  train, labels_train = data[train_idx], labels[train_idx]
  test, labels_test = data[test_idx], labels[test_idx]

 
  return train, test, labels_train, labels_test


################################################################################
# Accuracy Train:                                                              #
################################################################################

def accuracy(out, target):

    batch_size = target.shape[0]

    _, pred = torch.max(out, dim=-1)

    correct = pred.eq(target).sum() * 1.0

    acc = correct.item() / batch_size

    return acc

