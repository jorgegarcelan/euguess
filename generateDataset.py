
import os
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import box
import matplotlib.patches as patches
import random
from shapely.geometry import Point
import matplotlib.pyplot as plt
from copy import copy
from pyproj import Proj, transform
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import requests
import time
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
import torch
from PIL import Image
import torchvision.transforms as trans



class generateDataset:

  def __init__(self, api_key):

    
    self.countries_dict = {"Albania": ["al", 369], "Austria": ["at", 369], "Belgium": ["be", 369], "Bosnia Herzegovina": ["ba", 123], "Bulgaria": ["bg", 369], "Croatia": ["hr", 369], "Czech Republic": ["cz", 369], "Denmark": ["dk", 369], "Estonia": ["ee", 369], "Finland": ["fi", 369], "France": ["fr", 860], "North Macedonia": ["mk", 123], "Germany": ["de", 860], "Great Britain": ["gb", 860], "Greece": ["gr", 369], "Hungary": ["hu", 369], "Italy": ["it", 860], "Latvia": ["lv", 123], "Lithuania": ["lt", 369], "Luxembourg": ["lu", 123], "Montenegro": ["me", 123], "Netherlands": ["nl", 369], "Poland": ["pl", 860], "Portugal": ["pt", 369], "Romania": ["ro", 860], "Serbia": ["rs", 369], "Slovakia": ["sk", 369], "Slovenia": ["si", 123], "Spain": ["es", 860], "Sweden": ["se", 369], "Switzerland": ["ch", 369], "Turkey": ["tr", 860]}

    self.api_key = api_key
    self.meta_base = "https://maps.googleapis.com/maps/api/streetview/metadata?"
    self.pic_base = "https://maps.googleapis.com/maps/api/streetview?"
    self.input_proj = Proj(init='epsg:3035')    # define required projections from epsg:
    self.output_proj = Proj(init='epsg:4326')   # define required projections from epsg:
    self.perspectives = [0, 120 , 240]
    self.num_perspectives = len(self.perspectives)
    self.locations = []
    self.labels = []
    self.images = []
    self.track = []

  ################################################################################
  # Generate Country Table:                                                      #
  ################################################################################
  def generate_country_table(self, country):
    """
    Generate num_im in num_loc random location coordinates:

    :param country STRING: name of the country

    :return table:
    """
    os.chdir("/content/drive/MyDrive/GEOMODEL/IMAGES/countries_shp")
    name = self.countries_dict[country][0]
    var = str(name) + '_10km.shp'
    self.table = gpd.read_file(var, dtype={'plz': str})



  ################################################################################
  # Generate Bounds:                                                             #
  ################################################################################
  def generate_bounds(self, geom):
    """
    Generate num_im in num_loc random location coordinates:

    :param geom INT: number of random location coordinates

    :return x, y lists of location coordinates, longetude and latitude
    """
    min_x = min(geom.bounds.minx)
    max_x = max(geom.bounds.maxx)
    min_y = min(geom.bounds.miny)
    max_y = max(geom.bounds.maxy)

    return min_x, max_x, min_y, max_y

  ################################################################################
  # Transfor Image to Tensor:                                                    #
  ################################################################################
  def image_to_tensor(self, name):
    """
    Generate num_im in num_loc random location coordinates:

    :param name STRING: number of random location coordinates

    :return
    """

    # Read a PIL image
    image = Image.open(name)
  
    # image to a Torch tensor
    transform = trans.Compose([trans.PILToTensor()])

    # Convert the PIL image to Torch tensor
    img_tensor = transform(image)
    
    return img_tensor



  ################################################################################
  # Generate Dataset:                                                            #
  ################################################################################
  def generate_random_images(self, num_loc, num_im, perspectives, polygon, country):
    """
    Generate num_im in num_loc random location coordinates:

    :param num_loc INT: number of random location coordinates
    :param num_im INT: number of different perspective from the same location
    :param perspectives LIST of len num_im: angles for the different perspective from the same location (len)
    :param polygon: geopandas.geoseries.GeoSeries the polygon of the region

    :return x, y lists of location coordinates, longetude and latitude
    """

    os.chdir("/content/drive/MyDrive/GEOMODEL/GAME")

    # define boundaries:
    minx, maxx, miny, maxy = self.generate_bounds(polygon)

    # main loop:
    i = 0
    x = []
    y = []
    pbar = tqdm(desc = "while loop", total = num_loc*self.num_perspectives)
    while i < num_loc:
      
      # generate random location coordinates
      x_t = np.random.uniform(minx, maxx)
      y_t = np.random.uniform(miny, maxy)

      for p in polygon:
        if Point(x_t, y_t).within(p):

          y0, x0 = transform(self.input_proj, self.output_proj, x_t, y_t)
          location = str(x0) + "," + str(y0)
          #print(location)

          # define the params for the metadata request:
          meta_params = {'key': self.api_key,
                        'location': location}


          # obtain the metadata:
          meta_response = requests.get(self.meta_base, params=meta_params)
  

          if meta_response.json()['status'] == "OK":

            # define the params for the picture request:

            for j in range(num_im):

              pic_params = {'key': self.api_key,
                  'location': location,
                  'size': "640x640",
                  'heading': self.perspectives[j]}

              
              # GENERATE IMAGE:
              pic_response = requests.get(self.pic_base, params=pic_params)
              self.locations.append(location)
              self.labels.append(country)
              
              
              with open("{}_{}_{}.jpg".format(country, i, j), 'wb') as file:
                  file.write(pic_response.content)

              self.track.append((country, i, j)) # keep track of current country, location and perspective

              pic_response.close()

              pbar.update(1)

              # PLOT: 
              #plt.figure(figsize=(10, 10))
              #img=mpimg.imread("{}_{}_{}.jpg".format(country, i, j))
              #imgplot = plt.imshow(img)
              #plt.show()

            i = i + 1
            x.append(x_t)
            y.append(y_t)
            
    pbar.close()
    return x, y
 


################################################################################
# Generate Dataset Loop:                                                       #
################################################################################

dataset = generateDataset(api_key = "your_api_key")

num_perspectives = dataset.num_perspectives
perspectives = dataset.perspectives


for country in tqdm(dataset.countries_dict):

  # variables for each country:
  num_loc = dataset.countries_dict[country][1]
  dataset.generate_country_table(country)
  polygon = dataset.table["geometry"]

  x, y = dataset.generate_random_images(num_loc, num_perspectives, perspectives, polygon, country)












