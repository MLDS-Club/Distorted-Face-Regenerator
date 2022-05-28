import sys
import pickle
import os
import numpy as np
import PIL.Image
from IPython.display import Image
import matplotlib.pyplot as plt

sys.path.insert(0, '/content/stylegan2-ada')

import dnnlib
import dnnlib.tflib as tflib

def seed2vec(Gs, seed):
  rnd = np.random.RandomState(seed)
  return rnd.randn(1, *Gs.input_shape[1:])

def init_random_state(Gs, seed):
  rnd = np.random.RandomState(seed)
  noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
  tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})

def display_image(image):
  plt.axis('off')
  plt.imshow(image)
  plt.show()

def generate_image(Gs, z, truncation_psi):
  Gs_kwargs = {
      'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
      'randomize_noise': False
  }
  if truncation_psi is not None:
    Gs_kwargs['truncation_psi'] = truncation_psi
  
  label = np.zeros([1] + Gs.input_shapes[1][1:])
  images = Gs.run(z, label, **Gs_kwargs)
  return images[0]