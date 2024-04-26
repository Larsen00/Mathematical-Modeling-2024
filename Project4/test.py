# from load_images import load_images
# path = 'Project4/Processedfull'
# target_days = ['0317']
# for target in target_days:
#     V, timesDay, times, mask = load_images(target, path)

import matplotlib.pyplot as plt
from ground import extract_groundintensity

x = extract_groundintensity()
plt.imshow(x, cmap='viridis')
plt.show()