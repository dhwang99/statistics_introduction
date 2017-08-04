# encoding:utf8

import matplotlib.pyplot as plt
import numpy as np

grid = np.random.random((20,20))

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(6,10))

ax1.imshow(grid, extent=[0,100,0,100])
ax1.set_title('Default')

ax2.imshow(grid, extent=[0,100,0,1], aspect='auto')
ax2.set_title('Auto-scaled Aspect')

ax3.imshow(grid, extent=[0,100,0,1], aspect=100, cmap='gray')
ax3.set_title('Manually Set Aspect')

plt.tight_layout()

plt.savefig('images/test_imshow.png', format='png')

plt.clf()

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(6,10))

ax1.imshow(grid, extent=[0,100,0,1])
ax1.set_title('Default')

ax2.imshow(grid, extent=[0,1,0,1], aspect='auto')
ax2.set_title('Auto-scaled Aspect')

ax3.imshow(grid, extent=[0,1,0,1], aspect=100)
ax3.set_title('Manually Set Aspect')

plt.tight_layout()

plt.savefig('images/test_imshow2.png', format='png')
