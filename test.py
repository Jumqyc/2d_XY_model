from XY import XY
import numpy as np
import matplotlib.pyplot as plt

model = XY(0.2,32)
model.run(1)
spin = model.get_spin()
spin_x = spin[::2].reshape((model.get_L(), model.get_L()))
spin_y = spin[1::2].reshape((model.get_L(), model.get_L()))
print(spin_x**2 + spin_y**2) # should be 1

plt.quiver(spin_x, spin_y)
plt.show()