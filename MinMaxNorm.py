import numpy as np

an_array = np.random.rand(10)*10
print(an_array)

norm = np.linalg.norm(an_array)
normal_array = an_array/norm
print(normal_array)