import numpy as np

J = 2

eye = np.identity(J)
not_ad = np.repeat(np.identity(J), repeats=J, axis=0).reshape([J, J, J])
for i in range(J):
    not_ad[i, i, i] = 0
print(eye)
print(not_ad)