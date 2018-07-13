import numpy as np
import pickle
import sys
import time

A = np.random.rand(100,100)

for i in range(int(sys.argv[1])):
    A = A*A

f = open("monresultat-{}".format(time.time()), "wb")
pickle.dump(A, f)
print("Resultat final :", A)
f.close()
