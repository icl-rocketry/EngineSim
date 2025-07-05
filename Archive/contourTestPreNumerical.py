import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

r = np.load('r.npy') #load radius contour
r *= 100 #convert to mm
z = -100*np.load('z.npy') - 1.51#load z coords

df_big = pd.DataFrame()
df_big.loc[:,1] = r
df_big.loc[:,0] = z
df_big.to_csv("contourtest.csv", index=False, header=False)