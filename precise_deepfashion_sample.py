import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cPickle
import numpy as np

input = open('feat_deepfashion_sample.pkl', 'rb')
feat = cPickle.load(input)
imNamelist = cPickle.load(input)
input.close()
        
lines = open('datasets/DeepFashion/Consumer2Shop/Eval/pair.txt', 'r').readlines()

similarsame = []
similardiff = []
imNamelist = np.array(imNamelist)
for num, line in enumerate(lines):
    pic1 = line.split()[0]
    pic2 = line.split()[1]
    a1 = np.where(imNamelist==pic1)[0]
    if len(a1) == 0:
        continue
    num1 = np.where(imNamelist==pic1)[0][0]
    a2 = np.where(imNamelist==pic2)[0]
    if len(a2) == 0:
        continue
    num2 = a2[0]

    norm1 = 1e-8 + np.sqrt((feat[num1] ** 2).sum())
    norm2 = 1e-8 + np.sqrt((feat[num2] ** 2).sum())
    
    if num >= 1543:
        similardiff.append(np.dot(feat[num1], feat[num2]) / norm1 / norm2)
    else:
        similarsame.append(np.dot(feat[num1], feat[num2]) / norm1 / norm2)

import pdb;pdb.set_trace()    
plt.figure(1)
x = np.linspace(1, len(similarsame), len(similarsame))
plt.plot(x, similarsame)
plt.savefig("1.jpg")

ratioall = []

for threshold in np.arange(0, 1, 0.001):
    numpos = 0
    numneg = 0
    for i in range(len(similarsame)):
        if similarsame[i] >= threshold:
            numpos += 1
        else:
            numneg += 1
        if similardiff[i] < threshold:
            numpos += 1
        else:
            numneg += 1
    
    ratio = float(numpos) / (numpos + numneg)
    ratioall.append(ratio)


plt.figure(2)
x = np.linspace(0.001, 1, 1000)
plt.plot(x, ratioall)
plt.savefig("2.jpg")
