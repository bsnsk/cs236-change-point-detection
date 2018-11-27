#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

path = "./data/raw/syn"
labelFile = path + "label.txt"


def sampleMu():
    return np.random.randint(0, 50)


def sampleSigma():
    return np.random.randint(5, 10)


def genData(tag):
    length = 1000
    numSegs = np.random.randint(2, 10)
    while True:
        starts = np.random.choice(range(length), numSegs - 1)
        starts = np.sort(np.insert(starts, 0, 0))
        segLens = [starts[i] - starts[i-1] for i in range(1, numSegs)]
        if segLens == [] or min(segLens) >= length / 10:
            break
    print("seg={}, starts={}".format(numSegs, starts))
    data = []
    mus = []
    mu, sigma = sampleMu(), sampleSigma()
    for i in range(numSegs):
        mus.append(mu)
        size = starts[i + 1] - starts[i] if i + 1 < numSegs \
            else length - starts[i]
        x = np.random.normal(mu, sigma, size)
        data = data + list(x)
        previousMu = mu
        while True:
            mu, sigma = sampleMu(), sampleSigma()
            if abs(previousMu - mu) > 10:
                break
    with open("{}{}.txt".format(path, tag), "w") as f:
        for x in data:
            f.write("{} ".format(x))
    plt.clf()
    plt.plot(range(length), data, '.-')
    plt.plot(starts, mus, 'dr')
    plt.savefig("{}{}.png".format(path, tag))

    with open(labelFile, "a") as f:
        f.write("{} {}\n".format(tag, starts[1:]))


with open(labelFile, "w") as f:
    pass
for i in range(50):
    genData(str(i))
