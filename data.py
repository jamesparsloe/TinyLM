from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np

ds = load_dataset("roneneldan/TinyStories")
train_ds = ds["train"]

lengths = []
for item in train_ds:
    text = item["text"]
    lengths.append(len(text))


fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].hist(lengths, bins=1000)
ax[1].hist(lengths, bins=1000, cumulative=True, histtype="step")
ax[1].set_xlabel("Length (characters)")
plt.suptitle("TinyStories length distribution")
plt.savefig("dist.png")
