import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap

models = ("Validation\nAccuracy", "Test\nAccuracy", "Test\nAccuracy\n(majority voting)")
values = {
    'CNN': (64.3, 62.6, 75),
    'CNN with attention': (62.8, 61.7, 80),
}

x = np.arange(len(models))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

cmap = get_cmap('Blues')

for attribute, measurement in values.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=cmap(50*multiplier*len(values)+100))
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy %')
ax.set_xticks(x + width, models)
plt.xticks(rotation = 70)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 100)
ax.spines[['right', 'top']].set_visible(False)

models_att = ("Validation\nAccuracy", "Test 1\nAccuracy", "Test 1\nAccuracy\n(majority voting)", "Test 2\nAccuracy", "Test 2\nAccuracy\n(majority voting)", "Test 3\nAccuracy", "Test 3\nAccuracy\n(majority voting)")
values_att = {
    'CNN': (78.7, 47.9, 61.2, 45.2, 50, 57.4, 70),
    'CNN with attention': (80.9, 48.1, 68.7, 46.1, 50, 58.4, 72.5),
}

x = np.arange(len(models_att))  # the label locations
width = 0.40  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in values_att.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=cmap(50*multiplier*len(values)+100))
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy %')
ax.set_xticks(x + width, models_att)
plt.xticks(rotation = 70)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 100)
ax.spines[['right', 'top']].set_visible(False)

plt.show()