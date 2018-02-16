import numpy as np
import matplotlib.pyplot as plt

"""Coin-flip random walk example"""

np.random.seed(123)
final_tails = []
for i in range(10000):
    tails = [0]
    for x in range(10):
        coin = np.random.randint(0,2)
        tails.append(tails[x] + coin)
    final_tails.append(tails[-1])
plt.hist(final_tails, bins = 10)
plt.grid()
plt.title('Distribution of Heads/Tails Outcomes')
plt.show()

"""Dice-roll random walk example"""

np.random.seed(123)
all_walks = []
for i in range(200) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))
plt.plot(np_aw_t)
plt.grid()
plt.title('Random-walk Paths')
plt.show()

# Create and plot of distribution of final location
ends = np_aw_t[-1]
plt.hist(ends)
plt.title('Distribution of RW Final Distance')
plt.show()


