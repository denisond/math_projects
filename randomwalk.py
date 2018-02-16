import random

def random_walk(n):
    x = 0
    y = 0
    for i in range(n):
        (dx, dy) = random.choice([(0, 1), (0, -1), (1,0), (-1,0)])
        x += dx
        y += dy

    return (x,y)

number_of_walks = 10000

for walk_length in range(1, 31):
    short_walk = 0
    for i in range(number_of_walks):
        (x,y) = (random_walk(walk_length))
        distance = abs(x) + abs(y)
        if distance <= 5:
            short_walk += 1
    print("If the walk length is equal to " + str(walk_length) + ", then " + str('%.4f'%((short_walk/number_of_walks)*100)) + "% of the walks will be short.")



for i in range(1,31):
    walk = random_walk(10)
    print(walk, "Distance from home:", abs(walk[0]) + abs(walk[1]))
    random_walk.append(abs(walk[0]) + abs(walk[1]))
average_walk = functools.reduce(lambda x, y: x + y, random_walk)
print(average_walk/10000)



