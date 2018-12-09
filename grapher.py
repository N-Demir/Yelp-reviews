import matplotlib.pyplot as plt
import sys

file_name = sys.argv[1]
avg_over = float(sys.argv[2])

x = []
y = []
with open(file_name, "r") as f:
	i = 0.
	run_distances = []
	for line in f.readlines():
		split_lines = line.split(",")
		run_distances.append(float(split_lines[1]))
		i += 1.
		if i % avg_over == 0:
			x.append(i)
			y.append(sum(run_distances) / avg_over)
			run_distances = []

plt.plot(x, y)
plt.show()