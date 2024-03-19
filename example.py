import matplotlib.pyplot as plt

# Disable axes
plt.axis('off')

# Draw your object
# For example, let's say you're drawing a circle
circle = plt.Circle((0.5, 0.5), 0.4, color='red')

# Get current axes
ax = plt.gca()

# Add the circle to the axes
ax.add_patch(circle)

# Adjust the limits of the axes to include the entire object
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.show()