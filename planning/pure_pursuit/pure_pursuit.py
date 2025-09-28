import numpy as np
import matplotlib.pyplot as plt

def pid(curr, target):
    pass

def pursuit(curr, target):
    pass

if __name__ == "__main__":
    center_line_xlist = np.linspace(10, 30, 100)
    center_line_ylist = 0.1 * (center_line_xlist**2)
    plt.plot(center_line_xlist, center_line_ylist)
    plt.axis('equal')
    plt.grid(True)
    plt.show()
