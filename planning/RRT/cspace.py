import matplotlib.pyplot as plt
import matplotlib.patches as patches

class CSpace:
    def __init__(self, size, obsList):
        self.max_x, self.max_y = size
        self.min_x, self.min_y = -self.max_x, -self.max_y
        self.cspace = []
        self.obstacles = obsList

    def plot(self):
        
        for obs in self.obstacles:
            circle = patches.Circle(
                obs[:2],
                radius=obs[2],
                fill=False,
                edgecolor='k',
                linewidth=2
            )
            plt.gca().add_patch(circle)

        plt.plot(
            [self.min_x, self.max_x, self.max_x, self.min_x, self.min_x],
            [self.min_y, self.min_y, self.max_y, self.max_y, self.min_y],
            "-k"
        )