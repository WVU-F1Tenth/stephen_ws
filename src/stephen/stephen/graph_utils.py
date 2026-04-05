import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from heapq import heappush, heappop

class Graph:
    def __init__(self, grid, start, goal, method):
        self.grid = grid
        self.start = np.flip(start, axis=0)
        self.goal = np.flip(goal, axis=0)
        self.method = method
        self.padded_grid = np.pad(grid, pad_width=1, mode='constant', constant_values=0)

    def show(self):
        OCCUPIED = 0
        FREE = 1
        VISITED = 2
        START = 3
        GOAL = 4
        self.gen = self.get_gen()
        grid = self.grid.astype(np.int8)
        grid[tuple(self.start)] = START
        grid[tuple(self.goal)] = GOAL
        plt.close('all')
        fig, ax = plt.subplots()
        im = ax.imshow(grid, cmap='gray', origin='lower', vmin=0, vmax=5)
        def update(pos):
            grid[pos] = VISITED
            im.set_data(grid)
            return [im]
        def frame_gen(grid):
            grid.fill(0)
            return self.gen()
        ani = FuncAnimation(fig, update, frames=frame_gen(grid), interval=200, repeat=True, blit=True)

    def get_gen(self):
        if self.method == 'a*':
            return self.Astar_gen
        else:
            raise ValueError(f'{self.method} is not a recognized method')
        
    def neighbors(self, index):
        row, col = index
        return np.argwhere(self.padded_grid[row-1:row+1, col-1:col+1] > 0) + index

    def Astar_gen(self):
        def h(index, goal):
            return np.linalg.norm(index, goal)
        shape = self.grid.shape
        frontier = []
        heappush(frontier, (0, self.start))
        previous = np.full((*shape, 2), -1, dtype=np.int16)
        g_score = np.full(shape, np.inf, dtype=np.float32)
        g_score[self.start] = 0

        while frontier:
            _, cur = heappop(frontier)
            if cur == self.goal:
                return cur
            for neigh in self.neighbors(cur):
                h_score = h(neigh, self.goal)
                if g_score[neigh] > g_score[cur] + 1:
                    previous[neigh] = cur
                    g_score[neigh] = g_score[cur] + 1
                    f_score = g_score[cur] + h_score
                    heappush(frontier, (f_score, neigh))
                    yield neigh
        return None
