import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from heapq import heappush, heappop

class Graph:

    OCCUPIED = 0
    FREE = 1
    VISITED = 2
    START = 3
    GOAL = 4
    PATH = 5

    def __init__(self, grid, start, goal, method):
        self.grid = grid
        self.start = tuple(np.flip(start, axis=0))
        self.goal = tuple(np.flip(goal, axis=0))
        self.method = method
        self.padded_grid = np.pad(grid, pad_width=1, mode='constant', constant_values=0)

    def show(self):
        grid = self.grid.astype(np.int8).copy()
        grid[self.start] = self.START
        grid[self.goal] = self.GOAL
        plt.close('all')
        fig, ax = plt.subplots()
        im = ax.imshow(grid, cmap='Accent', origin='lower', vmin=0, vmax=5)
        
        def update(ret):
            visited, pos = ret
            if visited:
                grid[pos] = self.VISITED
            else:
                grid[pos] = self.PATH
            im.set_data(grid)
            return [im]
        
        gen = self.get_gen()()
        def frames():
            try:
                while True:
                    yield (True, next(gen))
            except StopIteration as e:
                path = e.value
                if path is not None:
                    for node in path:
                        yield (False, node)

        self.ani = FuncAnimation(fig, update, frames=frames, interval=0.1, repeat=False, blit=True)
        plt.show()

    def get_gen(self):
        if self.method == 'a*':
            return self.Astar_gen
        else:
            raise ValueError(f'{self.method} is not a recognized method')
        
    def neighbors(self, index):
        row, col = index
        local_mask = self.padded_grid[row:row+3, col:col+3] > 0
        vals = np.argwhere(local_mask) + np.asarray(index) - 1
        return [tuple(val) for val in vals if tuple(val) != index]

    def Astar_gen(self):
        def h(index, goal):
            return np.linalg.norm(np.asarray(index) - np.asarray(goal))
        shape = self.grid.shape
        frontier = []
        heappush(frontier, (0, self.start))
        previous = np.full((*shape, 2), -1, dtype=np.int16)
        g_score = np.full(shape, np.inf, dtype=np.float32)
        g_score[self.start] = 0

        while frontier:
            f_cur, cur = heappop(frontier)
            if f_cur > g_score[cur] + h(cur, self.goal):
                continue
            yield cur
            if cur == self.goal:
                return self.reconstruct_path(previous)
            for neigh in self.neighbors(cur):
                h_score = h(neigh, self.goal)
                step_cost = np.sqrt(2) if abs(neigh[0]-cur[0]) + abs(neigh[1]-cur[1]) == 2 else 1.0
                if g_score[neigh] > g_score[cur] + step_cost:
                    previous[neigh] = cur
                    g_score[neigh] = g_score[cur] + step_cost
                    f_score = g_score[neigh] + h_score
                    heappush(frontier, (f_score, neigh))
        return None
    
    def reconstruct_path(self, previous):
        path = []
        cur = self.goal
        while not np.all(previous[cur] == -1):
            path.append(cur)
            cur = tuple(previous[cur])
        path.reverse()
        return path