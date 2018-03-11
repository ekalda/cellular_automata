import numpy as np
import matplotlib.pyplot as plt
from pylab import cm
import time
import simplejson as json
from collections import OrderedDict
import ast


class GameOfLife(object):
    def __init__(self, x_dim=50, y_dim=50, system_type='glider', create_anim=True):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.system_type = system_type
        if system_type == 'random':
            self.system = self.create_random_system()
        elif system_type == 'oscillator':
            self.system = self.create_oscillator_system()
        elif system_type == 'glider':
            self.system = self.create_glider_system()
            # creating a file that will hold the glider data
            timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
            self.out_file = 'glider_' + timestr
        self.alive_neighbours = np.zeros((self.y_dim, self.x_dim))
        self.sweep = self.x_dim * self.y_dim
        self.create_anim = create_anim

    def create_random_system(self):
        print('Creating a random system...')
        system = np.random.randint(0, 2, size=(self.y_dim, self.x_dim))
        return system

    def create_oscillator_system(self):
        assert self.x_dim > 3 and self.y_dim > 3, "system is too small for creating an oscillator"
        print('creating a system with an oscillator...')
        system = np.zeros((self.y_dim, self.x_dim))
        x_mid = int(self.x_dim/2)
        y_mid = int(self.y_dim/2)
        system[y_mid, x_mid] = 1
        system[y_mid+1, x_mid] = 1
        system[y_mid-1, x_mid] = 1
        return system

    def create_glider_system(self):
        assert self.x_dim > 4 and self.y_dim > 4, "system is too small for creating a glider"
        print('creating a system with a glider...')
        system = np.zeros((self.y_dim, self.x_dim))
        x_mid = int(self.x_dim/2)
        y_mid = int(self.y_dim/2)
        system[y_mid-1, x_mid] = 1
        system[y_mid, x_mid+1] = 1
        system[y_mid+1, x_mid-1] = 1
        system[y_mid+1, x_mid] = 1
        system[y_mid+1, x_mid + 1] = 1
        return system

    # method that determines how many neighbours of a site x, y are alive
    def find_alive_neighbours(self, x, y):
        x_len = self.x_dim
        y_len = self.y_dim
        site_n = self.system[(y - 1 + y_len) % y_len, x]
        site_s = self.system[(y + 1) % y_len, x]
        site_e = self.system[y, (x + 1) % x_len]
        site_w = self.system[y, (x - 1 + x_len) % x_len]
        site_ne = self.system[(y - 1 + y_len) % y_len, (x + 1) % x_len]
        site_se = self.system[(y + 1) % y_len, (x + 1) % x_len]
        site_sw = self.system[(y + 1) % y_len, (x - 1 + x_len) % x_len]
        site_nw = self.system[(y - 1 + y_len) % y_len, (x - 1 + x_len) % x_len]
        return site_n + site_s + site_e + site_w + site_ne + site_se + site_sw + site_nw

    # method that loops over the whole alive_neighbours array and updates the number of alive neighbours
    def update_neighbour_array(self):
        for x in range(self.x_dim):
            for y in range(self.y_dim):
                self.alive_neighbours[y, x] = self.find_alive_neighbours(x, y)

    # method that changes (or doesn't change) the state of a site x, y for next timestep
    def change_state(self, x, y):
        n = self.alive_neighbours[y, x]
        if self.system[y, x] == 1:
            if n < 2 or n > 3:
                self.system[y, x] = 0
        else:
            if n == 3:
                self.system[y, x] = 1

    def update_states(self):
        for x in range(self.x_dim):
            for y in range(self.y_dim):
                self.change_state(x, y)

    def animate(self):
        im = plt.imshow(self.system, cmap=cm.summer)
        plt.ion()
        plt.show()
        plt.pause(0.00001)
        plt.cla()

    def find_glider_com(self, time_step):
        x_coords = []
        y_coords = []
        # looping over the whole array to find the glider. Probably not the most efficient way
        for x in range(self.x_dim):
            for y in range(self.y_dim):
                if self.system[y, x] == 1:
                    x_coords.append(x)
                    y_coords.append(y)
        x_com = np.mean(x_coords)
        y_com = np.mean(y_coords)
        x_coords.sort()
        y_coords.sort()
        # printing the coordinates of centre of mass on a file (disregarding data when the glider is crossing the boundary)
        if x_coords[-1] - x_coords[0] < 4 and y_coords[-1] - y_coords[0] < 4:
            with open(self.out_file, 'a+') as f:
                f.write(str(time_step) + ' ' + str(x_com) + ' ' + str(y_com) + '\n')

    def simulate(self):
        for i in range(100*self.sweep):
            self.update_neighbour_array()
            self.update_states()
            if self.create_anim and i % 1 == 0:
                self.animate()
            if self.system_type == 'glider':
                self.find_glider_com(i)


def main():
    # importing data from the input file
    print('reading in the input...')
    with open('input_gol.dat', 'r') as f:
        config = json.load(f, object_pairs_hook=OrderedDict)
    x_dim = config["x dimension"]
    y_dim = config["y dimension"]
    system_type = config["system type"]
    create_anim = ast.literal_eval(config["create animation"])
    print('creating a {x} x {y} system...'.format(x=x_dim, y=y_dim))
    sim = GameOfLife(x_dim=x_dim, y_dim=y_dim, system_type=system_type, create_anim=create_anim)
    print('running the simulation...')
    sim.simulate()
    if system_type == 'glider':
        glider_data = np.loadtxt(sim.out_file)
        timestep = glider_data[:, 0]
        x_coord = glider_data[:, 1]
        y_coord = glider_data[:, 2]
        # find the coordinates of the first stripe for x coordinate
        _min_x = 1
        while glider_data[_min_x-1, 1] <= glider_data[_min_x, 1]:
            _min_x += 1
        _max_x = 0
        while glider_data[_min_x + _max_x, 1] <= glider_data[_min_x + _max_x+1, 1]:
            _max_x += 1
        # coordinates of the first stripe for y
        _min_y = 1
        while glider_data[_min_y - 1, 2] <= glider_data[_min_y, 2]:
            _min_y += 1
        _max_y = 0
        while glider_data[_min_y + _max_y, 2] <= glider_data[_min_y + _max_y+1, 2]:
            _max_y += 1
        fit_x = np.polyfit(timestep[_min_x: _max_x+1], x_coord[_min_x: _max_x+1], 1)
        fit_y = np.polyfit(timestep[_min_y:_max_y+1], y_coord[_min_y: _max_y+1], 1)
        print('glider\'s speed along the x-axis: %s ' %(fit_x[0]))
        print('glider\'s speed along the y-axis: %s ' %(fit_y[0]))
        plt.plot((timestep/sim.sweep), x_coord, '.', label='x')
        plt.plot((timestep/sim.sweep), y_coord, '.', label='y')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()


#sim = GameOfLife()
#sim.create_oscillator_system()
#print(sim.system)
#print(sim.alive_neighbours)
#sim.update_neighbour_array()
#print(sim.alive_neighbours)
#sim.simulate()
#print(sim.system)
#sim.animate()


