import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
import os
import sys

def rastrigin(x, y):
    return 20 + x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y)

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def ackley(x, y):
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    return term1 + term2 + 20 + np.e

def sphere(x, y):

    return x**2 + y**2

def zakharov(x, y):

    sum1 = x**2 + y**2
    sum2 = 0.5*x + y
    sum3 = sum2**2
    sum4 = sum2**4
    return sum1 + sum3 + sum4

def griewank(x, y):

    term1 = (x**2 + y**2) / 4000
    term2 = np.cos(x / np.sqrt(1)) * np.cos(y / np.sqrt(2))
    return term1 - term2 + 1

def schwefel(x, y):

    term1 = x * np.sin(np.sqrt(np.abs(x)))
    term2 = y * np.sin(np.sqrt(np.abs(y)))
    return 418.9829 * 2 - term1 - term2

def michalewicz(x, y):

    m = 10  
    term1 = np.sin(x) * np.sin((1 * x**2) / np.pi)**(2*m)
    term2 = np.sin(y) * np.sin((2 * y**2) / np.pi)**(2*m)
    return -1 * (term1 + term2)

def easom(x, y):

    term1 = -np.cos(x) * np.cos(y)
    term2 = np.exp(-((x - np.pi)**2 + (y - np.pi)**2))
    return term1 * term2

def griewank(x, y):

    term1 = (x**2 + y**2) / 4000
    term2 = np.cos(x / np.sqrt(1)) * np.cos(y / np.sqrt(2))
    return term1 - term2 + 1

def schwefel(x, y):

    term1 = x * np.sin(np.sqrt(np.abs(x)))
    term2 = y * np.sin(np.sqrt(np.abs(y)))
    return 418.9829 * 2 - term1 - term2

def michalewicz(x, y):

    m = 10  
    term1 = np.sin(x) * np.sin((1 * x**2) / np.pi)**(2*m)
    term2 = np.sin(y) * np.sin((2 * y**2) / np.pi)**(2*m)
    return -1 * (term1 + term2)

def easom(x, y):

    term1 = -np.cos(x) * np.cos(y)
    term2 = np.exp(-((x - np.pi)**2 + (y - np.pi)**2))
    return term1 * term2

class BlackHoleOptimization:
    def __init__(self, objective_func, n_stars=50, dimensions=2, bounds=(-5.12, 5.12), max_iter=100):
        self.objective_func = objective_func
        self.n_stars = n_stars
        self.dimensions = dimensions
        self.bounds = bounds
        self.max_iter = max_iter
        
        self.stars = np.random.uniform(bounds[0], bounds[1], (n_stars, dimensions))
        self.star_fitness = np.zeros(n_stars)
        
        self.black_hole = None
        self.black_hole_fitness = float('inf')
        
        self.history = []
        
    def update_fitness(self):
        for i in range(self.n_stars):
            self.star_fitness[i] = self.objective_func(self.stars[i, 0], self.stars[i, 1])
            
        best_idx = np.argmin(self.star_fitness)
        
        if self.star_fitness[best_idx] < self.black_hole_fitness:
            self.black_hole = self.stars[best_idx].copy()
            self.black_hole_fitness = self.star_fitness[best_idx]
    
    def calculate_event_horizon(self):
        total_fitness = np.sum(self.star_fitness)
        return self.black_hole_fitness / total_fitness
    
    def optimize(self):
        for iteration in range(self.max_iter):
            self.update_fitness()
            
            event_horizon = self.calculate_event_horizon()
            
            for i in range(self.n_stars):
                distance = np.linalg.norm(self.stars[i] - self.black_hole)
                
                r = np.random.random()
                self.stars[i] = self.stars[i] + r * (self.black_hole - self.stars[i])
                
                if distance < event_horizon:
                    self.stars[i] = np.random.uniform(self.bounds[0], self.bounds[1], self.dimensions)
            
            self.stars = np.clip(self.stars, self.bounds[0], self.bounds[1])
            
            self.history.append({
                'stars': self.stars.copy(),
                'black_hole': self.black_hole.copy(),
                'event_horizon': event_horizon,
                'iteration': iteration,
                'best_fitness': self.black_hole_fitness
            })
            
        return self.black_hole, self.black_hole_fitness, self.history

class BigBangBigCrunch:
    def __init__(self, objective_func, n_particles=50, dimensions=2, bounds=(-5.12, 5.12), 
                 max_iter=100, initial_explosion_radius=10.0, final_explosion_radius=0.1):
        self.objective_func = objective_func
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.max_iter = max_iter
        self.initial_explosion_radius = initial_explosion_radius
        self.final_explosion_radius = final_explosion_radius
        
        self.particles = np.random.uniform(bounds[0], bounds[1], (n_particles, dimensions))
        self.fitness = np.zeros(n_particles)
        
        self.center_of_mass = None
        self.best_fitness = float('inf')
        self.best_solution = None
        
        self.history = []
    
    def calculate_fitness(self):
        for i in range(self.n_particles):
            self.fitness[i] = self.objective_func(self.particles[i, 0], self.particles[i, 1])
            
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.particles[i].copy()
    
    def big_crunch(self):
        epsilon = 1e-10
        
        inverse_fitness = 1.0 / (self.fitness + epsilon)
        total_inverse_fitness = np.sum(inverse_fitness)
        
        self.center_of_mass = np.zeros(self.dimensions)
        for i in range(self.n_particles):
            weight = inverse_fitness[i] / total_inverse_fitness
            self.center_of_mass += weight * self.particles[i]
    
    def big_bang(self, iteration):
        alpha = iteration / self.max_iter
        current_radius = self.initial_explosion_radius * (1 - alpha) + self.final_explosion_radius * alpha
        
        for i in range(self.n_particles):
            random_offset = np.random.normal(0, current_radius, self.dimensions)
            self.particles[i] = self.center_of_mass + random_offset
            
        self.particles = np.clip(self.particles, self.bounds[0], self.bounds[1])
    
    def optimize(self):
        self.calculate_fitness()
        
        self.history.append({
            'particles': self.particles.copy(),
            'center_of_mass': None,
            'iteration': 0,
            'radius': self.initial_explosion_radius,
            'best_fitness': self.best_fitness,
            'best_solution': self.best_solution.copy() if self.best_solution is not None else None
        })
        
        for iteration in range(self.max_iter):
            self.big_crunch()
            
            self.history.append({
                'particles': self.particles.copy(),
                'center_of_mass': self.center_of_mass.copy(),
                'iteration': iteration + 0.5,
                'radius': 0,
                'best_fitness': self.best_fitness,
                'best_solution': self.best_solution.copy()
            })
            
            self.big_bang(iteration)
            
            self.calculate_fitness()
            
            self.history.append({
                'particles': self.particles.copy(),
                'center_of_mass': self.center_of_mass.copy(),
                'iteration': iteration + 1,
                'radius': self.initial_explosion_radius * (1 - (iteration + 1) / self.max_iter) + 
                          self.final_explosion_radius * ((iteration + 1) / self.max_iter),
                'best_fitness': self.best_fitness,
                'best_solution': self.best_solution.copy()
            })
        
        return self.best_solution, self.best_fitness, self.history

class SolarSystemOptimization:
    def __init__(self, objective_func, n_planets=10, dimensions=2, bounds=(-5.12, 5.12), 
                 max_iter=100, eccentricity_range=(0.1, 0.7), orbit_period_range=(3, 10),
                 perturbation_prob=0.1, perturbation_strength=0.2):
        self.objective_func = objective_func
        self.n_planets = n_planets
        self.dimensions = dimensions
        self.bounds = bounds
        self.max_iter = max_iter
        self.eccentricity_range = eccentricity_range
        self.orbit_period_range = orbit_period_range
        self.perturbation_prob = perturbation_prob
        self.perturbation_strength = perturbation_strength
        
        self.planets = np.random.uniform(bounds[0], bounds[1], (n_planets, dimensions))
        self.planet_fitness = np.zeros(n_planets)
        
        self.semi_major_axes = np.zeros(n_planets)
        self.eccentricities = np.random.uniform(eccentricity_range[0], eccentricity_range[1], n_planets)
        self.orbit_periods = np.random.randint(orbit_period_range[0], orbit_period_range[1], n_planets)
        self.current_angles = np.random.uniform(0, 2*np.pi, n_planets)
        
        self.sun = None
        self.sun_fitness = float('inf')
        
        self.history = []
        
    def update_fitness(self):
        for i in range(self.n_planets):
            self.planet_fitness[i] = self.objective_func(self.planets[i, 0], self.planets[i, 1])
            
            if self.planet_fitness[i] < self.sun_fitness:
                self.sun = self.planets[i].copy()
                self.sun_fitness = self.planet_fitness[i]
    
    def update_orbital_parameters(self):
        max_fitness = np.max(self.planet_fitness)
        min_fitness = np.min(self.planet_fitness)
        
        if max_fitness == min_fitness:
            relative_fitness = np.ones(self.n_planets) * 0.5
        else:
            relative_fitness = (self.planet_fitness - min_fitness) / (max_fitness - min_fitness)
        
        min_axis = 0.5
        max_axis = 5.0
        self.semi_major_axes = min_axis + relative_fitness * (max_axis - min_axis)
    
    def move_planets(self, iteration):
        for i in range(self.n_planets):
            angle_increment = 2 * np.pi / self.orbit_periods[i]
            self.current_angles[i] = (self.current_angles[i] + angle_increment) % (2 * np.pi)
            
            a = self.semi_major_axes[i]
            e = self.eccentricities[i]
            b = a * np.sqrt(1 - e**2)
            
            angle = self.current_angles[i]
            relative_x = a * np.cos(angle)
            relative_y = b * np.sin(angle)
            
            if np.random.random() < self.perturbation_prob:
                relative_x += np.random.normal(0, self.perturbation_strength)
                relative_y += np.random.normal(0, self.perturbation_strength)
            
            self.planets[i, 0] = self.sun[0] + relative_x
            self.planets[i, 1] = self.sun[1] + relative_y
            
        self.planets = np.clip(self.planets, self.bounds[0], self.bounds[1])
    
    def optimize(self):
        self.update_fitness()
        
        self.history.append({
            'planets': self.planets.copy(),
            'sun': self.sun.copy(),
            'semi_major_axes': self.semi_major_axes.copy(),
            'eccentricities': self.eccentricities.copy(),
            'current_angles': self.current_angles.copy(),
            'iteration': 0,
            'sun_fitness': self.sun_fitness
        })
        
        self.update_orbital_parameters()
        
        for iteration in range(self.max_iter):
            self.move_planets(iteration)
            
            self.update_fitness()
            
            self.update_orbital_parameters()
            
            self.history.append({
                'planets': self.planets.copy(),
                'sun': self.sun.copy(),
                'semi_major_axes': self.semi_major_axes.copy(),
                'eccentricities': self.eccentricities.copy(),
                'current_angles': self.current_angles.copy(),
                'iteration': iteration + 1,
                'sun_fitness': self.sun_fitness
            })
        
        return self.sun, self.sun_fitness, self.history

class MultiverseOptimizer:
    def __init__(self, objective_func, n_universes=50, dimensions=2, bounds=(-5.12, 5.12), 
                 max_iter=100, wormhole_existence_prob=0.6, travelling_distance_rate_min=0.2,
                 travelling_distance_rate_max=0.8):
        self.objective_func = objective_func
        self.n_universes = n_universes
        self.dimensions = dimensions
        self.bounds = bounds
        self.max_iter = max_iter
        self.wormhole_existence_prob = wormhole_existence_prob
        self.tdr_min = travelling_distance_rate_min
        self.tdr_max = travelling_distance_rate_max
        
        self.universes = np.random.uniform(bounds[0], bounds[1], (n_universes, dimensions))
        self.universe_fitness = np.zeros(n_universes)
        self.universe_inflation_rates = np.zeros(n_universes) 
        
        self.best_universe = None
        self.best_fitness = float('inf')
        
        self.history = []
    
    def calculate_fitness(self):
        for i in range(self.n_universes):
            self.universe_fitness[i] = self.objective_func(self.universes[i, 0], self.universes[i, 1])
            
            if self.universe_fitness[i] < self.best_fitness:
                self.best_fitness = self.universe_fitness[i]
                self.best_universe = self.universes[i].copy()
    
    def calculate_inflation_rates(self):
        max_fitness = np.max(self.universe_fitness)
        min_fitness = np.min(self.universe_fitness)
        
        if max_fitness == min_fitness:
            self.universe_inflation_rates = np.ones(self.n_universes) * 0.5
        else:
            self.universe_inflation_rates = 1 - (self.universe_fitness - min_fitness) / (max_fitness - min_fitness)
    
    def sort_universes(self):
        sorted_indices = np.argsort(-self.universe_inflation_rates)
        self.universes = self.universes[sorted_indices]
        self.universe_fitness = self.universe_fitness[sorted_indices]
        self.universe_inflation_rates = self.universe_inflation_rates[sorted_indices]
    
    def update_universes(self, iteration):
        normalized_iter = 1 - iteration / self.max_iter
        
        wep = self.wormhole_existence_prob * normalized_iter
        
        tdr = self.tdr_min + normalized_iter * (self.tdr_max - self.tdr_min)
        
        roulette_wheel = np.cumsum(self.universe_inflation_rates)
        
        previous_fitness = self.universe_fitness.copy()
        
        for i in range(self.n_universes):
            r1 = np.random.random()
            selected_universe_idx = np.searchsorted(roulette_wheel, r1 * roulette_wheel[-1])
            selected_universe_idx = min(selected_universe_idx, self.n_universes - 1) 
            
            random_dim = np.random.randint(0, self.dimensions)
            self.universes[i, random_dim] = self.universes[selected_universe_idx, random_dim]
            
            r2 = np.random.random(self.dimensions)
            r3 = np.random.random(self.dimensions)
            r4 = np.random.random(self.dimensions)
            
            for j in range(self.dimensions):
                if r2[j] < wep:  
                    if r3[j] < 0.5:  
                        distance = tdr * ((self.bounds[1] - self.bounds[0]) * r4[j] + self.bounds[0])
                        self.universes[i, j] = self.best_universe[j] + distance
                    else:  
                        distance = tdr * ((self.bounds[1] - self.bounds[0]) * r4[j] + self.bounds[0])
                        self.universes[i, j] = self.best_universe[j] - distance
        
        self.universes = np.clip(self.universes, self.bounds[0], self.bounds[1])
        
        if self.objective_func == schwefel:
            for i in range(self.n_universes):
                self.universe_fitness[i] = self.objective_func(self.universes[i, 0], self.universes[i, 1])
                
            self.universes = schwefel_handler(
                "Multiverse", 
                self.universes, 
                self.bounds, 
                self.universe_fitness, 
                previous_fitness
            )
    
    def optimize(self):
        self.calculate_fitness()
        self.calculate_inflation_rates()
        
        if self.best_universe is None:
            best_idx = np.argmin(self.universe_fitness)
            self.best_universe = self.universes[best_idx].copy()
            self.best_fitness = self.universe_fitness[best_idx]
        
        self.history.append({
            'universes': self.universes.copy(),
            'best_universe': self.best_universe.copy(),
            'inflation_rates': self.universe_inflation_rates.copy(),
            'iteration': 0,
            'best_fitness': self.best_fitness,
            'wep': self.wormhole_existence_prob,
            'tdr': self.tdr_min
        })
        
        for iteration in range(self.max_iter):
            self.sort_universes()
            
            self.update_universes(iteration)
            
            self.calculate_fitness()
            self.calculate_inflation_rates()
            
            normalized_iter = 1 - iteration / self.max_iter
            wep = self.wormhole_existence_prob * normalized_iter
            tdr = self.tdr_min + normalized_iter * (self.tdr_max - self.tdr_min)
            
            self.history.append({
                'universes': self.universes.copy(),
                'best_universe': self.best_universe.copy(),
                'inflation_rates': self.universe_inflation_rates.copy(),
                'iteration': iteration + 1,
                'best_fitness': self.best_fitness,
                'wep': wep,
                'tdr': tdr
            })
        
        return self.best_universe, self.best_fitness, self.history

class MultiverseOptimizer:
    def __init__(self, objective_func, n_universes=50, dimensions=2, bounds=(-5.12, 5.12), 
                 max_iter=100, wormhole_existence_prob=0.6, travelling_distance_rate_min=0.2,
                 travelling_distance_rate_max=0.8):
        self.objective_func = objective_func
        self.n_universes = n_universes
        self.dimensions = dimensions
        self.bounds = bounds
        self.max_iter = max_iter
        self.wormhole_existence_prob = wormhole_existence_prob
        self.tdr_min = travelling_distance_rate_min
        self.tdr_max = travelling_distance_rate_max
        
        self.universes = np.random.uniform(bounds[0], bounds[1], (n_universes, dimensions))
        self.universe_fitness = np.zeros(n_universes)
        self.universe_inflation_rates = np.zeros(n_universes) 
        
        self.best_universe = None
        self.best_fitness = float('inf')
        
        self.history = []
    
    def calculate_fitness(self):
        for i in range(self.n_universes):
            self.universe_fitness[i] = self.objective_func(self.universes[i, 0], self.universes[i, 1])
            
            if self.universe_fitness[i] < self.best_fitness:
                self.best_fitness = self.universe_fitness[i]
                self.best_universe = self.universes[i].copy()
    
    def calculate_inflation_rates(self):
        max_fitness = np.max(self.universe_fitness)
        min_fitness = np.min(self.universe_fitness)
        
        if max_fitness == min_fitness:
            self.universe_inflation_rates = np.ones(self.n_universes) * 0.5
        else:
            self.universe_inflation_rates = 1 - (self.universe_fitness - min_fitness) / (max_fitness - min_fitness)
    
    def sort_universes(self):
        sorted_indices = np.argsort(-self.universe_inflation_rates)
        self.universes = self.universes[sorted_indices]
        self.universe_fitness = self.universe_fitness[sorted_indices]
        self.universe_inflation_rates = self.universe_inflation_rates[sorted_indices]
    
    def update_universes(self, iteration):
        normalized_iter = 1 - iteration / self.max_iter
        
        wep = self.wormhole_existence_prob * normalized_iter
        
        tdr = self.tdr_min + normalized_iter * (self.tdr_max - self.tdr_min)
        
        roulette_wheel = np.cumsum(self.universe_inflation_rates)
        
        for i in range(self.n_universes):
            r1 = np.random.random()
            selected_universe_idx = np.searchsorted(roulette_wheel, r1 * roulette_wheel[-1])
            selected_universe_idx = min(selected_universe_idx, self.n_universes - 1)  
            
            random_dim = np.random.randint(0, self.dimensions)
            self.universes[i, random_dim] = self.universes[selected_universe_idx, random_dim]
            
            r2 = np.random.random(self.dimensions)
            r3 = np.random.random(self.dimensions)
            r4 = np.random.random(self.dimensions)
            
            for j in range(self.dimensions):
                if r2[j] < wep:  
                    if r3[j] < 0.5:  
                        distance = tdr * ((self.bounds[1] - self.bounds[0]) * r4[j] + self.bounds[0])
                        self.universes[i, j] = self.best_universe[j] + distance
                    else:  
                        distance = tdr * ((self.bounds[1] - self.bounds[0]) * r4[j] + self.bounds[0])
                        self.universes[i, j] = self.best_universe[j] - distance
        
        self.universes = np.clip(self.universes, self.bounds[0], self.bounds[1])
    
    def optimize(self):
        self.calculate_fitness()
        self.calculate_inflation_rates()
        
        if self.best_universe is None:
            best_idx = np.argmin(self.universe_fitness)
            self.best_universe = self.universes[best_idx].copy()
            self.best_fitness = self.universe_fitness[best_idx]
        
        self.history.append({
            'universes': self.universes.copy(),
            'best_universe': self.best_universe.copy(),
            'inflation_rates': self.universe_inflation_rates.copy(),
            'iteration': 0,
            'best_fitness': self.best_fitness,
            'wep': self.wormhole_existence_prob,
            'tdr': self.tdr_min
        })
        
        for iteration in range(self.max_iter):
            self.sort_universes()
            
            self.update_universes(iteration)
            
            self.calculate_fitness()
            self.calculate_inflation_rates()
            
            normalized_iter = 1 - iteration / self.max_iter
            wep = self.wormhole_existence_prob * normalized_iter
            tdr = self.tdr_min + normalized_iter * (self.tdr_max - self.tdr_min)
            
            self.history.append({
                'universes': self.universes.copy(),
                'best_universe': self.best_universe.copy(),
                'inflation_rates': self.universe_inflation_rates.copy(),
                'iteration': iteration + 1,
                'best_fitness': self.best_fitness,
                'wep': wep,
                'tdr': tdr
            })
        
        return self.best_universe, self.best_fitness, self.history

class GalaxyBasedSearch:
    def __init__(self, objective_func, n_stars=50, dimensions=2, bounds=(-5.12, 5.12), 
                 max_iter=100, spiral_constant=0.7, rotation_angle=0.01, arm_count=3,
                 local_search_radius=0.1):
        self.objective_func = objective_func
        self.n_stars = n_stars
        self.dimensions = dimensions
        self.bounds = bounds
        self.max_iter = max_iter
        self.spiral_constant = spiral_constant
        self.rotation_angle = rotation_angle
        self.arm_count = arm_count
        self.local_search_radius = local_search_radius
        
        self.stars = np.random.uniform(bounds[0], bounds[1], (n_stars, dimensions))
        self.star_fitness = np.zeros(n_stars)
        self.star_radius = np.zeros(n_stars)
        self.star_angle = np.random.uniform(0, 2*np.pi, n_stars)
        self.star_arm = np.random.randint(0, arm_count, n_stars)
        
        self.black_hole = None  
        self.black_hole_fitness = float('inf')
        
        self.history = []
    
    def update_fitness(self):
        for i in range(self.n_stars):
            self.star_fitness[i] = self.objective_func(self.stars[i, 0], self.stars[i, 1])
            
            if self.star_fitness[i] < self.black_hole_fitness:
                self.black_hole = self.stars[i].copy()
                self.black_hole_fitness = self.star_fitness[i]
    
    def calculate_star_radius(self):
        max_fitness = np.max(self.star_fitness)
        min_fitness = np.min(self.star_fitness)
        
        if max_fitness == min_fitness:
            normalized_fitness = np.ones(self.n_stars) * 0.5
        else:
            normalized_fitness = (self.star_fitness - min_fitness) / (max_fitness - min_fitness)
        
        min_radius = 0.1
        max_radius = 4.0
        self.star_radius = max_radius * normalized_fitness + min_radius
    
    def move_stars(self):
        previous_fitness = self.star_fitness.copy()
        
        for i in range(self.n_stars):
            self.star_angle[i] += self.rotation_angle
            arm_offset = 2 * np.pi * self.star_arm[i] / self.arm_count
            
            r = self.star_radius[i] * np.exp(self.spiral_constant * self.star_angle[i])
            theta = self.star_angle[i] + arm_offset
            
            relative_x = r * np.cos(theta)
            relative_y = r * np.sin(theta)
            
            local_search = np.random.uniform(-self.local_search_radius, 
                                            self.local_search_radius, 
                                            self.dimensions)
            
            self.stars[i, 0] = self.black_hole[0] + relative_x + local_search[0]
            self.stars[i, 1] = self.black_hole[1] + relative_y + local_search[1]
        
        self.stars = np.clip(self.stars, self.bounds[0], self.bounds[1])
        
        if self.objective_func == schwefel:
            for i in range(self.n_stars):
                self.star_fitness[i] = self.objective_func(self.stars[i, 0], self.stars[i, 1])
                
            self.stars = schwefel_handler(
                "Galaxy", 
                self.stars, 
                self.bounds, 
                self.star_fitness, 
                previous_fitness
            )
    
    def optimize(self):
        self.update_fitness()
        self.calculate_star_radius()
        
        self.history.append({
            'stars': self.stars.copy(),
            'black_hole': self.black_hole.copy() if self.black_hole is not None else None,
            'star_radius': self.star_radius.copy(),
            'star_angle': self.star_angle.copy(),
            'star_arm': self.star_arm.copy(),
            'iteration': 0,
            'best_fitness': self.black_hole_fitness
        })
        
        for iteration in range(self.max_iter):
            self.move_stars()
            self.update_fitness()
            self.calculate_star_radius()
            
            self.history.append({
                'stars': self.stars.copy(),
                'black_hole': self.black_hole.copy(),
                'star_radius': self.star_radius.copy(),
                'star_angle': self.star_angle.copy(),
                'star_arm': self.star_arm.copy(),
                'iteration': iteration + 1,
                'best_fitness': self.black_hole_fitness
            })
        
        return self.black_hole, self.black_hole_fitness, self.history

class GravitationalSearch:
    def __init__(self, objective_func, n_agents=50, dimensions=2, bounds=(-5.12, 5.12), 
                 max_iter=100, G_init=100, G_reduction=0.8, epsilon=1e-10,
                 best_agent_ratio=0.05):
        self.objective_func = objective_func
        self.n_agents = n_agents
        self.dimensions = dimensions
        self.bounds = bounds
        self.max_iter = max_iter
        self.G_init = G_init  
        self.G_reduction = G_reduction 
        self.epsilon = epsilon  
        self.best_agent_ratio = best_agent_ratio  
        
        self.agents = np.random.uniform(bounds[0], bounds[1], (n_agents, dimensions))
        self.velocities = np.zeros((n_agents, dimensions))
        self.agent_fitness = np.zeros(n_agents)
        self.agent_mass = np.ones(n_agents)
        
        self.best_agent = None
        self.best_fitness = float('inf')
        self.G = G_init  
        
        self.history = []
    
    def update_fitness(self):
        for i in range(self.n_agents):
            self.agent_fitness[i] = self.objective_func(self.agents[i, 0], self.agents[i, 1])
            
            if self.agent_fitness[i] < self.best_fitness:
                self.best_agent = self.agents[i].copy()
                self.best_fitness = self.agent_fitness[i]
    
    def calculate_mass(self):
        max_fitness = np.max(self.agent_fitness)
        min_fitness = np.min(self.agent_fitness)
        
        if max_fitness == min_fitness:
            normalized_fitness = np.ones(self.n_agents)
        else:
            
            normalized_fitness = (max_fitness - self.agent_fitness) / (max_fitness - min_fitness)
        
        
        mass_sum = np.sum(normalized_fitness)
        if mass_sum > 0:
            self.agent_mass = normalized_fitness / mass_sum
        else:
            self.agent_mass = np.ones(self.n_agents) / self.n_agents
    
    def update_gravitational_constant(self, iteration):
        
        alpha = 20
        ratio = iteration / self.max_iter
        self.G = self.G_init * np.exp(-alpha * ratio)
    
    def calculate_forces(self):
        forces = np.zeros((self.n_agents, self.dimensions))
        
        
        n_best = max(1, int(self.n_agents * self.best_agent_ratio))
        best_indices = np.argsort(self.agent_fitness)[:n_best]
        
        for i in range(self.n_agents):
            for j in best_indices:
                if i != j:
                    
                    diff = self.agents[j] - self.agents[i]
                    distance = np.linalg.norm(diff) + self.epsilon
                    
                    force_magnitude = self.G * (self.agent_mass[i] * self.agent_mass[j]) / (distance**2)
                    
                    force_direction = diff / distance
                    
                    forces[i] += force_magnitude * force_direction
        
        return forces
    
    def update_velocities_and_positions(self, forces):
        previous_fitness = self.agent_fitness.copy()
        
        random_coef = np.random.random(size=(self.n_agents, self.dimensions))
        
        for i in range(self.n_agents):
            acceleration = forces[i] / (self.agent_mass[i] + self.epsilon)
            self.velocities[i] = random_coef[i] * self.velocities[i] + acceleration
        
        self.agents += self.velocities
        
        self.agents = np.clip(self.agents, self.bounds[0], self.bounds[1])
        
        if self.objective_func == schwefel:
            for i in range(self.n_agents):
                self.agent_fitness[i] = self.objective_func(self.agents[i, 0], self.agents[i, 1])
                
            self.agents = schwefel_handler(
                "Gravitational", 
                self.agents, 
                self.bounds, 
                self.agent_fitness, 
                previous_fitness
            )
    
    def optimize(self):
        self.update_fitness()
        self.calculate_mass()
        
        if self.best_agent is None:
            best_idx = np.argmin(self.agent_fitness)
            self.best_agent = self.agents[best_idx].copy()
            self.best_fitness = self.agent_fitness[best_idx]
        
        self.history.append({
            'agents': self.agents.copy(),
            'velocities': self.velocities.copy(),
            'agent_mass': self.agent_mass.copy(),
            'best_agent': self.best_agent.copy(),
            'iteration': 0,
            'G': self.G,
            'best_fitness': self.best_fitness
        })
        
        for iteration in range(self.max_iter):
            self.update_gravitational_constant(iteration)
            forces = self.calculate_forces()
            self.update_velocities_and_positions(forces)
            self.update_fitness()
            self.calculate_mass()
            
            self.history.append({
                'agents': self.agents.copy(),
                'velocities': self.velocities.copy(),
                'agent_mass': self.agent_mass.copy(),
                'best_agent': self.best_agent.copy(),
                'iteration': iteration + 1,
                'G': self.G,
                'best_fitness': self.best_fitness
            })
        
        return self.best_agent, self.best_fitness, self.history

class SupernovaOptimization:
    def __init__(self, objective_func, n_stars=50, dimensions=2, bounds=(-5.12, 5.12), 
                 max_iter=100, explosion_interval=10, explosion_radius_init=3.0,
                 explosion_radius_final=0.5, local_search_radius=0.1):
        self.objective_func = objective_func
        self.n_stars = n_stars
        self.dimensions = dimensions
        self.bounds = bounds
        self.max_iter = max_iter
        self.explosion_interval = explosion_interval
        self.explosion_radius_init = explosion_radius_init
        self.explosion_radius_final = explosion_radius_final
        self.local_search_radius = local_search_radius
        
        self.stars = np.random.uniform(bounds[0], bounds[1], (n_stars, dimensions))
        self.star_fitness = np.zeros(n_stars)
        self.star_mass = np.ones(n_stars)
        self.star_luminosity = np.ones(n_stars)
        
        self.best_star = None
        self.best_fitness = float('inf')
        
        self.supernova_stage = False
        self.explosion_center = None
        self.explosion_particles = None
        self.explosion_particles_fitness = None
        self.explosion_radius = explosion_radius_init
        self.time_since_explosion = 0
        
        self.history = []
        
    def update_fitness(self):
        for i in range(self.n_stars):
            self.star_fitness[i] = self.objective_func(self.stars[i, 0], self.stars[i, 1])
            
            if self.star_fitness[i] < self.best_fitness:
                self.best_star = self.stars[i].copy()
                self.best_fitness = self.star_fitness[i]
    
    def update_star_properties(self):
        max_fitness = np.max(self.star_fitness)
        min_fitness = np.min(self.star_fitness)
        
        if max_fitness == min_fitness:
            normalized_fitness = np.ones(self.n_stars) * 0.5
        else:
            normalized_fitness = 1.0 - (self.star_fitness - min_fitness) / (max_fitness - min_fitness)
        
        self.star_mass = 1.0 + 4.0 * normalized_fitness
        
        self.star_luminosity = 0.2 + 0.8 * normalized_fitness
    
    def evolve_stars(self):
        for i in range(self.n_stars):
            local_step = np.random.uniform(-self.local_search_radius, self.local_search_radius, self.dimensions)
            
            gravity_direction = np.zeros(self.dimensions)
            for j in range(self.n_stars):
                if i != j:
                    direction = self.stars[j] - self.stars[i]
                    distance = np.linalg.norm(direction) + 1e-10
                    
                    force = self.star_mass[j] / (distance**2)
                    gravity_direction += force * direction / distance
            
            gravity_norm = np.linalg.norm(gravity_direction)
            if gravity_norm > 0:
                gravity_direction = gravity_direction / gravity_norm * 0.1
            
            self.stars[i] = self.stars[i] + local_step + gravity_direction
            
        self.stars = np.clip(self.stars, self.bounds[0], self.bounds[1])
    
    def trigger_supernova(self, iteration):
        self.supernova_stage = True
        self.explosion_center = self.best_star.copy()
        
        progress = iteration / self.max_iter
        self.explosion_radius = self.explosion_radius_init * (1 - progress) + self.explosion_radius_final * progress
        
        n_particles = self.n_stars * 2
        self.explosion_particles = np.zeros((n_particles, self.dimensions))
        self.explosion_particles_fitness = np.zeros(n_particles)
        
        for i in range(n_particles):
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            
            distance = self.explosion_radius * np.random.uniform(0.2, 1.0)**0.5
            
            self.explosion_particles[i] = self.explosion_center + distance * np.array([x, y])
        
        self.explosion_particles = np.clip(self.explosion_particles, self.bounds[0], self.bounds[1])
        
        for i in range(n_particles):
            self.explosion_particles_fitness[i] = self.objective_func(
                self.explosion_particles[i, 0], self.explosion_particles[i, 1]
            )
    
    def form_new_stars(self):
        combined_solutions = np.vstack((self.stars, self.explosion_particles))
        combined_fitness = np.concatenate((self.star_fitness, self.explosion_particles_fitness))
        
        indices = np.argsort(combined_fitness)[:self.n_stars]
        self.stars = combined_solutions[indices].copy()
        self.star_fitness = combined_fitness[indices].copy()
        
        best_idx = np.argmin(self.star_fitness)
        if self.star_fitness[best_idx] < self.best_fitness:
            self.best_star = self.stars[best_idx].copy()
            self.best_fitness = self.star_fitness[best_idx]
        
        self.supernova_stage = False
        self.explosion_particles = None
        self.explosion_particles_fitness = None
        self.time_since_explosion = 0
    
    def optimize(self):
        self.update_fitness()
        self.update_star_properties()
        
        self.history.append({
            'stars': self.stars.copy(),
            'star_fitness': self.star_fitness.copy(),
            'star_luminosity': self.star_luminosity.copy(),
            'best_star': self.best_star.copy(),
            'explosion_active': False,
            'explosion_center': None,
            'explosion_particles': None,
            'explosion_radius': 0,
            'iteration': 0,
            'best_fitness': self.best_fitness
        })
        
        for iteration in range(self.max_iter):
            if (iteration + 1) % self.explosion_interval == 0:
                self.trigger_supernova(iteration)
                
                self.history.append({
                    'stars': self.stars.copy(),
                    'star_fitness': self.star_fitness.copy(),
                    'star_luminosity': self.star_luminosity.copy(),
                    'best_star': self.best_star.copy(),
                    'explosion_active': True,
                    'explosion_center': self.explosion_center.copy(),
                    'explosion_particles': self.explosion_particles.copy() if self.explosion_particles is not None else None,
                    'explosion_radius': self.explosion_radius,
                    'iteration': iteration + 0.5,
                    'best_fitness': self.best_fitness
                })
                
                self.form_new_stars()
            else:
                self.evolve_stars()
                
                self.update_fitness()
                self.update_star_properties()
            
            self.history.append({
                'stars': self.stars.copy(),
                'star_fitness': self.star_fitness.copy(),
                'star_luminosity': self.star_luminosity.copy(),
                'best_star': self.best_star.copy(),
                'explosion_active': False,
                'explosion_center': None,
                'explosion_particles': None,
                'explosion_radius': 0,
                'iteration': iteration + 1,
                'best_fitness': self.best_fitness
            })
        
        return self.best_star, self.best_fitness, self.history

def visualize_blackhole(objective_func, history, bounds, save_animation=False, filename='black_hole_optimization.gif'):
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_func(X, Y)
    
    fig = plt.figure(figsize=(15, 7))
    
    ax1 = fig.add_subplot(121, projection='3d')
    surface = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Objective Function')
    
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, 20, cmap=cm.viridis)
    fig.colorbar(contour, ax=ax2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Black Hole Optimization')
    ax2.set_xlim(bounds)
    ax2.set_ylim(bounds)
    
    stars_plot, = ax2.plot([], [], 'o', color='cyan', alpha=0.6, label='Stars')
    black_hole_plot, = ax2.plot([], [], 'o', color='black', markersize=10, label='Black Hole')
    horizon_plot = plt.Circle((0, 0), 0, color='red', alpha=0.3, label='Event Horizon')
    ax2.add_patch(horizon_plot)
    ax2.legend()
    
    fitness_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    def update(frame):
        frame_data = history[frame]
        
        stars_plot.set_data(frame_data['stars'][:, 0], frame_data['stars'][:, 1])
        black_hole_plot.set_data([frame_data['black_hole'][0]], [frame_data['black_hole'][1]])
        
        horizon_plot.center = (frame_data['black_hole'][0], frame_data['black_hole'][1])
        horizon_plot.radius = frame_data['event_horizon']
        
        fitness_text.set_text(f'Iteration: {frame_data["iteration"] + 1}/{len(history)}\n'
                             f'Best fitness: {frame_data["best_fitness"]:.6f}\n'
                             f'Position: ({frame_data["black_hole"][0]:.2f}, {frame_data["black_hole"][1]:.2f})')
        
        return stars_plot, black_hole_plot, horizon_plot, fitness_text
    
    ani = FuncAnimation(fig, update, frames=len(history), blit=False, interval=200)
    
    if save_animation:
        print(f"Saving animation to {filename}...")
        ani.save(filename, writer='pillow', fps=10, dpi=150)
        print("Animation saved successfully!")
    
    plt.tight_layout()
    plt.show()
    
    return ani

def visualize_bigbang(objective_func, history, bounds, save_animation=False, filename='big_bang_big_crunch.gif'):
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_func(X, Y)
    
    fig = plt.figure(figsize=(15, 7))
    
    ax1 = fig.add_subplot(121, projection='3d')
    surface = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Objective Function')
    
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, 20, cmap=cm.viridis)
    fig.colorbar(contour, ax=ax2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Big Bang-Big Crunch Optimization')
    ax2.set_xlim(bounds)
    ax2.set_ylim(bounds)
    
    particles_plot, = ax2.plot([], [], 'o', color='orange', alpha=0.6, label='Particles')
    center_plot, = ax2.plot([], [], 'D', color='red', markersize=10, label='Center of Mass')
    explosion_circle = plt.Circle((0, 0), 0, color='orange', alpha=0.2, fill=False, linestyle='--')
    ax2.add_patch(explosion_circle)
    best_solution_plot, = ax2.plot([], [], '*', color='green', markersize=12, label='Best Solution')
    ax2.legend()
    
    info_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    def update(frame_idx):
        frame_data = history[frame_idx]
        
        particles_plot.set_data(frame_data['particles'][:, 0], frame_data['particles'][:, 1])
        
        if frame_data['center_of_mass'] is not None:
            center_plot.set_data([frame_data['center_of_mass'][0]], [frame_data['center_of_mass'][1]])
            center_plot.set_visible(True)
            
            explosion_circle.center = (frame_data['center_of_mass'][0], frame_data['center_of_mass'][1])
            explosion_circle.radius = frame_data['radius']
            explosion_circle.set_visible(True)
        else:
            center_plot.set_visible(False)
            explosion_circle.set_visible(False)
        
        if frame_data['best_solution'] is not None:
            best_solution_plot.set_data([frame_data['best_solution'][0]], [frame_data['best_solution'][1]])
            best_solution_plot.set_visible(True)
        else:
            best_solution_plot.set_visible(False)
        
        iteration_info = f"Iteration: {int(frame_data['iteration'])+1}/{int(len(history)/2)}"
        if frame_data['iteration'] % 1 == 0:
            phase = "Big Bang Phase"
        else:
            phase = "Big Crunch Phase"
        
        info_text.set_text(f'{iteration_info}\n{phase}\n'
                          f'Best fitness: {frame_data["best_fitness"]:.6f}\n'
                          f'Position: ({frame_data["best_solution"][0]:.2f}, {frame_data["best_solution"][1]:.2f})')
        
        return particles_plot, center_plot, explosion_circle, best_solution_plot, info_text
    
    ani = FuncAnimation(fig, update, frames=len(history), blit=False, interval=300)
    
    if save_animation:
        print(f"Saving animation to {filename}...")
        ani.save(filename, writer='pillow', fps=15, dpi=200, bitrate=2000)
        print("Animation saved successfully!")
    
    plt.tight_layout()
    plt.show()
    
    return ani

def visualize_solarsystem(objective_func, history, bounds, save_animation=False, filename='solar_system_optimization.gif'):
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_func(X, Y)
    
    fig = plt.figure(figsize=(15, 7))
    
    ax1 = fig.add_subplot(121, projection='3d')
    surface = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Objective Function')
    
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, 20, cmap=cm.viridis, alpha=0.5)
    fig.colorbar(contour, ax=ax2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Solar System Optimization')
    ax2.set_xlim(bounds)
    ax2.set_ylim(bounds)
    
    planets_plot, = ax2.plot([], [], 'o', color='blue', alpha=0.7, label='Planets')
    sun_plot, = ax2.plot([], [], '*', color='yellow', markersize=15, label='Sun (Best Solution)')
    
    orbit_ellipses = []
    for i in range(len(history[0]['planets'])):
        ellipse = Ellipse((0, 0), 0, 0, fill=False, color='gray', alpha=0.3)
        ax2.add_patch(ellipse)
        orbit_ellipses.append(ellipse)
    
    ax2.legend()
    
    info_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    def update(frame_idx):
        frame_data = history[frame_idx]
        
        planets_plot.set_data(frame_data['planets'][:, 0], frame_data['planets'][:, 1])
        sun_plot.set_data([frame_data['sun'][0]], [frame_data['sun'][1]])
        
        for i, ellipse in enumerate(orbit_ellipses):
            if i < len(frame_data['planets']):
                a = frame_data['semi_major_axes'][i]
                e = frame_data['eccentricities'][i]
                b = a * np.sqrt(1 - e**2)
                
                ellipse.center = (frame_data['sun'][0], frame_data['sun'][1])
                ellipse.width = 2 * a
                ellipse.height = 2 * b
                ellipse.set_visible(True)
            else:
                ellipse.set_visible(False)
        
        info_text.set_text(f"Iteration: {frame_data['iteration'] + 1}/{len(history)}\n"
                          f"Best fitness: {frame_data['sun_fitness']:.6f}\n"
                          f"Sun position: ({frame_data['sun'][0]:.2f}, {frame_data['sun'][1]:.2f})")
        
        return [planets_plot, sun_plot] + orbit_ellipses + [info_text]
    
    ani = FuncAnimation(fig, update, frames=len(history), blit=False, interval=200)
    
    if save_animation:
        print(f"Saving animation to {filename}...")
        ani.save(filename, writer='pillow', fps=12, dpi=150)
        print("Animation saved successfully!")
    
    plt.tight_layout()
    plt.show()
    
    return ani

def visualize_multiverse(objective_func, history, bounds, save_animation=False, filename='multiverse_optimization.gif'):
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_func(X, Y)
    
    fig = plt.figure(figsize=(15, 7))
    
    ax1 = fig.add_subplot(121, projection='3d')
    surface = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Objective Function')
    
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, 20, cmap=cm.viridis, alpha=0.5)
    fig.colorbar(contour, ax=ax2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Multiverse Optimization')
    ax2.set_xlim(bounds)
    ax2.set_ylim(bounds)
    
    universe_cmap = LinearSegmentedColormap.from_list("universe_colors", ['purple', 'magenta', 'orange', 'yellow'])
    
    universe_plot = ax2.scatter([], [], c=[], s=[], cmap=universe_cmap, vmin=0, vmax=1, alpha=0.7)
    
    best_universe_plot, = ax2.plot([], [], '*', color='white', markersize=15, 
                                  markeredgecolor='black', label='Best Universe')
    
    wormhole_lines = []
    for i in range(5):  
        line, = ax2.plot([], [], 'w--', alpha=0.3, linewidth=1)
        wormhole_lines.append(line)
    
    ax2.legend()
    
    info_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    def update(frame_idx):
        frame_data = history[frame_idx]
        
        if frame_data['universes'] is not None and len(frame_data['universes']) > 0:
            universe_plot.set_offsets(frame_data['universes'])
            
            universe_plot.set_array(frame_data['inflation_rates'])
            
            universe_sizes = 20 + 130 * frame_data['inflation_rates']
            universe_plot.set_sizes(universe_sizes)
        
        if frame_data['best_universe'] is not None:
            best_universe_plot.set_data([frame_data['best_universe'][0]], [frame_data['best_universe'][1]])
            best_universe_plot.set_visible(True)
        else:
            best_universe_plot.set_visible(False)
        
        n_wormholes = min(5, len(frame_data['universes']))
        wormhole_indices = np.random.choice(len(frame_data['universes']), n_wormholes, replace=False)
        
        for i, line in enumerate(wormhole_lines):
            if i < n_wormholes:
                idx = wormhole_indices[i]
                line.set_data([frame_data['universes'][idx, 0], frame_data['best_universe'][0]],
                             [frame_data['universes'][idx, 1], frame_data['best_universe'][1]])
                line.set_visible(np.random.random() < frame_data['wep'])
            else:
                line.set_visible(False)
        
        info_text.set_text(f"Iteration: {frame_data['iteration'] + 1}/{len(history)}\n"
                          f"Wormhole Probability: {frame_data['wep']:.3f}\n"
                          f"Travel Distance Rate: {frame_data['tdr']:.3f}\n"
                          f"Best fitness: {frame_data['best_fitness']:.6f}")
        
        return [universe_plot, best_universe_plot] + wormhole_lines + [info_text]
    
    ani = FuncAnimation(fig, update, frames=len(history), blit=False, interval=200)
    
    if save_animation:
        print(f"Saving animation to {filename}...")
        ani.save(filename, writer='pillow', fps=12, dpi=150)
        print("Animation saved successfully!")
    
    plt.tight_layout()
    plt.show()
    
    return ani

def visualize_multiverse(objective_func, history, bounds, save_animation=False, filename='multiverse_optimization.gif'):
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_func(X, Y)
    
    fig = plt.figure(figsize=(15, 7))
    
    ax1 = fig.add_subplot(121, projection='3d')
    surface = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Objective Function')
    
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, 20, cmap=cm.viridis, alpha=0.5)
    fig.colorbar(contour, ax=ax2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Multiverse Optimization')
    ax2.set_xlim(bounds)
    ax2.set_ylim(bounds)
    
    universe_cmap = LinearSegmentedColormap.from_list("universe_colors", ['purple', 'magenta', 'orange', 'yellow'])
    
    universe_plot = ax2.scatter([], [], c=[], s=[], cmap=universe_cmap, vmin=0, vmax=1, alpha=0.7)
    
    best_universe_plot, = ax2.plot([], [], '*', color='white', markersize=15, 
                                  markeredgecolor='black', label='Best Universe')
    
    wormhole_lines = []
    for i in range(5):  
        line, = ax2.plot([], [], 'w--', alpha=0.3, linewidth=1)
        wormhole_lines.append(line)
    
    ax2.legend()
    
    info_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    def update(frame_idx):
        frame_data = history[frame_idx]
        
        if frame_data['universes'] is not None and len(frame_data['universes']) > 0:
            universe_plot.set_offsets(frame_data['universes'])
            
            universe_plot.set_array(frame_data['inflation_rates'])
            
            universe_sizes = 20 + 130 * frame_data['inflation_rates']
            universe_plot.set_sizes(universe_sizes)
        
        if frame_data['best_universe'] is not None:
            best_universe_plot.set_data([frame_data['best_universe'][0]], [frame_data['best_universe'][1]])
            best_universe_plot.set_visible(True)
        else:
            best_universe_plot.set_visible(False)
        
        n_wormholes = min(5, len(frame_data['universes']))
        wormhole_indices = np.random.choice(len(frame_data['universes']), n_wormholes, replace=False)
        
        for i, line in enumerate(wormhole_lines):
            if i < n_wormholes:
                idx = wormhole_indices[i]
                line.set_data([frame_data['universes'][idx, 0], frame_data['best_universe'][0]],
                             [frame_data['universes'][idx, 1], frame_data['best_universe'][1]])
                line.set_visible(np.random.random() < frame_data['wep'])
            else:
                line.set_visible(False)
        
        info_text.set_text(f"Iteration: {frame_data['iteration'] + 1}/{len(history)}\n"
                          f"Wormhole Probability: {frame_data['wep']:.3f}\n"
                          f"Travel Distance Rate: {frame_data['tdr']:.3f}\n"
                          f"Best fitness: {frame_data['best_fitness']:.6f}")
        
        return [universe_plot, best_universe_plot] + wormhole_lines + [info_text]
    
    ani = FuncAnimation(fig, update, frames=len(history), blit=False, interval=200)
    
    if save_animation:
        print(f"Saving animation to {filename}...")
        ani.save(filename, writer='pillow', fps=12, dpi=150)
        print("Animation saved successfully!")
    
    plt.tight_layout()
    plt.show()
    
    return ani

def visualize_galaxy(objective_func, history, bounds, save_animation=False, filename='galaxy_optimization.gif'):
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_func(X, Y)
    
    fig = plt.figure(figsize=(15, 7))
    
    ax1 = fig.add_subplot(121, projection='3d')
    surface = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Objective Function')
    
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, 20, cmap=cm.viridis, alpha=0.5)
    fig.colorbar(contour, ax=ax2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Galaxy-Based Search Optimization')
    ax2.set_xlim(bounds)
    ax2.set_ylim(bounds)
    
    arm_colors = ['cyan', 'magenta', 'yellow', 'lightgreen', 'orange']
    stars_plots = []
    for i in range(max(history[0]['star_arm']) + 1):
        stars_plot, = ax2.plot([], [], 'o', color=arm_colors[i % len(arm_colors)], alpha=0.6, 
                              label=f'Arm {i+1}', markersize=5)
        stars_plots.append(stars_plot)
    
    black_hole_plot, = ax2.plot([], [], 'o', color='black', markersize=10, 
                              label='Galaxy Center (Best Solution)')
    
    ax2.legend()
    
    info_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    def update(frame_idx):
        frame_data = history[frame_idx]
        
        for arm_idx, stars_plot in enumerate(stars_plots):
            arm_mask = frame_data['star_arm'] == arm_idx
            if np.any(arm_mask):
                arm_stars = frame_data['stars'][arm_mask]
                stars_plot.set_data(arm_stars[:, 0], arm_stars[:, 1])
                stars_plot.set_visible(True)
            else:
                stars_plot.set_visible(False)
        
        if frame_data['black_hole'] is not None:
            black_hole_plot.set_data([frame_data['black_hole'][0]], [frame_data['black_hole'][1]])
            black_hole_plot.set_visible(True)
        else:
            black_hole_plot.set_visible(False)
        
        info_text.set_text(f"Iteration: {frame_data['iteration'] + 1}/{len(history)}\n"
                          f"Best fitness: {frame_data['best_fitness']:.6f}\n"
                          f"Position: ({frame_data['black_hole'][0]:.2f}, {frame_data['black_hole'][1]:.2f})")
        
        return stars_plots + [black_hole_plot, info_text]
    
    ani = FuncAnimation(fig, update, frames=len(history), blit=False, interval=200)
    
    if save_animation:
        print(f"Saving animation to {filename}...")
        ani.save(filename, writer='pillow', fps=12, dpi=150)
        print("Animation saved successfully!")
    
    plt.tight_layout()
    plt.show()
    
    return ani

def visualize_gravitational(objective_func, history, bounds, save_animation=False, filename='gravitational_optimization.gif'):
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_func(X, Y)
    
    fig = plt.figure(figsize=(15, 7))
    
    ax1 = fig.add_subplot(121, projection='3d')
    surface = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Objective Function')
    
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, 20, cmap=cm.viridis, alpha=0.5)
    fig.colorbar(contour, ax=ax2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Gravitational Search Optimization')
    ax2.set_xlim(bounds)
    ax2.set_ylim(bounds)
    
    blue_cmap = LinearSegmentedColormap.from_list("blue_gradient", ["lightblue", "darkblue"])
    agents_plot = ax2.scatter([], [], c=[], cmap=blue_cmap, s=[], alpha=0.7)
    
    best_agent_plot, = ax2.plot([], [], '*', color='red', markersize=15, label='Best Agent')
    
    ax2.legend()
    
    info_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    def update(frame_idx):
        frame_data = history[frame_idx]
        
        if frame_data['agents'] is not None and len(frame_data['agents']) > 0:
            agents_plot.set_offsets(frame_data['agents'])
            
            mass_norm = frame_data['agent_mass'] / np.max(frame_data['agent_mass'])
            agents_plot.set_array(mass_norm)
            
            mass_sizes = 20 + 100 * mass_norm
            agents_plot.set_sizes(mass_sizes)
        
        if frame_data['best_agent'] is not None:
            best_agent_plot.set_data([frame_data['best_agent'][0]], [frame_data['best_agent'][1]])
            best_agent_plot.set_visible(True)
        else:
            best_agent_plot.set_visible(False)
        
        info_text.set_text(f"Iteration: {frame_data['iteration'] + 1}/{len(history)}\n"
                          f"Gravitational Constant: {frame_data['G']:.3f}\n"
                          f"Best fitness: {frame_data['best_fitness']:.6f}\n"
                          f"Position: ({frame_data['best_agent'][0] if frame_data['best_agent'] is not None else 0:.2f}, "
                          f"{frame_data['best_agent'][1] if frame_data['best_agent'] is not None else 0:.2f})")
        
        return agents_plot, best_agent_plot, info_text
    
    ani = FuncAnimation(fig, update, frames=len(history), blit=False, interval=200)
    
    if save_animation:
        print(f"Saving animation to {filename}...")
        ani.save(filename, writer='pillow', fps=12, dpi=150)
        print("Animation saved successfully!")
    
    plt.tight_layout()
    plt.show()
    
    return ani

def visualize_supernova(objective_func, history, bounds, save_animation=False, filename='supernova_optimization.gif'):
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_func(X, Y)
    
    fig = plt.figure(figsize=(15, 7))
    
    ax1 = fig.add_subplot(121, projection='3d')
    surface = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Objective Function')
    
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, 20, cmap=cm.viridis, alpha=0.5)
    fig.colorbar(contour, ax=ax2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Supernova Optimization')
    ax2.set_xlim(bounds)
    ax2.set_ylim(bounds)
    
    star_cmap = LinearSegmentedColormap.from_list("star_colors", ['darkblue', 'blue', 'cyan', 'yellow'])
    
    stars_plot = ax2.scatter([], [], c=[], s=[], cmap=star_cmap, vmin=0, vmax=1, alpha=0.8)
    best_star_plot, = ax2.plot([], [], '*', color='gold', markersize=15, label='Best Star')
    explosion_particles_plot = ax2.scatter([], [], c='red', s=10, alpha=0.5)
    explosion_circle = plt.Circle((0, 0), 0, color='orange', alpha=0.2, fill=False, linestyle='--')
    ax2.add_patch(explosion_circle)
    
    ax2.legend()
    
    info_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    def update(frame_idx):
        frame_data = history[frame_idx]
        
        if frame_data['stars'] is not None and len(frame_data['stars']) > 0:
            stars_plot.set_offsets(frame_data['stars'])
            stars_plot.set_array(np.array(frame_data['star_luminosity']))
            star_sizes = 20 + 80 * np.array(frame_data['star_luminosity'])
            stars_plot.set_sizes(star_sizes)
            stars_plot.set_visible(True)
        else:
            stars_plot.set_visible(False)
        
        if frame_data['best_star'] is not None:
            best_star_plot.set_data([frame_data['best_star'][0]], [frame_data['best_star'][1]])
            best_star_plot.set_visible(True)
        else:
            best_star_plot.set_visible(False)
        
        if frame_data['explosion_active'] and frame_data['explosion_particles'] is not None:
            explosion_particles_plot.set_offsets(frame_data['explosion_particles'])
            explosion_particles_plot.set_visible(True)
            
            explosion_circle.center = (frame_data['explosion_center'][0], frame_data['explosion_center'][1])
            explosion_circle.radius = frame_data['explosion_radius']
            explosion_circle.set_visible(True)
        else:
            explosion_particles_plot.set_visible(False)
            explosion_circle.set_visible(False)
        
        if frame_data['explosion_active']:
            phase = "SUPERNOVA EXPLOSION"
            iter_num = int(frame_data['iteration'])
        else:
            phase = "Star Evolution"
            iter_num = int(frame_data['iteration'])
        
        info_text.set_text(f"Iteration: {iter_num + 1}/{int(history[-1]['iteration']) + 1}\n"
                          f"Phase: {phase}\n"
                          f"Best fitness: {frame_data['best_fitness']:.6f}\n"
                          f"Position: ({frame_data['best_star'][0]:.2f}, {frame_data['best_star'][1]:.2f})")
        
        return stars_plot, best_star_plot, explosion_particles_plot, explosion_circle, info_text
    
    ani = FuncAnimation(fig, update, frames=len(history), blit=False, interval=200)
    
    if save_animation:
        print(f"Saving animation to {filename}...")
        ani.save(filename, writer='pillow', fps=5)
        print("Animation saved successfully!")
    
    plt.tight_layout()
    plt.show()
    
    return ani

def blackhole(objective_func, n_stars=50, bounds=(-5.0, 5.0), max_iter=50, 
              save_animation=True, filename='black_hole_optimization.gif'):
    print(f"Running Black Hole Optimization with {n_stars} stars for {max_iter} iterations...")
    
    bho = BlackHoleOptimization(
        objective_func=objective_func,
        n_stars=n_stars,
        bounds=bounds,
        max_iter=max_iter
    )
    
    best_solution, best_fitness, history = bho.optimize()
    
    print(f"Optimization completed!")
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    
    print(f"Creating visualization...")
    ani = visualize_blackhole(
        objective_func=objective_func,
        history=history,
        bounds=bounds,
        save_animation=save_animation,
        filename=filename
    )
    
    return best_solution, best_fitness

def bigbang(objective_func, n_particles=50, bounds=(-5.0, 5.0), max_iter=20, 
            init_radius=2.0, final_radius=0.1, save_animation=True, 
            filename='big_bang_big_crunch.gif'):
    print(f"Running Big Bang-Big Crunch with {n_particles} particles for {max_iter} iterations...")
    
    bbbc = BigBangBigCrunch(
        objective_func=objective_func,
        n_particles=n_particles,
        bounds=bounds,
        max_iter=max_iter,
        initial_explosion_radius=init_radius,
        final_explosion_radius=final_radius
    )
    
    best_solution, best_fitness, history = bbbc.optimize()
    
    print(f"Optimization completed!")
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    
    print(f"Creating visualization...")
    ani = visualize_bigbang(
        objective_func=objective_func,
        history=history,
        bounds=bounds,
        save_animation=save_animation,
        filename=filename
    )
    
    return best_solution, best_fitness

def solarsystem(objective_func, n_planets=10, bounds=(-5.0, 5.0), max_iter=50, 
                ecc_range=(0.1, 0.7), orbit_range=(3, 10),
                save_animation=True, filename='solar_system_optimization.gif'):
    print(f"Running Solar System Optimization with {n_planets} planets for {max_iter} iterations...")
    
    ssbo = SolarSystemOptimization(
        objective_func=objective_func,
        n_planets=n_planets,
        bounds=bounds,
        max_iter=max_iter,
        eccentricity_range=ecc_range,
        orbit_period_range=orbit_range
    )
    
    best_solution, best_fitness, history = ssbo.optimize()
    
    print(f"Optimization completed!")
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    
    print(f"Creating visualization...")
    ani = visualize_solarsystem(
        objective_func=objective_func,
        history=history,
        bounds=bounds,
        save_animation=save_animation,
        filename=filename
    )
    
    return best_solution, best_fitness

def multiverse(objective_func, n_universes=50, bounds=(-5.0, 5.0), max_iter=50,
              wormhole_prob=0.6, tdr_min=0.2, tdr_max=0.8,
              save_animation=True, filename='multiverse_optimization.gif'):
    print(f"Running Multiverse Optimizer with {n_universes} universes for {max_iter} iterations...")
    print(f"Wormhole existence probability: {wormhole_prob}")
    
    mvo = MultiverseOptimizer(
        objective_func=objective_func,
        n_universes=n_universes,
        bounds=bounds,
        max_iter=max_iter,
        wormhole_existence_prob=wormhole_prob,
        travelling_distance_rate_min=tdr_min,
        travelling_distance_rate_max=tdr_max
    )
    
    best_solution, best_fitness, history = mvo.optimize()
    
    print(f"Optimization completed!")
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    
    print(f"Creating visualization...")
    ani = visualize_multiverse(
        objective_func=objective_func,
        history=history,
        bounds=bounds,
        save_animation=save_animation,
        filename=filename
    )
    
    return best_solution, best_fitness

def multiverse(objective_func, n_universes=50, bounds=(-5.0, 5.0), max_iter=50,
              wormhole_prob=0.6, tdr_min=0.2, tdr_max=0.8,
              save_animation=True, filename='multiverse_optimization.gif'):
    print(f"Running Multiverse Optimizer with {n_universes} universes for {max_iter} iterations...")
    print(f"Wormhole existence probability: {wormhole_prob}")
    
    mvo = MultiverseOptimizer(
        objective_func=objective_func,
        n_universes=n_universes,
        bounds=bounds,
        max_iter=max_iter,
        wormhole_existence_prob=wormhole_prob,
        travelling_distance_rate_min=tdr_min,
        travelling_distance_rate_max=tdr_max
    )
    
    best_solution, best_fitness, history = mvo.optimize()
    
    print(f"Optimization completed!")
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    
    print(f"Creating visualization...")
    ani = visualize_multiverse(
        objective_func=objective_func,
        history=history,
        bounds=bounds,
        save_animation=save_animation,
        filename=filename
    )
    
    return best_solution, best_fitness

def galaxy(objective_func, n_stars=50, bounds=(-5.0, 5.0), max_iter=50,
          spiral_constant=0.7, rotation_angle=0.01, arm_count=3,
          save_animation=True, filename='galaxy_optimization.gif'):
    print(f"Running Galaxy-Based Search with {n_stars} stars for {max_iter} iterations...")
    print(f"Using {arm_count} spiral arms with spiral constant {spiral_constant}")
    
    gbsa = GalaxyBasedSearch(
        objective_func=objective_func,
        n_stars=n_stars,
        bounds=bounds,
        max_iter=max_iter,
        spiral_constant=spiral_constant,
        rotation_angle=rotation_angle,
        arm_count=arm_count
    )
    
    best_solution, best_fitness, history = gbsa.optimize()
    
    print(f"Optimization completed!")
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    
    print(f"Creating visualization...")
    ani = visualize_galaxy(
        objective_func=objective_func,
        history=history,
        bounds=bounds,
        save_animation=save_animation,
        filename=filename
    )
    
    return best_solution, best_fitness

def gravitational(objective_func, n_agents=50, bounds=(-5.0, 5.0), max_iter=50,
                G_init=100, G_reduction=0.8, best_agent_ratio=0.05,
                save_animation=True, filename='gravitational_optimization.gif'):
    print(f"Running Gravitational Search with {n_agents} agents for {max_iter} iterations...")
    print(f"Initial gravitational constant: {G_init}")
    
    gsa = GravitationalSearch(
        objective_func=objective_func,
        n_agents=n_agents,
        bounds=bounds,
        max_iter=max_iter,
        G_init=G_init,
        G_reduction=G_reduction,
        best_agent_ratio=best_agent_ratio
    )
    
    best_solution, best_fitness, history = gsa.optimize()
    
    print(f"Optimization completed!")
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    
    print(f"Creating visualization...")
    ani = visualize_gravitational(
        objective_func=objective_func,
        history=history,
        bounds=bounds,
        save_animation=save_animation,
        filename=filename
    )
    
    return best_solution, best_fitness

def supernova(objective_func, n_stars=50, bounds=(-5.0, 5.0), max_iter=50, 
              explosion_interval=10, explosion_radius_init=3.0, 
              explosion_radius_final=0.5,
              save_animation=True, filename='supernova_optimization.gif'):
    print(f"Running Supernova Optimization with {n_stars} stars for {max_iter} iterations...")
    print(f"Supernova explosions will occur every {explosion_interval} iterations")
    
    sno = SupernovaOptimization(
        objective_func=objective_func,
        n_stars=n_stars,
        bounds=bounds,
        max_iter=max_iter,
        explosion_interval=explosion_interval,
        explosion_radius_init=explosion_radius_init,
        explosion_radius_final=explosion_radius_final
    )
    
    best_solution, best_fitness, history = sno.optimize()
    
    print(f"Optimization completed!")
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    
    print(f"Creating visualization...")
    ani = visualize_supernova(
        objective_func=objective_func,
        history=history,
        bounds=bounds,
        save_animation=save_animation,
        filename=filename
    )
    
    return best_solution, best_fitness

def schwefel_handler(algorithm_name, particles, bounds, fitness_values=None, previous_fitness=None, replacement_rate=0.3):

    n_particles = particles.shape[0]
    n_replace = max(1, int(n_particles * replacement_rate))
    
    if fitness_values is not None and previous_fitness is not None:
        not_improved = fitness_values >= previous_fitness
        
        if sum(not_improved) > n_replace:
            indices = np.argsort(fitness_values)[::-1]  
            replace_indices = indices[:n_replace]
        else:
            replace_indices = np.where(not_improved)[0]
    else:
        replace_indices = np.random.choice(n_particles, n_replace, replace=False)
    
    for idx in replace_indices:
        if np.random.random() < 0.5:  
            particles[idx] = np.array([420.9687, 420.9687]) + np.random.uniform(-50, 50, 2)
        else:
            particles[idx] = np.random.uniform(bounds[0], bounds[1], 2)
    
    return particles

def main():
    np.random.seed(42)
    
    algorithms = {
        1: ("Black Hole Optimization", blackhole),
        2: ("Big Bang-Big Crunch", bigbang),
        3: ("Solar System Optimization", solarsystem),
        4: ("Supernova Optimization", supernova),
        5: ("Galaxy-Based Search", galaxy),
        6: ("Gravitational Search Algorithm", gravitational),
        7: ("Multiverse Optimizer", multiverse)
    }
    
    objective_functions = {
        1: ("Rastrigin", rastrigin, (-5.0, 5.0)),
        2: ("Rosenbrock", rosenbrock, (-2.0, 2.0)),
        3: ("Himmelblau", himmelblau, (-5.0, 5.0)),
        4: ("Ackley", ackley, (-5.0, 5.0)),
        5: ("Sphere", sphere, (-5.0, 5.0)),
        6: ("Zakharov", zakharov, (-5.0, 5.0)),
        7: ("Griewank", griewank, (-600.0, 600.0)),
        8: ("Schwefel", schwefel, (-500.0, 500.0)),
        9: ("Michalewicz", michalewicz, (0.0, np.pi)),
        10: ("Easom", easom, (-100.0, 100.0))
    }
    
    print("\n===== SPACE-INSPIRED OPTIMIZATION ALGORITHMS =====")
    print("\nAvailable algorithms:")
    for key, (name, _) in algorithms.items():
        print(f"{key}. {name}")
    
    try:
        algo_choice = int(input("\nSelect an algorithm (1-7): "))
        if algo_choice not in algorithms:
            algo_choice = 1
            print("Invalid selection. Using Black Hole Optimization as default.")
    except ValueError:
        algo_choice = 1
        print("Invalid input. Using Black Hole Optimization as default.")
    
    algo_name, algo_func = algorithms[algo_choice]
    
    print("\nAvailable objective functions:")
    for key, (name, _, _) in objective_functions.items():
        print(f"{key}. {name}")
    
    try:
        func_choice = int(input("\nSelect an objective function (1-10): "))
        if func_choice not in objective_functions:
            func_choice = 1
            print("Invalid selection. Using Rastrigin function as default.")
    except ValueError:
        func_choice = 1
        print("Invalid input. Using Rastrigin function as default.")
    
    func_name, objective_func, bounds = objective_functions[func_choice]
    
    try:
        max_iter = int(input("\nEnter maximum iterations (default 50): ") or "50")
        if max_iter <= 0:
            max_iter = 50
            print("Invalid value. Using default of 50 iterations.")
    except ValueError:
        max_iter = 50
        print("Invalid input. Using default of 50 iterations.")
    
    if algo_choice == 1:  
        try:
            n_stars = int(input("Enter number of stars (default 50): ") or "50")
            if n_stars <= 0:
                n_stars = 50
                print("Invalid value. Using default of 50 stars.")
        except ValueError:
            n_stars = 50
            print("Invalid input. Using default of 50 stars.")
        
        filename = f"black_hole_{func_name.lower()}.gif"
        
        best_solution, best_fitness = blackhole(
            objective_func=objective_func,
            n_stars=n_stars,
            bounds=bounds,
            max_iter=max_iter,
            save_animation=True,
            filename=filename
        )
    
    elif algo_choice == 2:  
        try:
            n_particles = int(input("Enter number of particles (default 50): ") or "50")
            if n_particles <= 0:
                n_particles = 50
                print("Invalid value. Using default of 50 particles.")
        except ValueError:
            n_particles = 50
            print("Invalid input. Using default of 50 particles.")
        
        try:
            init_radius = float(input("Enter initial explosion radius (default 2.0): ") or "2.0")
            final_radius = float(input("Enter final explosion radius (default 0.1): ") or "0.1")
        except ValueError:
            init_radius = 2.0
            final_radius = 0.1
            print("Invalid input. Using default radius values.")
        
        filename = f"big_bang_big_crunch_{func_name.lower()}.gif"
        
        best_solution, best_fitness = bigbang(
            objective_func=objective_func,
            n_particles=n_particles,
            bounds=bounds,
            max_iter=max_iter,
            init_radius=init_radius,
            final_radius=final_radius,
            save_animation=True,
            filename=filename
        )
    
    elif algo_choice == 3:  
        try:
            n_planets = int(input("Enter number of planets (default 10): ") or "10")
            if n_planets <= 0:
                n_planets = 10
                print("Invalid value. Using default of 10 planets.")
        except ValueError:
            n_planets = 10
            print("Invalid input. Using default of 10 planets.")
        
        filename = f"solar_system_{func_name.lower()}.gif"
        
        best_solution, best_fitness = solarsystem(
            objective_func=objective_func,
            n_planets=n_planets,
            bounds=bounds,
            max_iter=max_iter,
            save_animation=True,
            filename=filename
        )
    
    elif algo_choice == 4:  
        try:
            n_stars = int(input("Enter number of stars (default 50): ") or "50")
            if n_stars <= 0:
                n_stars = 50
                print("Invalid value. Using default of 50 stars.")
        except ValueError:
            n_stars = 50
            print("Invalid input. Using default of 50 stars.")
        
        try:
            explosion_interval = int(input("Enter explosion interval (default 10): ") or "10")
            if explosion_interval <= 0:
                explosion_interval = 10
                print("Invalid value. Using default interval of 10.")
        except ValueError:
            explosion_interval = 10
            print("Invalid input. Using default interval of 10.")
        
        filename = f"supernova_{func_name.lower()}.gif"
        
        best_solution, best_fitness = supernova(
            objective_func=objective_func,
            n_stars=n_stars,
            bounds=bounds,
            max_iter=max_iter,
            explosion_interval=explosion_interval,
            save_animation=True,
            filename=filename
        )
    
    elif algo_choice == 5: 
        try:
            n_stars = int(input("Enter number of stars (default 50): ") or "50")
            if n_stars <= 0:
                n_stars = 50
                print("Invalid value. Using default of 50 stars.")
        except ValueError:
            n_stars = 50
            print("Invalid input. Using default of 50 stars.")
        
        try:
            arm_count = int(input("Enter number of spiral arms (default 3): ") or "3")
            if arm_count <= 0:
                arm_count = 3
                print("Invalid value. Using default of 3 spiral arms.")
        except ValueError:
            arm_count = 3
            print("Invalid input. Using default of 3 spiral arms.")
        
        try:
            spiral_constant = float(input("Enter spiral constant (default 0.7): ") or "0.7")
            if spiral_constant <= 0:
                spiral_constant = 0.7
                print("Invalid value. Using default spiral constant of 0.7.")
        except ValueError:
            spiral_constant = 0.7
            print("Invalid input. Using default spiral constant of 0.7.")
        
        filename = f"galaxy_{func_name.lower()}.gif"
        
        best_solution, best_fitness = galaxy(
            objective_func=objective_func,
            n_stars=n_stars,
            bounds=bounds,
            max_iter=max_iter,
            spiral_constant=spiral_constant,
            arm_count=arm_count,
            save_animation=True,
            filename=filename
        )
    
    elif algo_choice == 6:  
        try:
            n_agents = int(input("Enter number of agents (default 50): ") or "50")
            if n_agents <= 0:
                n_agents = 50
                print("Invalid value. Using default of 50 agents.")
        except ValueError:
            n_agents = 50
            print("Invalid input. Using default of 50 agents.")
        
        try:
            G_init = float(input("Enter initial gravitational constant (default 100): ") or "100")
            if G_init <= 0:
                G_init = 100
                print("Invalid value. Using default G_init of 100.")
        except ValueError:
            G_init = 100
            print("Invalid input. Using default G_init of 100.")
        
        try:
            best_agent_ratio = float(input("Enter best agent ratio (default 0.05): ") or "0.05")
            if best_agent_ratio <= 0 or best_agent_ratio > 1:
                best_agent_ratio = 0.05
                print("Invalid value. Using default ratio of 0.05.")
        except ValueError:
            best_agent_ratio = 0.05
            print("Invalid input. Using default ratio of 0.05.")
        
        filename = f"gravitational_{func_name.lower()}.gif"
        
        best_solution, best_fitness = gravitational(
            objective_func=objective_func,
            n_agents=n_agents,
            bounds=bounds,
            max_iter=max_iter,
            G_init=G_init,
            best_agent_ratio=best_agent_ratio,
            save_animation=True,
            filename=filename
        )
    
    elif algo_choice == 7: 
        try:
            n_universes = int(input("Enter number of universes (default 50): ") or "50")
            if n_universes <= 0:
                n_universes = 50
                print("Invalid value. Using default of 50 universes.")
        except ValueError:
            n_universes = 50
            print("Invalid input. Using default of 50 universes.")
        
        try:
            wormhole_prob = float(input("Enter wormhole existence probability (default 0.6): ") or "0.6")
            if wormhole_prob < 0 or wormhole_prob > 1:
                wormhole_prob = 0.6
                print("Invalid value. Using default probability of 0.6.")
        except ValueError:
            wormhole_prob = 0.6
            print("Invalid input. Using default probability of 0.6.")
        
        filename = f"multiverse_{func_name.lower()}.gif"
        
        best_solution, best_fitness = multiverse(
            objective_func=objective_func,
            n_universes=n_universes,
            bounds=bounds,
            max_iter=max_iter,
            wormhole_prob=wormhole_prob,
            save_animation=True,
            filename=filename
        )
    
    elif algo_choice == 7:  
        try:
            n_universes = int(input("Enter number of universes (default 50): ") or "50")
            if n_universes <= 0:
                n_universes = 50
                print("Invalid value. Using default of 50 universes.")
        except ValueError:
            n_universes = 50
            print("Invalid input. Using default of 50 universes.")
        
        try:
            wormhole_prob = float(input("Enter wormhole existence probability (default 0.6): ") or "0.6")
            if wormhole_prob < 0 or wormhole_prob > 1:
                wormhole_prob = 0.6
                print("Invalid value. Using default probability of 0.6.")
        except ValueError:
            wormhole_prob = 0.6
            print("Invalid input. Using default probability of 0.6.")
        
        filename = f"multiverse_{func_name.lower()}.gif"
        
        best_solution, best_fitness = multiverse(
            objective_func=objective_func,
            n_universes=n_universes,
            bounds=bounds,
            max_iter=max_iter,
            wormhole_prob=wormhole_prob,
            save_animation=True,
            filename=filename
        )
    
    print("\n===== OPTIMIZATION RESULTS =====")
    print(f"Algorithm: {algo_name}")
    print(f"Objective Function: {func_name}")
    print(f"Best Solution: {best_solution}")
    print(f"Best Fitness: {best_fitness}")
    print(f"Animation saved as: {filename}")

if __name__ == "__main__":
    main()
