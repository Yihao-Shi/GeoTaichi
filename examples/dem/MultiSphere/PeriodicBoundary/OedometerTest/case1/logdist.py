import numpy as np
from scipy.stats import lognorm
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def write_sphere_text(N, xlim, ylim, zlim, D50, sigma_ln, dmin, dmax, vis=True):
    print('#', "Writing sphere(s) into 'SpherePacking' ......")
    print(f"Inserted Sphere Number: {N}")
    diameter = generate_lognormal_particle_sizes(N, D50*1e6, sigma_ln*1e6, dmin*1e6, dmax*1e6)
    position, radius = generate_particle_positions_3d(diameter, xlim, ylim, zlim)
    np.savetxt('SpherePacking.txt', np.column_stack((position, radius)), header="     PositionX            PositionY                PositionZ            Radius", delimiter=" ")
    if vis:
        plt.hist(diameter, bins=100, density=True, alpha=0.6, color='steelblue')
        plt.xlabel("Diameter (µm)")
        plt.ylabel("Probability Density")
        plt.title("Log-normal Particle Size Distribution")
        plt.grid(True)
        plt.show()

def generate_particle_positions_3d(diameter, x_range, y_range, z_range, max_attempts=1000):
    N = len(diameter)
    radii = 0.5 * diameter[::-1]
    positions = np.zeros((N, 3))
    placed = 0
    attempts = 0
    tree = cKDTree(np.empty((0, 3)))

    while placed < N and attempts < max_attempts * N:
        attempts += 1
        pos = np.array([np.random.uniform(*x_range), np.random.uniform(*y_range), np.random.uniform(*z_range)])
        ri = radii[placed]

        if placed > 0:
            neighbors = tree.query_ball_point(pos, r=ri + np.max(radii))
            if neighbors:
                conflict = False
                for j in neighbors:
                    rj = radii[j]
                    dist = np.linalg.norm(pos - positions[j])
                    if dist < 0.9*(ri+rj):
                        conflict = True
                        break
                if conflict:
                    continue

        positions[placed] = pos
        placed += 1
        tree = cKDTree(positions[:placed]) 

    if placed < N:
        print(f"Warning: Only placed {placed} of {N} particles.")
        return positions[:placed], radii[:placed]
    return positions[::-1, :], radii[::-1]

def generate_lognormal_particle_sizes(N, mean_linear, std_linear, dmin, dmax):
    oversample = int(N * 1.5)
    diameters = lognorm.rvs(s=std_linear, scale=mean_linear, size=oversample)
    valid = diameters[(diameters >= dmin) & (diameters <= dmax)]
    while len(valid) < N:
        extra = lognorm.rvs(s=std_linear, scale=mean_linear, size=(N - len(valid)) * 2)
        valid = np.concatenate([valid, extra[(extra >= dmin) & (extra <= dmax)]])
    return 1e-6*np.sort(valid[:N])

if __name__ == '__main__':
    N = 13000
    D50 = 18e-6      # µm
    sigma_ln = 0.45e-6 # µm
    dmin = 1e-6      # µm
    dmax = 70e-6     # µm
    x_range = (6e-5, 2.9e-4)
    y_range = (6e-5, 2.9e-4)
    z_range = (6e-5, 35.6e-4)
    write_sphere_text(N, x_range, y_range, z_range, D50, sigma_ln, dmin, dmax)


