import numpy as np

p = 2
def get_data(path, name):
    particle = np.load(path+'particles/MPMParticle{0:06d}.npz'.format(1), allow_pickle=True)
    v = particle['velocity']
    state_vars = particle['state_vars'].item()
    stress = particle['stress']
    print(f'File name: {name}')
    print('kinetic energy:', np.sum(v[:,0]**2+v[:,1]**2+v[:,2]**2))
    print(f'stress at particle{p}:', stress[p])
    

get_data('ImpactForce45/', '1e-6')
get_data('ImpactForce145/', '1e-4')
