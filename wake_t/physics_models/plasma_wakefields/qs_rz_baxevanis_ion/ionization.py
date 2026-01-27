from wake_t.utilities.numba import njit_serial
import numpy as np

@njit_serial
def Ionization(species_list):
    for species in species_list:
        if species.is_ion:
            ion_species = species
        else:
            electron_species = species

    idx_ion = np.random.randint(0, ion_species.num_particles, 2)
    new_size = electron_species.num_particles + idx_ion.shape[0]
    electron_species.r = np.resize(electron_species.r, new_size)
    electron_species.dr_p = np.resize(electron_species.dr_p, new_size)
    electron_species.pr = np.resize(electron_species.pr, new_size)
    electron_species.pz = np.resize(electron_species.pz, new_size)
    electron_species.gamma = np.resize(electron_species.gamma, new_size)
    electron_species.w = np.resize(electron_species.w, new_size)
    electron_species.w_center = np.resize(electron_species.w_center, new_size)
    
    for idx, value in enumerate(idx_ion):
        electron_species.r[idx] = ion_species.r[value]
        electron_species.dr_p[idx] = ion_species.dr_p[value]
        electron_species.pr[idx] = ion_species.pr[value]
        electron_species.pz[idx] = ion_species.pz[value]
        electron_species.gamma[idx] = ion_species.gamma[value]
        electron_species.w[idx] = ion_species.w[value]
        electron_species.w_center[idx] = ion_species.w_center[value]

