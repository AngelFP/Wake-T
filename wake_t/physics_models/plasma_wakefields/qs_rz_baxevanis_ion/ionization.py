from wake_t.utilities.numba import njit_serial
import numpy as np


@njit_serial
def Ionization(species_list, step):
    for species in species_list:
        if species.is_ion:
            ion_species = species
        else:
            electron_species = species

    idx_ion = np.random.randint(0, ion_species.num_particles, 1)
    new_size = step + 1
    electron_species.r = np.resize(electron_species.r, new_size)
    electron_species.dr_p = np.resize(electron_species.dr_p, new_size)
    electron_species.pr = np.resize(electron_species.pr, new_size)
    electron_species.pz = np.resize(electron_species.pz, new_size)
    electron_species.gamma = np.resize(electron_species.gamma, new_size)
    electron_species.w = np.resize(electron_species.w, new_size)
    electron_species.w_center = np.resize(electron_species.w_center, new_size)
    electron_species.id = np.resize(electron_species.id, new_size)
    electron_species.r_to_x = np.resize(electron_species.r_to_x, new_size)
    electron_species.num_particles = electron_species.r.shape[0]

    for idx, value in enumerate(idx_ion):
        electron_species.r[step] = ion_species.r[value]
        electron_species.dr_p[step] = ion_species.dr_p[value]
        electron_species.pr[step] = ion_species.pr[value]
        electron_species.pz[step] = ion_species.pz[value]
        electron_species.gamma[step] = ion_species.gamma[value]
        electron_species.w[step] = ion_species.w[value]
        electron_species.w_center[step] = ion_species.w_center[value]
        electron_species.r_to_x[step] = ion_species.r_to_x[value]

    electron_species.id = np.arange(electron_species.num_particles, dtype=np.int32)

    return (
        electron_species.r,
        electron_species.dr_p,
        electron_species.pr,
        electron_species.pz,
        electron_species.gamma,
        electron_species.w,
        electron_species.w_center,
        electron_species.id,
        electron_species.r_to_x,
    )
