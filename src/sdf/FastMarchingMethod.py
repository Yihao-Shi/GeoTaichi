from src.utils.errors import NonfiniteStateError, SimulationStateError
import copy, math
import numpy as np
from functools import partial
from multiprocessing.pool import ThreadPool as Pool

from src.utils.linalg import certesian2_sphere, linearize, vectorize


class FastMarchingMethod(object):
    def __init__(self, space, coords, gnum) -> None:
        self.speed = 1

        # Definitions:
        # - knownState (0): that gp has a distance value we're sure about / can not modify anymore anyway
        # - trialState (1): for a gp in the narrow band, with at least one "known" neighbour. It carries some finite (no longer infinite) distance value we are unsure of
        # - farState   (2): just the initial state for all gridpoints
        self.gpStates = np.zeros(coords.shape[0]) + 2
        self.trials = np.array([], dtype=np.int32)
        self.known = np.array([])
        self.pool = Pool()
        self.phiField = None

        self.grid_space = space
        self.coords = copy.deepcopy(coords)
        self.gnum = gnum
        self.gridSum = int(gnum[0] * gnum[1] * gnum[2])

    def clear(self):
        del self.gpStates, self.trials, self.known, self.phiField, self.grid_space, self.coords, self.gnum, self.gridSum

    def phi_ini(self, objects):
        sides = objects.side(self.coords)
        self.phiField = np.select([sides > 0.0, sides < 0.0], [math.inf, -math.inf], default=0.0)

        for side in range(2):
            for i in range(self.gridSum):
                self.find_neighbors(i, side, objects)

    def find_neighbor(self, i, xyz, xInd, yInd, zInd, otherVal):
        n_g_px, n_g_py, n_g_pz = self.gnum
        if xyz == 0:
            otherVal = self.phiField[i] if xInd == 0 else self.phiField[linearize(xInd - 1, yInd, zInd, n_g_px, n_g_py)]
        elif xyz == 1:
            otherVal = (
                self.phiField[i]
                if xInd == n_g_px - 1
                else self.phiField[linearize(xInd + 1, yInd, zInd, n_g_px, n_g_py)]
            )
        elif xyz == 2:
            otherVal = self.phiField[i] if yInd == 0 else self.phiField[linearize(xInd, yInd - 1, zInd, n_g_px, n_g_py)]
        elif xyz == 3:
            otherVal = (
                self.phiField[i]
                if yInd == n_g_py - 1
                else self.phiField[linearize(xInd, yInd + 1, zInd, n_g_px, n_g_py)]
            )
        elif xyz == 4:
            otherVal = self.phiField[i] if zInd == 0 else self.phiField[linearize(xInd, yInd, zInd - 1, n_g_px, n_g_py)]
        elif xyz == 5:
            otherVal = (
                self.phiField[i]
                if zInd == n_g_pz - 1
                else self.phiField[linearize(xInd, yInd, zInd + 1, n_g_px, n_g_py)]
            )
        return otherVal

    def find_neighbors(self, i, exterior, objects):
        x_ind, y_ind, z_ind = vectorize(i, self.gnum[0], self.gnum[1])
        other_val = 0.0
        if (self.phiField[i] < 0.0 and exterior == 0) or (self.phiField[i] > 0.0 and exterior == 1):
            for xyz in range(6):
                other_val = self.find_neighbor(i, xyz, x_ind, y_ind, z_ind, other_val)

                if (other_val < 0.0 and exterior == 1) or (other_val > 0.0 and exterior == 0):
                    self.phiField[i] = objects._approximate_distance(self.coords[i])
                    if (self.phiField[i] < 0.0 and exterior == 1) or (self.phiField[i] > 0.0 and exterior == 0):
                        raise SimulationStateError("Not on the good side !")

    # =============================================================================================================================== #
    def fio_rose(self, grid_point):
        gp = certesian2_sphere(grid_point)
        r, theta, phi = gp[0], gp[1], gp[2]
        return r - 3 - 1.5 * np.sin(5 * theta) * np.sin(4 * phi)

    def grad_fio_rose(self, grid_point):
        gp = certesian2_sphere(grid_point)
        r, theta, phi = gp[0], gp[1], gp[2]
        if np.sin(gp[1]) == 0:
            raise SimulationStateError("theta = 0 [pi], gradient of rose fction not defined for its z component")
        return np.array(
            1,
            -7.5 / r * np.cos(5 * theta) * np.sin(4 * phi),
            -6 / r * np.sin(5 * theta) / np.sin(theta) * np.cos(4 * phi),
        )

    def ini_front(self, exterior, phiField, i):
        return_val = -1
        phi_val = phiField[i]
        if math.isfinite(phi_val):
            if (phi_val >= 0 and exterior == 1) or (phi_val <= 0 and exterior == 0):
                return_val = i
        return return_val

    def ini_states(self, exterior, phiField, gpStates, i):
        return_val = gpStates[i]
        phi_val = phiField[i]
        if math.isfinite(phi_val):
            if (phi_val >= 0 and exterior == 1) or (phi_val <= 0 and exterior == 0):
                return_val = 0
        return int(return_val)

    def trialize_from_known(self, i, exterior):
        n_g_px, n_g_py, n_g_pz = self.gnum
        x_ind, y_ind, z_ind = vectorize(i, n_g_px, n_g_py)
        if x_ind > 0:
            self.trialize(
                linearize(x_ind - 1, y_ind, z_ind, n_g_px, n_g_py), exterior
            )  # looking at the x- neighbor, if possible test whether that gp actually needs to be trialized will be performed therein
        if x_ind < n_g_px - 1:
            self.trialize(linearize(x_ind + 1, y_ind, z_ind, n_g_px, n_g_py), exterior)  # the x+ neighbor, if possible
        if y_ind > 0:
            self.trialize(linearize(x_ind, y_ind - 1, z_ind, n_g_px, n_g_py), exterior)  # y- neighbor if possible
        if y_ind < n_g_py - 1:
            self.trialize(linearize(x_ind, y_ind + 1, z_ind, n_g_px, n_g_py), exterior)  # y+
        if z_ind > 0:
            self.trialize(linearize(x_ind, y_ind, z_ind - 1, n_g_px, n_g_py), exterior)  # z-
        if z_ind < n_g_pz - 1:
            self.trialize(linearize(x_ind, y_ind, z_ind + 1, n_g_px, n_g_py), exterior)  # z+

    def trialize(self, i, exterior):
        if self.gpStates[i] != 0 and (
            (exterior == 1 and self.phiField[i] > 0) or (exterior == 0 and self.phiField[i] < 0)
        ):
            if self.gpStates[i] != 1:
                self.gpStates[i] = 1
                self.trials = np.append(self.trials, int(i))
            self.update_fast_marching_method(i, exterior)

    def phi_when_known(self, i, exterior):
        ret = 0.0
        if self.gpStates[i] == 0:
            ret = self.phiField[i]
        else:
            ret = math.inf if exterior == 1 else -math.inf
        return ret

    def eik_discr2(self, space, m0, m1):
        return 2 * space * space - math.pow(m0 - m1, 2)

    def eik_discr3(self, space, m0, m1, m2):
        return 3 * space * space - (math.pow(m0 - m1, 2) + math.pow(m0 - m2, 2) + math.pow(m1 - m2, 2))

    def surroundings(self, i, exterior):
        known_surr_val = []
        neigh = np.zeros(3)

        n_g_px, n_g_py, n_g_pz = self.gnum
        x_ind, y_ind, z_ind = vectorize(i, n_g_px, n_g_py)

        if x_ind == 0:
            neigh[0] = self.phi_when_known(linearize(x_ind + 1, y_ind, z_ind, n_g_px, n_g_py), exterior)
        elif x_ind == n_g_px - 1:
            neigh[0] = self.phi_when_known(linearize(x_ind - 1, y_ind, z_ind, n_g_px, n_g_py), exterior)
        else:
            neigh[0] = (
                min(
                    self.phi_when_known(linearize(x_ind - 1, y_ind, z_ind, n_g_px, n_g_py), exterior),
                    self.phi_when_known(linearize(x_ind + 1, y_ind, z_ind, n_g_px, n_g_py), exterior),
                )
                if exterior == 1
                else max(
                    self.phi_when_known(linearize(x_ind - 1, y_ind, z_ind, n_g_px, n_g_py), exterior),
                    self.phi_when_known(linearize(x_ind + 1, y_ind, z_ind, n_g_px, n_g_py), exterior),
                )
            )

        if y_ind == 0:
            neigh[1] = self.phi_when_known(linearize(x_ind, y_ind + 1, z_ind, n_g_px, n_g_py), exterior)
        elif y_ind == n_g_py - 1:
            neigh[1] = self.phi_when_known(linearize(x_ind, y_ind - 1, z_ind, n_g_px, n_g_py), exterior)
        else:
            neigh[1] = (
                min(
                    self.phi_when_known(linearize(x_ind, y_ind - 1, z_ind, n_g_px, n_g_py), exterior),
                    self.phi_when_known(linearize(x_ind, y_ind + 1, z_ind, n_g_px, n_g_py), exterior),
                )
                if exterior == 1
                else max(
                    self.phi_when_known(linearize(x_ind, y_ind - 1, z_ind, n_g_px, n_g_py), exterior),
                    self.phi_when_known(linearize(x_ind, y_ind + 1, z_ind, n_g_px, n_g_py), exterior),
                )
            )

        if z_ind == 0:
            neigh[2] = self.phi_when_known(linearize(x_ind, y_ind, z_ind + 1, n_g_px, n_g_py), exterior)
        elif z_ind == n_g_pz - 1:
            neigh[2] = self.phi_when_known(linearize(x_ind, y_ind, z_ind - 1, n_g_px, n_g_py), exterior)
        else:
            neigh[2] = (
                min(
                    self.phi_when_known(linearize(x_ind, y_ind, z_ind - 1, n_g_px, n_g_py), exterior),
                    self.phi_when_known(linearize(x_ind, y_ind, z_ind + 1, n_g_px, n_g_py), exterior),
                )
                if exterior == 1
                else max(
                    self.phi_when_known(linearize(x_ind, y_ind, z_ind - 1, n_g_px, n_g_py), exterior),
                    self.phi_when_known(linearize(x_ind, y_ind, z_ind + 1, n_g_px, n_g_py), exterior),
                )
            )

        for cpt in range(3):
            if math.isfinite(neigh[cpt]):
                known_surr_val.append(neigh[cpt])

        delta_pr = 0.0
        space = (
            self.grid_space if self.speed == 1 else self.grid_space * np.linalg.norm(self.grad_fio_rose(self.coords[i]))
        )
        if len(known_surr_val) == 1:
            delta_pr = -1  # no need for a discriminant in 1D propagation anyway
        elif len(known_surr_val) == 2:
            delta_pr = self.eik_discr2(space, known_surr_val[0], known_surr_val[1])
        elif len(known_surr_val) == 3:
            delta_pr = self.eik_discr3(space, known_surr_val[0], known_surr_val[1], known_surr_val[2])
        return np.array(known_surr_val), delta_pr, space

    def update_fast_marching_method(self, i, exterior):
        known_phi, delta_pr, space = self.surroundings(i, exterior)
        n_known = known_phi.shape[0]
        if n_known == 0:
            raise SimulationStateError(f"Gridpoint {i} goes through updateFastMarchingMethod no any known gp")
        elif n_known == 1:
            self.phiField[i] = known_phi[0] + space if exterior == 1 else known_phi[0] - space
        elif n_known == 2:
            m0, m1 = known_phi[0], known_phi[1]
            if delta_pr >= 0:
                self.phiField[i] = self.phi_from_eik2(m0, m1, delta_pr, exterior)
            else:
                self.phiField[i] = min(m0, m1) + space if exterior == 1 else max(m0, m1) - space
        elif n_known == 3:
            m0, m1, m2 = known_phi[0], known_phi[1], known_phi[2]
            if delta_pr >= 0.0:
                self.phiField[i] = self.phi_from_eik3(m0, m1, m2, delta_pr, exterior)
            else:
                two_ddiscr = np.array(
                    [self.eik_discr2(space, m0, m1), self.eik_discr2(space, m0, m2), self.eik_discr2(space, m1, m2)]
                )
                possible_phi = np.array([])
                for i_bis in range(3):
                    if two_ddiscr[i_bis] >= 0:
                        second_ind = 2 if i_bis > 1 else i_bis + 1
                        possible_phi = np.append(
                            possible_phi,
                            self.phi_from_eik3(
                                known_phi[i_bis / 2.0], known_phi[second_ind], two_ddiscr[i_bis], exterior
                            ),
                        )
                if possible_phi.shape[0] > 0:
                    self.phiField[i] = np.min(possible_phi) if exterior == 1 else np.max(possible_phi)
                else:
                    self.phiField[i] = np.min(possible_phi) + space if exterior == 1 else np.max(possible_phi) - space
        else:
            raise SimulationStateError(f"Unexpected case of {n_known} known neighbors around gp {i}")

        if (self.phiField[i] < 0 and exterior == 1) or (self.phiField[i] > 0 and exterior == 0):
            strings = "exterior" if exterior else "interior"
            raise SimulationStateError(
                f"We finally assigned phi = {self.phiField[i]} to {i} supposed to be in the {strings}."
            )

    def phi_from_eik2(self, m0, m1, disc, exterior):
        return (m0 + m1 + np.sqrt(disc)) / 2 if exterior else (m0 + m1 - np.sqrt(disc)) / 2

    def phi_from_eik3(self, m0, m1, m2, disc, exterior):
        return (m0 + m1 + m2 + np.sqrt(disc)) / 3 if exterior else (m0 + m1 + m2 - np.sqrt(disc)) / 3

    def confirm(self, i, phiVal, exterior, checkState=True):
        if checkState and self.gpStates[i] != 1:
            raise SimulationStateError(f"How comes ?? Current status is {self.gpStates[i]}")
        self.phiField[i] = phiVal
        self.gpStates[i] = 0
        self.trialize_from_known(i, exterior)

    def loop_trials(self, exterior):
        trial_gp = np.zeros(3, dtype=np.int32)
        closest = np.zeros(3, dtype=np.int32)
        while self.trials.shape[0] != 0:
            closest_it = 0
            closest_phi = self.phiField[int(self.trials[0])]
            curr_phi_value = copy.deepcopy(closest_phi)
            for i in range(self.trials.shape[0]):
                trial_gp = int(self.trials[i])
                curr_phi_value = self.phiField[trial_gp]
                if (exterior == 1 and curr_phi_value == math.inf) or (exterior == 0 and curr_phi_value == -math.inf):
                    raise NonfiniteStateError(
                        f"Skipping GP {trial_gp} in the loop because it still carries an +/- infinite value"
                    )
                if (exterior == 1 and curr_phi_value < closest_phi) or (exterior == 0 and curr_phi_value > closest_phi):
                    closest_it = i
                    closest_phi = curr_phi_value
            closest = int(self.trials[closest_it])
            self.trials = np.delete(self.trials, closest_it)
            self.confirm(closest, closest_phi, exterior)

    def phi(self):
        for side in range(2):
            func1 = partial(self.ini_states, side, self.phiField, self.gpStates)
            self.gpStates = np.array(self.pool.map(func1, range(self.gridSum)))
            func2 = partial(self.ini_front, side, self.phiField)
            known_tmp = np.array(self.pool.map(func2, range(self.gridSum)))
            known_tmp = known_tmp[known_tmp > -1]

            for gp_known in range(known_tmp.shape[0]):
                self.trialize_from_known(known_tmp[gp_known], side)
            self.loop_trials(side)

            self.known = np.append(self.known, known_tmp)
            del known_tmp
        self.pool.close()
        self.pool.join()
