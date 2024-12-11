import copy, math
import numpy as np
from functools import partial
from multiprocessing.pool import ThreadPool as Pool

from src.utils.linalg import Certesian2Sphere, linearize, vectorize


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

    def phiIni(self, objects):
        sides = objects.side(self.coords)
        self.phiField = np.select([sides > 0., sides < 0.], [math.inf, -math.inf], default=0.)

        for side in range(2):
            for i in range(self.gridSum):
                self.find_neighbors(i, side, objects)

    def find_neighbor(self, i, xyz, xInd, yInd, zInd, otherVal):
        nGPx, nGPy, nGPz = self.gnum
        if xyz == 0:
            otherVal = self.phiField[i] if xInd == 0 else self.phiField[linearize(xInd-1, yInd, zInd, nGPx, nGPy)]
        elif xyz == 1:
            otherVal = self.phiField[i] if xInd == nGPx - 1 else self.phiField[linearize(xInd+1, yInd, zInd, nGPx, nGPy)]
        elif xyz == 2:
            otherVal = self.phiField[i] if yInd == 0 else self.phiField[linearize(xInd, yInd-1, zInd, nGPx, nGPy)]
        elif xyz == 3:
            otherVal = self.phiField[i] if yInd == nGPy - 1 else self.phiField[linearize(xInd, yInd+1, zInd, nGPx, nGPy)]
        elif xyz == 4:
            otherVal = self.phiField[i] if zInd == 0 else self.phiField[linearize(xInd, yInd, zInd-1, nGPx, nGPy)]
        elif xyz == 5:
            otherVal = self.phiField[i] if zInd == nGPz - 1 else self.phiField[linearize(xInd, yInd, zInd+1, nGPx, nGPy)]
        return otherVal
    
    def find_neighbors(self, i, exterior, objects):
        xInd, yInd, zInd = vectorize(i, self.gnum[0], self.gnum[1])
        otherVal = 0.
        if (self.phiField[i] < 0. and exterior == 0) or (self.phiField[i] > 0. and exterior == 1): 
            for xyz in range(6):
                otherVal = self.find_neighbor(i, xyz, xInd, yInd, zInd, otherVal)

                if (otherVal < 0. and exterior == 1) or (otherVal > 0. and exterior == 0):
                    self.phiField[i] = objects._approximate_distance(self.coords[i])
                    if (self.phiField[i] < 0. and exterior == 1) or (self.phiField[i] > 0. and exterior == 0):
                        raise RuntimeError("Not on the good side !")
    
    # =============================================================================================================================== #
    def fioRose(self, grid_point):
        gp = Certesian2Sphere(grid_point)
        r, theta, phi = gp[0], gp[1], gp[2]
        return r - 3 - 1.5 * np.sin(5 * theta) * np.sin(4 * phi)

    def grad_fioRose(self, grid_point):
        gp = Certesian2Sphere(grid_point)
        r, theta, phi = gp[0], gp[1], gp[2]
        if np.sin(gp[1]) == 0:
            raise RuntimeError("theta = 0 [pi], gradient of rose fction not defined for its z component")
        return np.array(1, -7.5 / r * np.cos(5 * theta) * np.sin(4 * phi), -6 / r * np.sin(5 * theta) / np.sin(theta) * np.cos(4 * phi))

    def iniFront(self, exterior, phiField, i):
        returnVal = -1
        phiVal = phiField[i]
        if math.isfinite(phiVal):
            if (phiVal >= 0 and exterior == 1) or (phiVal <= 0 and exterior == 0):
                returnVal = i
        return returnVal
    
    def iniStates(self, exterior, phiField, gpStates, i):
        returnVal = gpStates[i]
        phiVal = phiField[i]
        if math.isfinite(phiVal):
            if (phiVal >= 0 and exterior == 1) or (phiVal <= 0 and exterior == 0):
                returnVal = 0
        return int(returnVal)
    
    def trializeFromKnown(self, i, exterior):
        nGPx, nGPy, nGPz = self.gnum
        xInd, yInd, zInd = vectorize(i, nGPx, nGPy)
        if xInd > 0:   self.trialize(linearize(xInd - 1, yInd, zInd, nGPx, nGPy), exterior)                 # looking at the x- neighbor, if possible test whether that gp actually needs to be trialized will be performed therein
        if xInd < nGPx - 1: self.trialize(linearize(xInd + 1, yInd, zInd, nGPx, nGPy), exterior)            # the x+ neighbor, if possible
        if yInd > 0: self.trialize(linearize(xInd, yInd-1, zInd, nGPx, nGPy), exterior)                     # y- neighbor if possible
        if yInd < nGPy - 1: self.trialize(linearize(xInd, yInd+1, zInd, nGPx, nGPy), exterior)              # y+
        if zInd > 0: self.trialize(linearize(xInd, yInd, zInd-1, nGPx, nGPy), exterior)                     # z-
        if zInd < nGPz - 1: self.trialize(linearize(xInd, yInd, zInd+1, nGPx, nGPy), exterior)              # z+

    def trialize(self, i, exterior):
        if self.gpStates[i] != 0 and ((exterior == 1 and self.phiField[i] > 0) or (exterior == 0 and self.phiField[i] < 0)):
            if self.gpStates[i] != 1:
                self.gpStates[i] = 1
                self.trials = np.append(self.trials, int(i))
            self.updateFastMarchingMethod(i, exterior)

    def phiWhenKnown(self, i, exterior):
        ret = 0.
        if self.gpStates[i] == 0:
            ret = self.phiField[i]
        else:
            ret = math.inf if exterior == 1 else -math.inf
        return ret
    
    def eikDiscr2(self, space, m0, m1):
        return 2 * space * space - math.pow(m0 - m1, 2)
    
    def eikDiscr3(self, space, m0, m1, m2):
        return 3 * space * space - (math.pow(m0 - m1, 2) + math.pow(m0 - m2, 2) + math.pow(m1 - m2, 2))
    
    def surroundings(self, i, exterior):
        knownSurrVal = []
        neigh = np.zeros(3)

        nGPx, nGPy, nGPz = self.gnum
        xInd, yInd, zInd = vectorize(i, nGPx, nGPy)

        if xInd == 0:
            neigh[0] = self.phiWhenKnown(linearize(xInd + 1, yInd, zInd, nGPx, nGPy), exterior)
        elif xInd == nGPx - 1:
            neigh[0] = self.phiWhenKnown(linearize(xInd - 1, yInd, zInd, nGPx, nGPy), exterior)
        else:
            neigh[0] = min(self.phiWhenKnown(linearize(xInd - 1, yInd, zInd, nGPx, nGPy), exterior), 
                           self.phiWhenKnown(linearize(xInd + 1, yInd, zInd, nGPx, nGPy), exterior)) if exterior == 1 else \
                       max(self.phiWhenKnown(linearize(xInd - 1, yInd, zInd, nGPx, nGPy), exterior), 
                           self.phiWhenKnown(linearize(xInd + 1, yInd, zInd, nGPx, nGPy), exterior))
            
        if yInd == 0:
            neigh[1] = self.phiWhenKnown(linearize(xInd, yInd+1, zInd, nGPx, nGPy), exterior)
        elif yInd == nGPy - 1:
            neigh[1] = self.phiWhenKnown(linearize(xInd, yInd-1, zInd, nGPx, nGPy), exterior)
        else:
            neigh[1] = min(self.phiWhenKnown(linearize(xInd, yInd-1, zInd, nGPx, nGPy), exterior), 
                           self.phiWhenKnown(linearize(xInd, yInd+1, zInd, nGPx, nGPy), exterior)) if exterior == 1 else \
                       max(self.phiWhenKnown(linearize(xInd, yInd-1, zInd, nGPx, nGPy), exterior), 
                           self.phiWhenKnown(linearize(xInd, yInd+1, zInd, nGPx, nGPy), exterior))
            
        if zInd == 0:
            neigh[2] = self.phiWhenKnown(linearize(xInd, yInd, zInd+1, nGPx, nGPy), exterior)
        elif zInd == nGPz - 1:
            neigh[2] = self.phiWhenKnown(linearize(xInd, yInd, zInd-1, nGPx, nGPy), exterior)
        else:
            neigh[2] = min(self.phiWhenKnown(linearize(xInd, yInd, zInd-1, nGPx, nGPy), exterior), 
                           self.phiWhenKnown(linearize(xInd, yInd, zInd+1, nGPx, nGPy), exterior)) if exterior == 1 else \
                       max(self.phiWhenKnown(linearize(xInd, yInd, zInd-1, nGPx, nGPy), exterior), 
                           self.phiWhenKnown(linearize(xInd, yInd, zInd+1, nGPx, nGPy), exterior))
            
        for cpt in range(3):
            if math.isfinite(neigh[cpt]):
                knownSurrVal.append(neigh[cpt])

        deltaPr = 0.
        space = self.grid_space if self.speed == 1 else self.grid_space * np.linalg.norm(self.grad_fioRose(self.coords[i]))
        if len(knownSurrVal) == 1:
            deltaPr = -1                         # no need for a discriminant in 1D propagation anyway
        elif len(knownSurrVal) == 2:
            deltaPr = self.eikDiscr2(space, knownSurrVal[0], knownSurrVal[1])
        elif len(knownSurrVal) == 3:
            deltaPr = self.eikDiscr3(space, knownSurrVal[0], knownSurrVal[1], knownSurrVal[2])
        return np.array(knownSurrVal), deltaPr, space

    def updateFastMarchingMethod(self, i, exterior):
        knownPhi, deltaPr, space = self.surroundings(i, exterior)
        nKnown = knownPhi.shape[0]
        if nKnown == 0: raise RuntimeError(f"Gridpoint {i} goes through updateFastMarchingMethod no any known gp")
        elif nKnown == 1:
            self.phiField[i] = knownPhi[0] + space if exterior == 1 else knownPhi[0] - space
        elif nKnown == 2:
            m0, m1 = knownPhi[0], knownPhi[1]
            if deltaPr >= 0:
                self.phiField[i] = self.phiFromEik2(m0, m1, deltaPr, exterior)
            else:
                self.phiField[i] = min(m0, m1) + space if exterior == 1 else max(m0, m1) - space
        elif nKnown == 3:
            m0, m1, m2 = knownPhi[0], knownPhi[1], knownPhi[2]
            if deltaPr >= 0.:
                self.phiField[i] = self.phiFromEik3(m0, m1, m2, deltaPr, exterior)
            else:
                twoDdiscr = np.array([self.eikDiscr2(space, m0, m1), self.eikDiscr2(space, m0, m2), self.eikDiscr2(space, m1, m2)])
                possiblePhi = np.array([])
                for iBis in range(3):
                    if twoDdiscr[iBis] >= 0:
                        secondInd = 2 if iBis > 1 else iBis + 1
                        possiblePhi = np.append(possiblePhi, self.phiFromEik3(knownPhi[iBis / 2.], knownPhi[secondInd], twoDdiscr[iBis], exterior))
                if possiblePhi.shape[0] > 0:
                    self.phiField[i] = np.min(possiblePhi) if exterior == 1 else np.max(possiblePhi)
                else:
                    self.phiField[i] = np.min(possiblePhi) + space if exterior == 1 else np.max(possiblePhi) - space
        else:
            raise RuntimeError(f"Unexpected case of {nKnown} known neighbors around gp {i}")
        
        if (self.phiField[i] < 0 and exterior == 1) or (self.phiField[i] > 0 and exterior == 0):
            strings = "exterior" if exterior else "interior"
            raise RuntimeError(f"We finally assigned phi = {self.phiField[i]} to {i} supposed to be in the {strings}.")
        
    def phiFromEik2(self, m0, m1, disc, exterior):
        return (m0 + m1 + np.sqrt(disc)) / 2 if exterior else (m0 + m1 - np.sqrt(disc)) / 2
    
    def phiFromEik3(self, m0, m1, m2, disc, exterior):
        return (m0 + m1 + m2 + np.sqrt(disc)) / 3 if exterior else (m0 + m1 + m2 - np.sqrt(disc)) / 3

    def confirm(self, i, phiVal, exterior, checkState=True):
        if checkState and self.gpStates[i] != 1:
            raise RuntimeError(f"How comes ?? Current status is {self.gpStates[i]}")
        self.phiField[i] = phiVal
        self.gpStates[i] = 0
        self.trializeFromKnown(i, exterior)

    def loopTrials(self, exterior):
        trialGP = np.zeros(3, dtype=np.int32)
        closest = np.zeros(3, dtype=np.int32)
        while self.trials.shape[0] != 0:
            closestIt = 0
            closestPhi = self.phiField[int(self.trials[0])]
            currPhiValue = copy.deepcopy(closestPhi)
            for i in range(self.trials.shape[0]):
                trialGP = int(self.trials[i])
                currPhiValue = self.phiField[trialGP]
                if (exterior == 1 and currPhiValue == math.inf) or (exterior == 0 and currPhiValue == -math.inf):
                    raise RuntimeError(f"Skipping GP {trialGP} in the loop because it still carries an +/- infinite value")
                if (exterior == 1 and currPhiValue < closestPhi) or (exterior == 0 and currPhiValue > closestPhi):
                    closestIt  = i
                    closestPhi = currPhiValue
            closest = int(self.trials[closestIt])
            self.trials = np.delete(self.trials, closestIt)
            self.confirm(closest, closestPhi, exterior)

    def phi(self):
        for side in range(2):
            func1 = partial(self.iniStates, side, self.phiField, self.gpStates)
            self.gpStates = np.array(self.pool.map(func1, range(self.gridSum)))
            func2 = partial(self.iniFront, side, self.phiField)
            knownTmp = np.array(self.pool.map(func2, range(self.gridSum)))
            knownTmp = knownTmp[knownTmp > -1]

            for gpKnown in range(knownTmp.shape[0]):
                self.trializeFromKnown(knownTmp[gpKnown], side)
            self.loopTrials(side)

            self.known = np.append(self.known, knownTmp)
            del knownTmp
        self.pool.close()
        self.pool.join()

        

