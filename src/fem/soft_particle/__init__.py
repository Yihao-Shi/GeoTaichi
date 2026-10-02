"""FEM soft-particle contact built on explicit surface geometry."""

from src.fem.soft_particle.ContactModel import FEMSoftParticleContactModel
from src.fem.soft_particle.ContactManager import FEMSoftParticleContactManager

__all__ = ["FEMSoftParticleContactManager", "FEMSoftParticleContactModel"]
