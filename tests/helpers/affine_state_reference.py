"""Host affine-state predictor/corrector oracle for tests only."""

from src.dem.engines.AffineBodyState import AffineBodyState


class ReferenceAffineBodyState(AffineBodyState):
    @staticmethod
    def from_scene(scene, sims):
        state = AffineBodyState.from_scene(scene, sims)
        return ReferenceAffineBodyState(state.bodies, state.gravity)

    def begin_step(self, dt):
        self.tilde_y = self.y + dt * self.v_y
        self.hat_y = self.y.copy()

    def accept_step(self, y_new, dt):
        old_y = self.y.copy()
        old_v = self.v_y.copy()
        self.v_y = (y_new - old_y) / dt
        self.y_n1 = old_y
        self.v_y_n1 = old_v
        self.y = y_new.copy()
