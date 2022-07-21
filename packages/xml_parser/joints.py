from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Joint:
    pos: List[float] = (0., 0., 0.)
    axis: List[float] = (0., 0., 1.)
    limits: List[float] = (0., 0.)
    damping: float = 0.
    armature: float = 0.
    stiffness: float = 0.
    gear: float = 1.
    rigid: bool = False
    name: Optional[str] = None

    def mirror(self, axis: int):
        if self.rigid:
            return self

        if self.axis[axis] == 1.:
            return self
        else:
            return Joint(pos=self.pos, axis=[-a for a in self.axis], limits=self.limits,
                         damping=self.damping, armature=self.armature,
                         stiffness=self.stiffness, gear=self.gear,
                         rigid=self.rigid, name=self.name)


class RigidJoint(Joint):
    def __init__(self):
        Joint.__init__(self, rigid=True)


@dataclass
class SmallJoint(Joint):
    damping: float = 1.0
    armature: float = 0.005
    stiffness: float = 2.0
    gear: float = 22.5


@dataclass
class MediumJoint(Joint):
    damping: float = 3.0
    armature: float = 0.01
    stiffness: float = 6.0
    gear: float = 45.0


@dataclass
class LargeJoint(Joint):
    damping: float = 5.0
    armature: float = 0.02
    stiffness: float = 12.0
    gear: float = 67.5


@dataclass
class MediumStiffJoint(MediumJoint):
    stiffness: float = 12.0


@dataclass
class MediumPowerJoint(MediumJoint):
    stiffness: float = 12.0
    gear: float = 90.0


@dataclass
class LargeStiffJoint(LargeJoint):
    stiffness: float = 24.0


@dataclass
class LargePowerJoint(LargeJoint):
    stiffness: float = 24.0
    gear: float = 135.0


if __name__ == '__main__':
    j = LargePowerJoint(pos=[0, 0, 0], axis=[1, 0, 0], limits=[-40, 40])
    print(j)
