from dataclasses import dataclass
from typing import List


class Geom():
    pass

    def mirror(self, axis: int):
        pass



@dataclass
class Sphere(Geom):
    pos: List[float] = (0., 0., 0.)
    size: float = 0.
    density: float = 1000.
    friction: List[float] = (1.5, 0.005, 0.0001)

    def mirror(self, axis: int):
        new_pos = []
        for i, x in enumerate(self.pos):
            if i == axis:
                new_pos.append(-x)
            else:
                new_pos.append(x)
        return Sphere(new_pos, self.size, self.density, self.friction)


@dataclass
class Capsule(Geom):
    p1: List[float]
    p2: List[float]
    size: float = 0.
    density: float = 1000.
    friction: List[float] = (1.5, 0.005, 0.0001)

    def mirror(self, axis: int):
        p1, p2 = [], []
        for i, (x1, x2) in enumerate(zip(self.p1, self.p2)):
            if i == axis:
                p1.append(-x1)
                p2.append(-x2)
            else:
                p1.append(x1)
                p2.append(x2)
        return Capsule(p1, p2, self.size, self.density, self.friction)


@dataclass
class Cylinder(Geom):
    p1: List[float]
    p2: List[float]
    size: float = 0.
    density: float = 1000.
    friction: List[float] = (1.5, 0.005, 0.0001)

    def mirror(self, axis: int):
        p1, p2 = [], []
        for i, (x1, x2) in enumerate(zip(self.p1, self.p2)):
            if i == axis:
                p1.append(-x1)
                p2.append(-x2)
            else:
                p1.append(x1)
                p2.append(x2)
        return Cylinder(p1, p2, self.size, self.density, self.friction)
