###
# Utils for transforms between frames
###

from utils import Transform
import numpy as np
from xml_parser import geoms
from typing import Sequence


def compose(*transforms):
    def f(geom):
        out = Transform()
        for t in reversed(transforms):
            out = out(t(geom))
        return out
    return f


def apply(t: Transform):
    def f(geom):
        return t
    return f


def translate_origin_to_p1(ind: int):
    def fn(geom: Sequence[geoms.Geom]):
        capsule = geom[ind]
        assert isinstance(capsule, geoms.Capsule)
        return Transform(pos=capsule.p1, quat=[0,0,0,1])
    return fn


def translate_origin_to_p2(ind: int):
    def fn(geom: Sequence[geoms.Geom]):
        capsule = geom[ind]
        assert isinstance(capsule, geoms.Capsule)
        return Transform(pos=capsule.p2, quat=[0,0,0,1])
    return fn


def translate_origin_to_center(ind: int):
    def fn(geom: Sequence[geoms.Geom]):
        capsule = geom[ind]
        assert isinstance(capsule, geoms.Capsule)
        center = (np.array(capsule.p1) + np.array(capsule.p2)) / 2.0
        return Transform(pos=center, quat=[0,0,0,1])
    return fn


def translate_origin_to_point(ind: int):
    def fn(geom: Sequence[geoms.Geom]):
        capsule = geom[ind]
        assert isinstance(capsule, geoms.Capsule)
        center = (np.array(capsule.p1) + np.array(capsule.p2)) / 2.0
        return Transform(pos=center, quat=[0,0,0,1])
    return fn
