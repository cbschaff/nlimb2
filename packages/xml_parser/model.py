# import lxml.etree as ET
import xml.etree.ElementTree as ET
from utils import Transform
import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation
from xml_parser.joints import Joint
from xml_parser.geoms import Geom, Sphere, Capsule


class NameManager():
    def __init__(self):
        self.counts = {}

    def get_name(self, tag: str):
        if tag not in self.counts:
            self.counts[tag] = 0
        name = f'{tag}{self.counts[tag]}'
        self.counts[tag] += 1
        return name


class XMLModel():
    def __init__(self, name=None):
        self.root = Body(name=name)

    @staticmethod
    def from_path(path):
        m = XMLModel()
        xml = ET.parse(path)
        m.root = Body.from_xml(xml.getroot().find('worldbody').find('body'),
                               xml.getroot().find('actuator'))
        return m

    def adjust_root(self, fac=1.0, eps=0.1):
        """Raise the root body so that there is no collision with the ground"""
        self.root.pos[2] = eps - fac * self.root.get_lowest_point()

    def get_height(self):
        return self.root.pos[2]

    def write(self, path):
        xml = ET.Element('mujoco', {'model': 'robot'})
        ET.SubElement(xml, 'compiler', {'angle': 'degree', 'inertiafromgeom': 'true'})
        assets = ET.SubElement(xml, 'asset')
        ET.SubElement(assets, 'material', attrib={'name': 'default', 'rgba': '0.97 0.38 0.06 1.0'})
        ET.SubElement(assets, 'material', attrib={'name': 'red', 'rgba': '1.0 0.0 0.0 1.0'})
        ET.SubElement(assets, 'material', attrib={'name': 'green', 'rgba': '0.0 1.0 0.0 1.0'})
        ET.SubElement(assets, 'material', attrib={'name': 'blue', 'rgba': '0.0 0.0 1.0 1.0'})
        ET.SubElement(assets, 'material', attrib={'name': 'white', 'rgba': '1.0 1.0 1.0 1.0'})
        ET.SubElement(assets, 'material', attrib={'name': 'black', 'rgba': '0.0 0.0 0.0 1.0'})
        worldbody = ET.SubElement(xml, 'worldbody')
        actuator = ET.SubElement(xml, 'actuator')
        def _to_str(l: list):
            return ' '.join([str(x) for x in l])
        root_body = ET.SubElement(worldbody, 'body', {
                    'name': 'root',
                    'pos': _to_str(self.root.pos),
                    'quat': _to_str(list(self.root.quat[3:4]) + list(self.root.quat[:3]))
        })
        ET.SubElement(root_body, 'freejoint', {'name': 'freejoint'})
        self.root.to_xml(root_body, actuator)

        with open(path, 'wb') as f:
            f.write(ET.tostring(xml))

    def write_urdf(self, path):
        xml = ET.Element('robot', {'name': 'robot'})
        self.root.to_urdf(xml)
        mat = ET.SubElement(xml, 'material', attrib={'name': 'link'})
        ET.SubElement(mat, 'color', attrib={'rgba': '0.97 0.38 0.06 1.0'})

        with open(path, 'wb') as f:
            f.write(ET.tostring(xml))


class Body():
    def __init__(self, name=None, pos=[0.,0.,0.], quat=[0.,0.,0.,1.]):
        self.name = name
        self.pos = np.array(pos)
        self.quat = np.array(quat)
        self._frame = None
        self.geoms = []
        self.joints = []
        self.bodies = []

    @property
    def frame(self):
        if self._frame is None:
            self._frame = Transform(self.pos, self.quat)
        return self._frame

    def add_geom(self, g: Geom):
        self.geoms.append(g)

    def add_joint(self, j: Joint):
        self.joints.append(j)

    def add_body(self, name=None, pos=[0.,0.,0.], quat=[0.,0.,0.,1.]):
        self.bodies.append(Body(name, pos, quat))
        return self.bodies[-1]

    def to_xml(self, xml: ET.Element, actuator: ET.Element,
               name_manager: Optional[NameManager] = None):

        def _to_str(l: list):
            return ' '.join([str(x) for x in l])

        if name_manager is None:
            name_manager = NameManager()
        if len(self.joints) == 3:
            mat_name = 'white'
        elif len(self.joints) == 0:
            mat_name = 'black'
        elif self.joints[0].axis[0] > 0.5:
            mat_name = 'red'
        elif self.joints[0].axis[1] > 0.5:
            mat_name = 'green'
        elif self.joints[0].axis[2] > 0.5:
            mat_name = 'blue'
        else:
            mat_name = 'default'
        for g in self.geoms:
            if isinstance(g, Capsule):
                ET.SubElement(xml, 'geom', attrib={
                    'type': 'capsule',
                    'fromto': _to_str(list(g.p1) + list(g.p2)),
                    'size': _to_str([g.size]),
                    'density': str(g.density),
                    'friction': _to_str(g.friction),
                    'material': mat_name,
                    'name': name_manager.get_name('geom')
                })
            if isinstance(g, Sphere):
                ET.SubElement(xml, 'geom', attrib={
                    'type': 'sphere',
                    'pos': _to_str(list(g.pos)),
                    'size': _to_str([g.size]),
                    'density': str(g.density),
                    'friction': _to_str(g.friction),
                    'material': mat_name,
                    'name': name_manager.get_name('geom')
                })

        for j in self.joints:
            joint_name = j.name if j.name is not None else name_manager.get_name('joint')
            ET.SubElement(xml, 'joint', attrib={
                'type': 'hinge',
                'pos': _to_str(list(j.pos)),
                'axis': _to_str(j.axis),
                'range': _to_str(j.limits),
                'limited': 'true',
                'damping': str(j.damping),
                'stiffness': str(j.stiffness),
                'armature': str(j.armature),
                'name': joint_name
            })
            ET.SubElement(actuator, 'motor', attrib={
                'ctrllimited': 'true',
                'ctrlrange': '-1 1',
                'gear': str(j.gear),
                'joint': joint_name,
                'name': name_manager.get_name('actuator'),
            })
        if len(self.joints) > 0:
            self._add_site(xml, self.joints, name_manager)

        for body in self.bodies:
            if body.name is None:
                body.name = name_manager.get_name('body')
            child = ET.SubElement(xml, 'body', attrib={
                        'name': body.name,
                        'pos': _to_str(list(body.pos)),
                        'quat': _to_str(list(body.quat[3:4]) + list(body.quat[:3])),
                    })

            body.to_xml(child, actuator, name_manager)

    def _add_site(self, element, joints, name_manager):
        def _to_str(l: list):
            return ' '.join([str(x) for x in l])

        if len(joints) == 3:
            # ball joint
            ET.SubElement(element, 'site', attrib={
                    'name': name_manager.get_name('site'),
                    'type': 'sphere',
                    'pos': _to_str(list(joints[0].pos)),
                    'size': '0.1',
                    'rgba': '0.2 1.0 1.0 1.0'
                })
        elif joints[0].rigid:
            ET.SubElement(element, 'site', attrib={
                    'name': name_manager.get_name('site'),
                    'type': 'box',
                    'pos': _to_str(list(joints[0].pos)),
                    'size': '0.04 0.04 0.04',
                    'rgba': '1.0 1.0 0.2 1.0'
                })
        else:
            j = joints[0]
            axis = np.array(j.axis)
            axis /= np.linalg.norm(axis)
            origin = np.array(j.pos)
            p1 = origin - 0.05 * axis
            p2 = origin + 0.05 * axis
            fromto = p1.tolist() + p2.tolist()
            ET.SubElement(element, 'site', attrib={
                    'type': 'cylinder',
                    'fromto': _to_str(fromto),
                    'size': '0.1',
                    'rgba': _to_str(axis.tolist()[::-1] + [1.0])
                })


    @staticmethod
    def from_xml(xml: ET.Element, actuator: ET.Element):

        def _parse(attrib: dict, k: str):
            v = attrib.get(k, None)
            if v is None:
                return v
            else:
                try:
                    return float(v)
                except ValueError:
                    if ' ' in v:
                        return [float(x) for x in attrib[k].split()]
                    else:
                        return v

        def _parse_available(attrib, keys):
            out = {}
            for k in keys:
                v = _parse(attrib, k)
                if v is not None:
                    out[k] = v
            return out

        attr = _parse_available(xml.attrib, ['name', 'pos', 'quat'])
        if 'quat' in attr:
            attr['quat'] = attr['quat'][1:] + attr['quat'][0:1]
        body = Body(**attr)
        for g in xml.findall('geom'):
            if g.attrib['type'] == 'capsule':
                attr = _parse_available(g.attrib, ['fromto', 'size', 'density', 'friction'])
                attr['p1'] = attr['fromto'][:3]
                attr['p2'] = attr['fromto'][3:]
                del attr['fromto']
                body.add_geom(Capsule(**attr))
            elif g.attrib['type'] == 'sphere':
                attr = _parse_available(g.attrib, ['pos', 'size', 'density', 'friction'])
                body.add_geom(Sphere(**attr))
            else:
                raise ValueError(f'Unknown Geom: {g.attrib["type"]}')
        for j in xml.findall('joint'):
            motor = None
            for m in actuator.findall('motor'):
                if m.attrib['joint'] == j.attrib['name']:
                    motor = m
            if motor is None:
                raise RuntimeError(f'Unable to find a motor associated with joint {j.attrib["name"]}')
            attr = _parse_available(j.attrib, ['pos', 'axis', 'range', 'damping', 'armature',
                                               'stiffness', 'name'])
            attr['limits'] = attr['range']
            del attr['range']
            attr['gear'] = _parse(motor.attrib, 'gear')
            body.add_joint(Joint(**attr))
        for b in xml.findall('body'):
            body.bodies.append(Body.from_xml(b, actuator))
        return body

    def get_lowest_point(self, transform=None):
        if transform is None:
            transform = self.frame
        else:
            transform = transform(self.frame)

        for joint in self.joints:
            angle = max(min(0.0, joint.limits[1]), joint.limits[0])
            angle *= np.pi / 180.  # convert to radians
            axis = joint.axis / np.linalg.norm(joint.axis)
            rot = Rotation.from_rotvec(axis * angle)
            pos = np.asarray(joint.pos) if joint.pos is not None else np.zeros(3)
            t1 = Transform(-pos, np.array([0, 0, 0, 1]))
            t2 = Transform(pos, rot.as_quat())
            transform = transform(t2(t1))

        minz = np.inf

        for body in self.bodies:
            minz = min(body.get_lowest_point(transform), minz)

        for geom in self.geoms:
            if isinstance(geom, Sphere):
                pos = np.asarray(geom.pos) if geom.pos is not None else np.zeros(3)
                p = transform(pos)
                z = p[2] - geom.size
                minz = min(minz, z)
            elif isinstance(geom, Capsule):
                points = transform(np.array([geom.p1, geom.p2]))
                z = points[:, 2] - geom.size
                minz = min(minz, np.min(z))
        return minz

    def get_body_count(self):
        count = 1
        for body in self.bodies:
            count += body.get_body_count()
        return count

    def to_urdf(self, xml: ET.Element,
                name_manager: Optional[NameManager] = None,
                parent_link_name: str = None):

        def _to_str(l: list):
            return ' '.join([str(x) for x in l])

        if name_manager is None:
            name_manager = NameManager()

        # child_link_name = name_manager.get_name('link')
        link = ET.SubElement(xml, 'link', attrib={'name': self.name})
        for g in self.geoms:
            visual = ET.SubElement(link, 'visual')
            ET.SubElement(visual, 'material', attrib={'name': 'link'})
            collision = ET.SubElement(link, 'collision')
            for element in [visual, collision]:
                if isinstance(g, Capsule):
                    # adjust ori so capsule axis is aligned with z axis
                    origin = [(x1 + x2) / 2. for x1,x2 in zip(g.p1, g.p2)]
                    geom_axis = np.array([x2 - x1 for x1,x2 in zip(g.p1, g.p2)])
                    length = np.linalg.norm(geom_axis)
                    geom_axis /= length
                    rot_axis = np.array([-geom_axis[1], geom_axis[0], 0])
                    if np.allclose(rot_axis, 0.0):
                        rpy = [0., 0., 0.]
                    else:
                        rot_axis /= np.linalg.norm(rot_axis)
                        angle = np.arccos(geom_axis[2])
                        rpy = Rotation.from_rotvec(angle * rot_axis).as_euler('xyz').tolist()
                    ET.SubElement(element, 'origin',
                                  attrib={'xyz': _to_str(origin), 'rpy': _to_str(rpy)})
                    geom = ET.SubElement(element, 'geometry')
                    ET.SubElement(geom, 'cylinder', attrib={
                        'radius': str(g.size),
                        'length': str(length)
                    })
                if isinstance(g, Sphere):
                    ET.SubElement(element, 'origin',
                                  attrib={'xyz': _to_str(g.pos), 'rpy': '0 0 0'})
                    geom = ET.SubElement(element, 'geometry')
                    ET.SubElement(geom, 'sphere', attrib={
                        'radius': str(g.size),
                    })
        if parent_link_name is None and len(self.joints) > 0:
            raise ValueError('njoints > 0, but node seems to be the root.')
        if len(self.joints) == 1:
            # hinge joint
            j = self.joints[0]
            joint_name = j.name if j.name is not None else name_manager.get_name('joint')
            joint = ET.SubElement(xml, 'joint', attrib={'name': joint_name, 'type': 'revolute'})
            rpy = Rotation.from_quat(self.quat).as_euler('xyz').tolist()
            limits = [np.pi * lim / 180. for lim in j.limits]
            ET.SubElement(joint, 'origin', attrib={'xyz': _to_str(self.pos), 'rpy': _to_str(rpy)})
            ET.SubElement(joint, 'parent', attrib={'link': parent_link_name})
            ET.SubElement(joint, 'child', attrib={'link': self.name})
            ET.SubElement(joint, 'axis', attrib={'xyz': _to_str(j.axis)})
            ET.SubElement(joint, 'limit', attrib={'lower': str(limits[0]),
                                                  'upper': str(limits[1]),
                                                  'effort': str(j.gear),
                                                  'velocity': '3.14'})
            ET.SubElement(joint, 'dynamics', attrib={'damping': str(j.damping)})
            marker = ET.SubElement(link, 'visual')
            # adjust joint axis so that it is aligned with z axis
            axis = np.array(j.axis)
            length = np.linalg.norm(axis)
            axis /= length
            rot_axis = np.array([-axis[1], axis[0], 0])
            if np.allclose(rot_axis, 0.0):
                rpy = [0., 0., 0.]
            else:
                rot_axis /= np.linalg.norm(rot_axis)
                angle = np.arccos(axis[2])
                rpy = Rotation.from_rotvec(angle * rot_axis).as_euler('xyz').tolist()
            ET.SubElement(marker, 'origin', attrib={'rpy': _to_str(rpy)})
            geom = ET.SubElement(marker, 'geometry')
            ET.SubElement(geom, 'box', attrib={'size': '0.05 0.05 0.1'})
            mat = ET.SubElement(marker, 'material', attrib={'name':'marker'})
            ET.SubElement(mat, 'color', attrib={'rgba': ' '.join([str(x) for x in j.axis]) + ' 1.0'})

        elif len(self.joints) > 1:
            # spherical joint
            assert len(self.joints) == 3
            j = self.joints[0]
            joint_name = j.name if j.name is not None else name_manager.get_name('joint')
            for i, j in enumerate(self.joints):
                joint = ET.SubElement(xml, 'joint', attrib={'name': joint_name + f'_{i}',
                                                            'type': 'revolute'})
                rpy = Rotation.from_quat(self.quat).as_euler('xyz').tolist()
                limits = [np.pi * lim / 180. for lim in j.limits]
                ET.SubElement(joint, 'origin', attrib={'xyz': _to_str(self.pos), 'rpy': _to_str(rpy)})
                ET.SubElement(joint, 'axis', attrib={'xyz': _to_str(j.axis)})
                ET.SubElement(joint, 'limit', attrib={'lower': str(limits[0]),
                                                      'upper': str(limits[1]),
                                                      'effort': str(j.gear),
                                                      'velocity': '3.14'})
                ET.SubElement(joint, 'dynamics', attrib={'damping': str(j.damping)})
                # if i == 0:
                #     pname = parent_link_name
                #     cname = self.name + '_j1'
                # elif i == 1:
                #     pname = self.name + '_j1'
                #     cname = self.name + '_j2'
                # else:
                #     pname = self.name + '_j2'
                #     cname = self.name
                ET.SubElement(joint, 'parent', attrib={'link': parent_link_name})
                ET.SubElement(joint, 'child', attrib={'link': self.name})

            marker = ET.SubElement(link, 'visual')
            geom = ET.SubElement(marker, 'geometry')
            ET.SubElement(geom, 'sphere', attrib={'radius': '0.1'})
            mat = ET.SubElement(marker, 'material', attrib={'name':'marker'})
            ET.SubElement(mat, 'color', attrib={'rgba': '1.0 1.0 1.0 1.0'})

        for body in self.bodies:
            if body.name is None:
                body.name = name_manager.get_name('body')
            body.to_urdf(xml, name_manager, self.name)


if __name__ == '__main__':
    from xml_parser.joints import Joint
    from dataclasses import dataclass
    from typing import List

    @dataclass
    class AntCapsule(Capsule):
        size: float = 0.08
        density: float = 5
        friction: List[float] = (1.5, 0.1, 0.1)

    @dataclass
    class AntJoint(Joint):
        armature: float = 0.01
        damping: float = 0.1
        gear: float = 15
        stiffness: float = 0

    model = XMLModel()
    torso = model.root
    torso.add_geom(Sphere(pos=[0,0,0], size=0.25, density=5, friction=(1.5, 0.1, 0.1)))
    torso.add_geom(AntCapsule(p1=[0,0,0], p2=[0.2, 0.2, 0]))
    torso.add_geom(AntCapsule(p1=[0,0,0], p2=[-0.2, 0.2, 0]))
    torso.add_geom(AntCapsule(p1=[0,0,0], p2=[-0.2, -0.2, 0]))
    torso.add_geom(AntCapsule(p1=[0,0,0], p2=[0.2, -0.2, 0]))

    leg = torso.add_body(pos=[0.2,0.2,0.0])
    leg.add_joint(AntJoint(pos=[0, 0, 0], axis=[0, 0, 1], limits=[-40, 40]))
    leg.add_geom(AntCapsule([0.0, 0.0, 0.0], [0.2, 0.2, 0.0]))
    foot = leg.add_body(pos=[0.2, 0.2, 0.0])
    foot.add_joint(AntJoint(pos=[0, 0, 0], axis=[-1, 1, 0], limits=[30, 100]))
    foot.add_geom(AntCapsule([0.0, 0.0, 0.0], [0.4, 0.4, 0.0]))

    leg = torso.add_body(pos=[-0.2,0.2,0.0])
    leg.add_joint(AntJoint(pos=[0, 0, 0], axis=[0, 0, 1], limits=[-40, 40]))
    leg.add_geom(AntCapsule([0.0, 0.0, 0.0], [-0.2, 0.2, 0.0]))
    foot = leg.add_body(pos=[-0.2, 0.2, 0.0])
    foot.add_joint(AntJoint(pos=[0, 0, 0], axis=[1, 1, 0], limits=[-100, -30]))
    foot.add_geom(AntCapsule([0.0, 0.0, 0.0], [-0.4, 0.4, 0.0]))

    leg = torso.add_body(pos=[-0.2,-0.2,0.0])
    leg.add_joint(AntJoint(pos=[0, 0, 0], axis=[0, 0, 1], limits=[-40, 40]))
    leg.add_geom(AntCapsule([0.0, 0.0, 0.0], [-0.2, -0.2, 0.0]))
    foot = leg.add_body(pos=[-0.2, -0.2, 0.0])
    foot.add_joint(AntJoint(pos=[0, 0, 0], axis=[-1, 1, 0], limits=[-100, -30]))
    foot.add_geom(AntCapsule([0.0, 0.0, 0.0], [-0.4, -0.4, 0.0]))

    leg = torso.add_body(pos=[0.2,-0.2,0.0])
    leg.add_joint(AntJoint(pos=[0, 0, 0], axis=[0, 0, 1], limits=[-40, 40]))
    leg.add_geom(AntCapsule([0.0, 0.0, 0.0], [0.2, -0.2, 0.0]))
    foot = leg.add_body(pos=[0.2, -0.2, 0.0])
    foot.add_joint(AntJoint(pos=[0, 0, 0], axis=[1, 1, 0], limits=[30, 100]))
    foot.add_geom(AntCapsule([0.0, 0.0, 0.0], [0.4, -0.4, 0.0]))
    model.adjust_root()
    model.write('../test/xmls/ant2.xml')

    model = XMLModel.from_path('../test/xmls/robogrammar0.xml')
    model.adjust_root()
    model.write('../test/xmls/nv_humanoid.xml')
