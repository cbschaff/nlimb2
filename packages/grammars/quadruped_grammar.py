"""
This grammar is for simple quadruped robots.
It gives robots two body links oriented in the x direction.
Each link has a mirrored set of limbs with atmost 3 links.
"""

import grammars.base as grammar
from grammars import transforms
from utils import Transform
from xml_parser import XMLModel, geoms, joints
from scipy.spatial.transform import Rotation
import numpy as np
import enum


class Symbols(grammar.SymbolSet):
    START = enum.auto()
    BODY_JOINT = enum.auto()
    BODY_PART = enum.auto()
    SHOULDER_JOINT = enum.auto()
    LIMB_JOINT = enum.auto()
    LIMB_PART = enum.auto()
    FIRST_LIMB_PART = enum.auto()
    SECOND_LIMB_PART = enum.auto()
    THIRD_LIMB_PART = enum.auto()

    BIG_BODY_LINK = enum.auto()
    SMALL_BODY_LINK = enum.auto()
    BIG_LIMB_LINK = enum.auto()
    SMALL_LIMB_LINK = enum.auto()

    BODY_ROLL_JOINT = enum.auto()
    BODY_PITCH_JOINT = enum.auto()
    BODY_YAW_JOINT = enum.auto()
    BODY_SPHERE_JOINT = enum.auto()

    SHOULDER_SPHERE_JOINT = enum.auto()
    SHOULDER_YAW_JOINT = enum.auto()

    LIMB_PITCH_JOINT = enum.auto()
    LIMB_ELBOW_JOINT = enum.auto()
    LIMB_KNEE_JOINT = enum.auto()
    LIMB_YAW_JOINT = enum.auto()


def init_quadroped_grammar_graph():
    g = grammar.Tree()
    g.add_node(grammar.NodeData(Symbols.START))
    return g

big_body_capsule = geoms.Capsule([-0.40, 0.0, 0.0], [0.0, 0.0, 0.0], 0.045)
big_mount_capsule = geoms.Capsule([-0.2, -0.2, 0.0], [-0.2, 0.2, 0.0], 0.035)
small_body_capsule = geoms.Capsule([-0.30, 0.0, 0.0], [0.0, 0.0, 0.0], 0.045)
small_mount_capsule = geoms.Capsule([-0.15, -0.2, 0.0], [-0.15, 0.2, 0.0], 0.035)
big_link_capsule = geoms.Capsule([-0.20, 0.0, 0.0], [0.0, 0.0, 0.0], 0.035)
small_link_capsule = geoms.Capsule([-0.15, 0.0, 0.0], [0.0, 0.0, 0.0], 0.035)


def make_quadroped_grammar_rule_set(hexapod_rule=False):
    rules = grammar.RuleSet()

    # START: BODY_PART -BODY_JOINT> BODY_PART
    g = grammar.Tree()
    g.add_node(grammar.NodeData(Symbols.BODY_PART))
    g.add_node(grammar.NodeData(Symbols.BODY_PART))
    g.add_edge(0, 1, grammar.EdgeData(Symbols.BODY_JOINT, transforms.translate_origin_to_p1(0)))
    rules.add_rule(grammar.NodeExpansion(Symbols.START, g,
                                         'START: BODY_PART --BODY_JOINT-> BODY_PART'))


    if hexapod_rule:
        # START: BODY_PART -BODY_JOINT> BODY_PART -BODY_JOINT-> BODY_PART
        g = grammar.Tree()
        g.add_node(grammar.NodeData(Symbols.BODY_PART))
        g.add_node(grammar.NodeData(Symbols.BODY_PART))
        g.add_node(grammar.NodeData(Symbols.BODY_PART))
        g.add_edge(0, 1, grammar.EdgeData(Symbols.BODY_JOINT, transforms.translate_origin_to_p1(0)))
        g.add_edge(1, 2, grammar.EdgeData(Symbols.BODY_JOINT, transforms.translate_origin_to_p1(0)))
        rules.add_rule(grammar.NodeExpansion(Symbols.START, g,
                                             'START: BODY_PART --BODY_JOINT-> BODY_PART'))


    # BODY_PART: BIG_BODY_LINK --2xSHOULDER_JOINT-> FIRST_LIMB_PART
    g = grammar.Tree()
    g.add_node(grammar.NodeData(Symbols.BIG_BODY_LINK, [big_body_capsule, big_mount_capsule]))
    g.add_node(grammar.NodeData(Symbols.FIRST_LIMB_PART))
    right_mount_transform = Transform(pos=np.zeros(3,),
                                      quat=Rotation.from_rotvec([0, 0, np.pi/2]).as_quat())
    mount_transform = transforms.compose(
        transforms.apply(right_mount_transform),
        transforms.translate_origin_to_p1(1)
    )
    g.add_edge(0, 1, grammar.EdgeData(Symbols.SHOULDER_JOINT, mount_transform, mirror_y=True))
    rules.add_rule(grammar.NodeExpansion(
        Symbols.BODY_PART, g, 'B: BIG_BODY_LINK --2xSHOULDER_JOINT-> FIRST_LIMB_PART'
    ))


    # BODY_PART: SMALL_BODY_LINK --2xSHOULDER_JOINT-> FIRST_LIMB_PART
    g = grammar.Tree()
    g.add_node(grammar.NodeData(Symbols.SMALL_BODY_LINK, [small_body_capsule, small_mount_capsule]))
    g.add_node(grammar.NodeData(Symbols.FIRST_LIMB_PART))
    right_mount_transform = Transform(pos=np.zeros(3,),
                                      quat=Rotation.from_rotvec([0, 0, np.pi/2]).as_quat())
    mount_transform = transforms.compose(
        transforms.apply(right_mount_transform),
        transforms.translate_origin_to_p1(1)
    )
    g.add_edge(0, 1, grammar.EdgeData(Symbols.SHOULDER_JOINT, mount_transform, mirror_y=True))
    rules.add_rule(grammar.NodeExpansion(
        Symbols.BODY_PART, g, 'B: SMALL_BODY_LINK --2xSHOULDER_JOINT-> FIRST_LIMB_PART'
    ))


    # BODY_JOINT: BODY_ROLL_JOINT
    data = grammar.EdgeData(Symbols.BODY_ROLL_JOINT,
                            joint=[joints.MediumPowerJoint(axis=[1., 0., 0.], limits=[-90., 90.])])
    rules.add_rule(grammar.EdgeExpansion(Symbols.BODY_JOINT, data, 'BODY_JOINT: BODY_ROLL_JOINT'))


    # BODY_JOINT: BODY_PITCH_JOINT
    data = grammar.EdgeData(Symbols.BODY_PITCH_JOINT,
                            joint=[joints.MediumPowerJoint(axis=[0., 1., 0.], limits=[-90., 90.])])
    rules.add_rule(grammar.EdgeExpansion(Symbols.BODY_JOINT, data, 'BODY_JOINT: BODY_PITCH_JOINT'))


    # BODY_JOINT: BODY_YAW_JOINT
    data = grammar.EdgeData(Symbols.BODY_YAW_JOINT,
                            joint=[joints.MediumPowerJoint(axis=[0., 0., 1.], limits=[-90., 90.])])
    rules.add_rule(grammar.EdgeExpansion(Symbols.BODY_JOINT, data, 'BODY_JOINT: BODY_YAW_JOINT'))


    # BODY_JOINT: BODY_SPHERE_JOINT
    data = grammar.EdgeData(Symbols.BODY_SPHERE_JOINT,
                            joint=[joints.MediumJoint(axis=[1., 0., 0.], limits=[-60., 60.]),
                                   joints.MediumJoint(axis=[0., 1., 0.], limits=[-60., 60.]),
                                   joints.MediumJoint(axis=[0., 0., 1.], limits=[-60., 60.])])
    rules.add_rule(grammar.EdgeExpansion(Symbols.BODY_JOINT, data, 'BODY_JOINT: BODY_SPHERE_JOINT'))


    # SHOULDER_JOINT: SHOULDER_SPHERE_JOINT
    init_pose = Transform(pos=np.zeros(3,),
                          quat=Rotation.from_rotvec([0., -0.5*np.pi, 0.]).as_quat())
    data = grammar.EdgeData(Symbols.SHOULDER_SPHERE_JOINT,
                            joint=[joints.SmallJoint(axis=[1., 0., 0.], limits=[-45., 45.]),
                                   joints.SmallJoint(axis=[0., 1., 0.], limits=[-60., 60.]),
                                   joints.SmallJoint(axis=[0., 0., 1.], limits=[-60., 60.])],
                            transform=transforms.apply(init_pose))
    rules.add_rule(grammar.EdgeExpansion(Symbols.SHOULDER_JOINT, data,
                                         'SHOULDER_JOINT: SHOULDER_SPHERE_JOINT'))


    # SHOULDER_JOINT: SHOULDER_YAW_JOINT
    init_pose = Transform(pos=np.zeros(3,),
                          quat=Rotation.from_rotvec([0., -0.5*np.pi, 0.]).as_quat())
    data = grammar.EdgeData(Symbols.SHOULDER_YAW_JOINT,
                            joint=[joints.MediumJoint(axis=[0., 0., 1.], limits=[-90., 90.])],
                            transform=transforms.apply(init_pose))
    rules.add_rule(grammar.EdgeExpansion(Symbols.SHOULDER_JOINT, data,
                                         'SHOULDER_JOINT: SHOULDER_YAW_JOINT'))


    # FIRST_LIMB_PART: LIMB_PART --LIMB_JOINT-> SECOND_LIMB_PART
    g = grammar.Tree()
    g.add_node(grammar.NodeData(Symbols.LIMB_PART))
    g.add_node(grammar.NodeData(Symbols.SECOND_LIMB_PART))
    g.add_edge(0, 1, grammar.EdgeData(Symbols.LIMB_JOINT, transforms.translate_origin_to_p1(0)))
    rules.add_rule(grammar.NodeExpansion(Symbols.FIRST_LIMB_PART, g,
                                    'FIRST_LIMB_PART: LIMB_PART --LIMB_JOINT-> SECOND_LIMB_PART'))

    # # SECOND_LIMB_PART: LIMB_PART --LIMB_JOINT-> THIRD_LIMB_PART
    # g = grammar.Tree()
    # g.add_node(grammar.NodeData(Symbols.LIMB_PART))
    # g.add_node(grammar.NodeData(Symbols.THIRD_LIMB_PART))
    # g.add_edge(0, 1, grammar.EdgeData(Symbols.LIMB_JOINT, transforms.translate_origin_to_p1(0)))
    # rules.add_rule(grammar.NodeExpansion(Symbols.SECOND_LIMB_PART, g,
    #                                 'SECOND_LIMB_PART: LIMB_PART --LIMB_JOINT-> THIRD_LIMB_PART'))


    # FIRST_LIMB_PART: LIMB_PART
    g = grammar.Tree()
    g.add_node(grammar.NodeData(Symbols.LIMB_PART))
    rules.add_rule(grammar.NodeExpansion(Symbols.FIRST_LIMB_PART, g, 'FIRST_LIMB_PART: LIMB_PART'))


    # SECOND_LIMB_PART: LIMB_PART
    g = grammar.Tree()
    g.add_node(grammar.NodeData(Symbols.LIMB_PART))
    rules.add_rule(grammar.NodeExpansion(Symbols.SECOND_LIMB_PART, g, 'SECOND_LIMB_PART: LIMB_PART'))


    # # THIRD_LIMB_PART: LIMB_PART
    # g = grammar.Tree()
    # g.add_node(grammar.NodeData(Symbols.LIMB_PART))
    # rules.add_rule(grammar.NodeExpansion(Symbols.THIRD_LIMB_PART, g, 'THIRD_LIMB_PART: LIMB_PART'))

    # LIMB_JOINT: LIMB_PITCH_JOINT
    data = grammar.EdgeData(Symbols.LIMB_PITCH_JOINT,
                            joint=[joints.SmallJoint(axis=[0., 1., 0.], limits=[-90., 90.])])
    rules.add_rule(grammar.EdgeExpansion(Symbols.LIMB_JOINT, data, 'LIMB_JOINT: LIMB_PITCH_JOINT'))


    # LIMB_JOINT: LIMB_ELBOW_JOINT
    data = grammar.EdgeData(Symbols.LIMB_ELBOW_JOINT,
                            joint=[joints.SmallJoint(axis=[0., 0., 1.], limits=[-180., 0.])])
    rules.add_rule(grammar.EdgeExpansion(Symbols.LIMB_JOINT, data, 'LIMB_JOINT: LIMB_ELBOW_JOINT'))


    # LIMB_JOINT: LIMB_KNEE_JOINT
    data = grammar.EdgeData(Symbols.LIMB_KNEE_JOINT,
                            joint=[joints.SmallJoint(axis=[0., 0., 1.], limits=[0., 180.])])
    rules.add_rule(grammar.EdgeExpansion(Symbols.LIMB_JOINT, data, 'LIMB_JOINT: LIMB_KNEE_JOINT'))



    # LIMB_JOINT: LIMB_YAW_JOINT
    data = grammar.EdgeData(Symbols.LIMB_YAW_JOINT,
                            joint=[joints.SmallJoint(axis=[0., 0., 1.], limits=[-90., 90.])])
    rules.add_rule(grammar.EdgeExpansion(Symbols.LIMB_JOINT, data, 'LIMB_JOINT: LIMB_YAW_JOINT'))


    # LIMB_PART: BIG_LIMB_LINK
    g = grammar.Tree()
    g.add_node(grammar.NodeData(Symbols.BIG_LIMB_LINK, [big_link_capsule]))
    rules.add_rule(grammar.NodeExpansion(Symbols.LIMB_PART, g, 'LIMB_PART: BIG_LIMB_LINK'))


    # LIMB_PART: SMALL_LIMB_LINK
    g = grammar.Tree()
    g.add_node(grammar.NodeData(Symbols.SMALL_LIMB_LINK, [small_link_capsule]))
    rules.add_rule(grammar.NodeExpansion(Symbols.LIMB_PART, g, 'LIMB_PART: SMALL_LIMB_LINK'))

    return rules


QUADRUPED_GRAMMAR = grammar.Grammar(Symbols, make_quadroped_grammar_rule_set(),
                                    init_quadroped_grammar_graph())

HEXAPOD_GRAMMAR = grammar.Grammar(Symbols, make_quadroped_grammar_rule_set(hexapod_rule=True),
                                  init_quadroped_grammar_graph())

if __name__ == '__main__':
    import random

    # count = 1
    # while count < 16:
    #     g = QUADRUPED_GRAMMAR.sample()
    #     grammar.to_xml(g).write(f'../test/xmls/robogrammar{count}.xml')
    #     count += 1

    graph = QUADRUPED_GRAMMAR.initialize_graph()
    rules = QUADRUPED_GRAMMAR.get_valid_expansions(graph)
    rules[0][0].apply(graph, 0)
    rules = QUADRUPED_GRAMMAR.get_valid_expansions(graph)
    rules[0][0].apply(graph, 0)
    rules[1][0].apply(graph, 1)
    rules = QUADRUPED_GRAMMAR.get_valid_expansions(graph)
    rules[(0, 1)][1].apply(graph, 0, 1)
    rules[(0, 2)][0].apply(graph, 0, 2)
    rules[(1, 3)][1].apply(graph, 1, 3)
    rules = QUADRUPED_GRAMMAR.get_valid_expansions(graph)
    rules[2][0].apply(graph, 2)
    rules[3][0].apply(graph, 3)
    rules = QUADRUPED_GRAMMAR.get_valid_expansions(graph)
    rules[4][0].apply(graph, 4)
    rules[5][0].apply(graph, 5)
    print(rules[(2,4)])
    rules[(2, 4)][0].apply(graph, 2, 4)
    rules[(3, 5)][1].apply(graph, 3, 5)
    rules = QUADRUPED_GRAMMAR.get_valid_expansions(graph)
    rules[2][0].apply(graph, 2)
    rules[3][0].apply(graph, 3)
    rules[4][0].apply(graph, 4)
    rules[5][0].apply(graph, 5)
    grammar.to_xml(graph).write('../test/xmls/grammar0.xml')

