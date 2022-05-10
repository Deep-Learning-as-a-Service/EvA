import os
import neat
from model_representation.ModelGenome.ModelGenome import ModelGenome
import utils.nas_settings as nas_settings
from model_representation.ParametrizedLayer.PDenseLayer import PDenseLayer
from model_representation.EvoParam.IntEvoParam import IntEvoParam
from model_representation.LayerManager.LayerManager import LayerMapper
from tensorflow import keras
from model_representation.ModelGenome.ModelNode.ModelNode import ModelNode
from typing import Union
from visualization.visualize_neat_genome import visualize_neat_genome
import drawSvg as draw


class Point():
        def __init__(self, x, y):
            self.x = x
            self.y = y
        
        def __str__(self):
            return "(" + str(self.x) + ", " + str(self.y) + ")"
        
        def __add__(self, other):
            return Point(self.x + other.x, self.y + other.y)
        
        def __sub__(self, other):
            return Point(self.x - other.x, self.y - other.y)

class SvgNode():
    def __init__(self, parent: 'Union[None, SvgNode]', model_node: 'ModelNode', nodes_already_created: 'list[SvgNode]'):
        self.model_node = model_node
        self.parents = [parent] if parent is not None else []
        self.childs = []
        self.position = None
        nodes_already_created.append(self)

        model_nodes_already_created = list(map(lambda svg_node: svg_node.model_node, nodes_already_created))
        for model_node_child in model_node.childs:
            if model_node_child in model_nodes_already_created:
                svg_child: SvgNode = next((svg_node for svg_node in nodes_already_created if svg_node.model_node == model_node_child), None)
                svg_child.add_parent(self)
                
            else:
                svg_child = SvgNode(
                        parent=self,
                        model_node=model_node_child,
                        nodes_already_created=nodes_already_created
                    )
            self.childs.append(svg_child)
    
    def add_parent(self, parent):
        self.parents.append(parent)

def calculate_visualization_layers(svg_input_node):

    # add position parameter to all nodes
    flatten = lambda list_of_lists: [item for sublist in list_of_lists for item in sublist]
    layers = [[svg_input_node]]
    childs_left = lambda svg_nodes_of_layer: False if 0 == sum(list(map(lambda svg_node: len(svg_node.childs), svg_nodes_of_layer))) else True
    node_in_layers = lambda svg_node: svg_node in flatten(layers)

    def all_parents_already_in_layers(svg_node: 'SvgNode', layers):
        finished_nodes = flatten(layers)
        for parent in svg_node.parents:
            if parent not in finished_nodes:
                return False
        return True
    
    # breadth first search
    while childs_left(layers[-1]):
        current_layer_childs: 'list[SvgNode]' = flatten([svg_node.childs for svg_node in layers[-1]])
        layers.append([])
        # which of the current_layer_childs will go to the next layer
        for current_layer_child in current_layer_childs:
            if all_parents_already_in_layers(current_layer_child, layers) and not node_in_layers(current_layer_child):
                layers[-1].append(current_layer_child)
    
    return layers

def add_visualization_positions(layers, width, height, node_radius, x_distance_between):
    bottom_top_offset = 30
    layer_height = len(layers) * node_radius * 2
    layer_distance_between = (height - layer_height - (2 * bottom_top_offset)) / (len(layers) - 1)
    x_distance_nodes = node_radius * 2 + x_distance_between

    current_y = height / 2 # we start from top middle and go down
    current_y -= bottom_top_offset
    for svg_nodes in layers:
        current_y -= node_radius
        n_svg_nodes = len(svg_nodes)

        x_values = []
        if n_svg_nodes == 1:
            x_values = [0]
        else:
            start_x = None
            if n_svg_nodes % 2 == 0:
                start_x = x_distance_nodes / 2
            else:
                start_x = 0
            for i in range(n_svg_nodes // 2):
                x_values += set([start_x, -start_x])
                start_x += x_distance_nodes 
        
        x_values = sorted(x_values)
        
        for i, svg_node in enumerate(svg_nodes):
            svg_node.position = Point(x=x_values[i], y=current_y)

        current_y -= node_radius + layer_distance_between # last current_y is not used
    
        

def visualize_from_svg_node(layers: 'list[list[SvgNode]]', width, height, node_radius, path):

    def add_text_circle(svg, point, in_text):
        text_offset = node_radius / 2
        svg.append(draw.Circle(cx=point.x, cy=point.y, r=node_radius, fill='#BCCDB9', stroke_width=0, stroke='black')) # bottom left
        svg.append(draw.Text(text=str(in_text), x=point.x - text_offset, y=point.y, fill='black', fontSize=5))

    def add_line(svg, start_point, end_point):
        svg.append(draw.Line(sx=start_point.x, sy=start_point.y, ex=end_point.x, ey=end_point.y, stroke_width=1, stroke='black'))

    def add_arrow(svg, start_point, end_point):
        arrow = draw.Marker(-0.1, -0.5, 0.9, 0.5, scale=5, orient='auto')
        arrow.append(draw.Lines(-0.1, -0.5, -0.1, 0.5, 0.9, 0, fill='black', close=True))

        svg.append(draw.Line(sx=start_point.x, sy=start_point.y, ex=end_point.x, ey=end_point.y,
                    stroke='black', stroke_width=1, fill='none',
                    marker_end=arrow))

    # orientation points
    left_upper_point = Point(x=-width/2, y=-height/2)
    right_lower_point = Point(x=width/2, y=height/2)
    left_middle_point = Point(x=-width/2, y=0)
    right_middle_point = Point(x=width/2, y=0)
    center_point = Point(x=0, y=0)

    # Drawing, Background
    svg = draw.Drawing(width, height, origin='center', displayInline=False)
    svg.append(draw.Rectangle(x=left_upper_point.x, y=left_upper_point.y, width='100%', height='100%', fill='white'))

    for svg_nodes_in_layer in layers:
        for svg_node in svg_nodes_in_layer:
            text = svg_node.model_node.layer.__class__.__name__
            add_text_circle(svg=svg, point=svg_node.position, in_text=text)
            for parent in svg_node.parents:
                add_arrow(svg=svg, start_point=parent.position - Point(x=0, y=node_radius), end_point=svg_node.position + Point(x=0, y=node_radius))

    svg.setPixelScale(8)
    svg.savePng(path + '.png')


def visualize_model_genome(model_genome: 'ModelGenome', path: str):

    # Wrap model graph in SVG Nodes
    nodes_already_created = []
    svg_input_node = SvgNode(
        parent=None,
        model_node=model_genome.get_input_model_node(), 
        nodes_already_created=nodes_already_created
    ) # create svg nodes recursive

    width = 500
    height = 500
    node_radius = 20
    x_distance_between = 20

    layers = calculate_visualization_layers(svg_input_node)

    add_visualization_positions(layers=layers, width=width, height=height, node_radius=node_radius, x_distance_between=x_distance_between)

    visualize_from_svg_node(layers=layers, width=width, height=height, node_radius=node_radius, path=path)



