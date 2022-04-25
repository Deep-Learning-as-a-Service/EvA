import cairosvg
import os

def svg_to_png(svg_string, path_to_file):
    path_to_file += '.png'
    cairosvg.svg2png(bytestring=svg_string,write_to=path_to_file)
