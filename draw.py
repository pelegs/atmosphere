import svgwrite
from subprocess import call

def html2rgb(code):
    """
    Converts html color code
    to an RGB triplet for pygame
    """
    R = int(code[:2], 16)
    G = int(code[2:4], 16)
    B = int(code[4:], 16)
    return (
     R, G, B)


def create_image(col, rad):
    name = 'images/{}_{}'.format(rad, col)
    radius = int(rad)
    s = '{}px'.format(radius)
    dwg = svgwrite.Drawing(filename='{}.svg'.format(name), size=(
     s, s))
    R, G, B = html2rgb(col)
    color = svgwrite.rgb(R, G, B)
    grad = dwg.defs.add(dwg.radialGradient())
    grad.add_stop_color(0, 'white')
    grad.add_stop_color(0.85, color)
    grad.add_stop_color(1, 'black')
    rect = dwg.rect((0, 0), (
     rad, rad), fill='black')
    circle = dwg.circle(center=(radius // 2, radius // 2), r=radius // 2, fill=grad.get_paint_server())
    dwg.add(rect)
    dwg.add(circle)
    dwg.save()
    call('convert -density 75 -resize {}x{} -transparent black {}.svg {}.png'.format(radius, radius, name, name), shell=True)


if __name__ == '__main__':
    create_image('0095FF', 100)
