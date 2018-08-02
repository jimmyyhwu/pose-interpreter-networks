import argparse
import os
import sys


sys.path.append(os.path.dirname(__file__))
from dataset import oilchange_scene


if __name__ == '__main__':
    # hack to accept negative numbers in arguments
    parser = argparse.ArgumentParser(prefix_chars='@')
    parser.add_argument('model_path')
    parser.add_argument('output_path')
    parser.add_argument('mode', help='object or mask')
    parser.add_argument('width', type=int)
    parser.add_argument('height', type=int)
    parser.add_argument('f_x', type=float)
    parser.add_argument('f_y', type=float)
    parser.add_argument('p_x', type=float)
    parser.add_argument('p_y', type=float)
    parser.add_argument('camera_scale', type=float)
    parser.add_argument('position')
    parser.add_argument('orientation')

    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    args = parser.parse_args(argv)

    camera_parameters = {
        'width': args.width,
        'height': args.height,
        'f_x': args.f_x,
        'f_y': args.f_y,
        'p_x': args.p_x,
        'p_y': args.p_y
    }

    oilchange_scene.init(args.model_path, camera_parameters, camera_scale=args.camera_scale)
    position = list(map(float, args.position.split(',')))
    orientation = list(map(float, args.orientation.split(',')))
    oilchange_scene.set_object_pose(position, orientation)
    if args.mode == 'object':
        oilchange_scene.set_mode_object()
    elif args.mode == 'mask':
        oilchange_scene.set_mode_mask()
    oilchange_scene.render(args.output_path)
