import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__))
import oilchange_scene

if __name__ == '__main__':
    # hack to accept negative numbers in arguments
    parser = argparse.ArgumentParser(prefix_chars='@')
    parser.add_argument('model_path')
    parser.add_argument('pose_list_path')
    parser.add_argument('subset_start_index', type=int)
    parser.add_argument('subset_size', type=int)
    parser.add_argument('output_dir')
    parser.add_argument('width', type=int)
    parser.add_argument('height', type=int)
    parser.add_argument('f_x', type=float)
    parser.add_argument('f_y', type=float)
    parser.add_argument('p_x', type=float)
    parser.add_argument('p_y', type=float)
    parser.add_argument('camera_scale', type=float)

    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    args = parser.parse_args(argv)

    output_dir = os.path
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    camera_parameters = {
        'width': args.width,
        'height': args.height,
        'f_x': args.f_x,
        'f_y': args.f_y,
        'p_x': args.p_x,
        'p_y': args.p_y
    }

    oilchange_scene.init(args.model_path, camera_parameters, camera_scale=args.camera_scale)
    poses = np.loadtxt(args.pose_list_path)
    for i in range(args.subset_start_index, args.subset_start_index + args.subset_size):
        position, orientation = poses[i][:3], poses[i][3:]
        oilchange_scene.set_object_pose(position, orientation)
        oilchange_scene.set_mode_object()
        oilchange_scene.render(os.path.join(args.output_dir, 'object_{:08}.png'.format(i + 1)))
        oilchange_scene.set_mode_mask()
        oilchange_scene.render(os.path.join(args.output_dir, 'mask_{:08}.png'.format(i + 1)))
