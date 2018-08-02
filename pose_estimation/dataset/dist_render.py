import argparse
import os

from pycocotools.coco import COCO
from rq import Connection, Queue
from redis import Redis
from tqdm import tqdm

from dist_wrappers import render_wrapper
from dist_wrappers import clean_wrapper

parser = argparse.ArgumentParser()
parser.add_argument('--oil_change_data_root', default='../../data/OilChangeDataset/', help='location of Oil Change dataset')
parser.add_argument('--ann_file', default='20171103_OilChange.json')
parser.add_argument('--cat_ids', default='1,2,4,5,6', help='comma-separated category ids')
parser.add_argument('--num_examples', type=int, default=70000, help='number of examples per category')
parser.add_argument('--pose_lists_dir', default='../data/cache/pose_lists/')
parser.add_argument('--output_cache_dir', default='../data/cache/images/')
parser.add_argument('--blender_path', default='blender')
parser.add_argument('--camera_name', default='floating_kinect1')
parser.add_argument('--camera_scale', type=float, default=0.5)
parser.add_argument('--subset_size', type=int, default=100, help='group rendered images into directories (subsets) containing subset_size images')
parser.add_argument('--redis_host', default='localhost')
parser.add_argument('--redis_port', type=int, default=6379)
parser.add_argument('--queue_name', default='render_queue')
parser.add_argument('--clean', action='store_true', help='clean up partially rendered subsets and mark as incomplete')
parser.set_defaults(clean=False)
args = parser.parse_args()

coco = COCO(os.path.join(args.oil_change_data_root, 'annotations', args.ann_file))
cats = {cat['id']: cat for cat in coco.dataset['categories']}

cameras = {camera['name']: camera for camera in coco.dataset['cameras']}
camera_info = cameras[args.camera_name]
camera_parameters = {
    'width': camera_info['width'],
    'height': camera_info['height'],
    'f_x': camera_info['K'][0],
    'f_y': camera_info['K'][4],
    'p_x': camera_info['K'][2],
    'p_y': camera_info['K'][5]
}

cat_ids = [int(c.strip()) for c in args.cat_ids.split(',')]
with Connection(Redis(args.redis_host, args.redis_port)):
    q = Queue(args.queue_name)
    for i, subset_start_index in enumerate(tqdm(range(0, args.num_examples, args.subset_size))):
        for cat_id in cat_ids:
            cat = cats[cat_id]
            model_path = os.path.join(args.oil_change_data_root, 'meshes', cat['mesh'])
            completed_dir = os.path.join(args.output_cache_dir, args.camera_name, cat['name'], 'completed')
            if not os.path.exists(completed_dir):
                os.makedirs(completed_dir)
            output_dir = os.path.join(args.output_cache_dir, args.camera_name, cat['name'], 'subset_{:08}'.format(i + 1))
            pose_list_path = os.path.join(args.pose_lists_dir, '{}.txt'.format(cat['name']))
            completed_path = os.path.join(completed_dir, 'subset_{:08}.txt'.format(i + 1))
            if args.clean:
                wrapper = clean_wrapper
            else:
                if os.path.exists(completed_path):
                    continue
                wrapper = render_wrapper
            q.enqueue(wrapper, args.blender_path, model_path,
                      pose_list_path, subset_start_index, args.subset_size, output_dir,
                      camera_parameters, args.camera_scale,
                      completed_path,
                      timeout=36000, result_ttl=0)
