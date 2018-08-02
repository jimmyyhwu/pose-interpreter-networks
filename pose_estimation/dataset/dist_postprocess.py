import argparse
import os

from tqdm import tqdm
from pycocotools.coco import COCO
from rq import Connection, Queue
from redis import Redis

from dist_wrappers import postprocess_wrapper

parser = argparse.ArgumentParser()
parser.add_argument('--oil_change_data_root', default='../../data/OilChangeDataset/', help='location of Oil Change dataset')
parser.add_argument('--ann_file', default='20171103_OilChange.json')
parser.add_argument('--cat_ids', default='1,2,4,5,6', help='comma-separated category ids')
parser.add_argument('--cache_subset_size', type=int, default=100, help='images per directory in cache rendered images')
parser.add_argument('--train_subset_size', type=int, default=6400, help='group train images into train_subset_size images per directory')
parser.add_argument('--process_val', action='store_true', help='whether to generate val set')
parser.set_defaults(process_val=False)  # should download provided val set to match numbers in paper
parser.add_argument('--val_subset_size', type=int, default=640)
parser.add_argument('--num_subsets', type=int, default=10, help='generate num_subsets directories of training images')
parser.add_argument('--pose_lists_dir', default='../data/cache/pose_lists/')
parser.add_argument('--cache_dir', default='../data/cache/images/')
parser.add_argument('--output_base_dir', default='../data/')
parser.add_argument('--camera_name', default='floating_kinect1')
parser.add_argument('--redis_host', default='localhost')
parser.add_argument('--redis_port', type=int, default=6379)
parser.add_argument('--queue_name', default='postprocess_queue')
args = parser.parse_args()

coco = COCO(os.path.join(args.oil_change_data_root, 'annotations', args.ann_file))
cats = {cat['id']: cat for cat in coco.dataset['categories']}

cat_ids = [int(c.strip()) for c in args.cat_ids.split(',')]
for cat_id in tqdm(cat_ids):
    cat = cats[cat_id]
    cat_name = cat['name']
    pose_list_path = os.path.join(args.pose_lists_dir, '{}.txt'.format(cat['name']))
    completed_dir = os.path.join(args.cache_dir, args.camera_name, cat['name'], 'completed')
    image_dir = os.path.join(args.cache_dir, args.camera_name, cat['name'])
    with Connection(Redis(args.redis_host, args.redis_port)):
        q = Queue(args.queue_name)
        q.enqueue(postprocess_wrapper, pose_list_path, completed_dir, image_dir, args.output_base_dir, args.camera_name, cat_name,
                  args.cache_subset_size, args.train_subset_size, args.val_subset_size, args.process_val, args.num_subsets,
                  timeout=3600000, result_ttl=0)
