import argparse
import socket
from multiprocessing import Process

from rq import Connection, Queue, Worker
from redis import Redis

def run_worker(queue, name):
    Worker(queue, name=name).work()

parser = argparse.ArgumentParser()
parser.add_argument('--redis_host', default='localhost')
parser.add_argument('--redis_port', type=int, default=6379)
parser.add_argument('--queue_name', default='render_queue')
parser.add_argument('--num_workers', type=int, default=8)

args = parser.parse_args()
hostname = socket.gethostname()
with Connection(Redis(args.redis_host, args.redis_port)):
    q = Queue(args.queue_name)
    for i in range(args.num_workers):
        name = '{}__{}.{}'.format(hostname, args.queue_name, i + 1)
        Process(target=run_worker, args=(q, name)).start()
