from redis import Redis
from rq import Worker


redis = Redis()
workers = Worker.all(connection=redis)
for worker in workers:
    worker.register_death()
