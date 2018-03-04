'''example
hostname, hour, load, memutil, netutil
'''

import random
import string
import os
import csv
import datetime


hostnames = ['abstract', 'merlyn', 'champion', 'tenor', 'bearface', 'ameer', 'dom']


def generate_log():
    start = datetime.datetime(year=2015, month=5, day=24)
    end = datetime.datetime(year=2018, month=2, day=28)
    inbound = random.choice([True, False])
    weird_port = True if (random.randint(0, 10) == 9) else False
    return [
        random.choice(hostnames), #hostnames
        random.randint(0,23), #hour
        random.randint(0,5), #load
        str(random.uniform(0.0,1.0))[:4], #memutil
        str(random.uniform(0.0,1.0))[:4]
        ]

logs = {}
with open('logs.csv', "w") as f:
    writer = csv.writer(f)
    writer.writerow(["hostname", "hour", "load", "memutil", "netutil"])
    for i in range(10000):
        log_list = generate_log()
        writer.writerow(log_list)
