import os
import time
import subprocess


def get_times():
    times = {}
    for root, _, files in os.walk("."):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                times[path] = os.path.getmtime(path)
    return times


last = get_times()

while True:
    time.sleep(1)
    current = get_times()

    if current != last:
        os.system("clear")
        subprocess.run(["make", "run"])
        last = current
