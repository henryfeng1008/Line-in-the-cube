import time
def now():
    start_time = time.localtime()
    print("{}-{}-{}: {}:{}:{}".format(start_time[0],start_time[1],start_time[2],start_time[3],start_time[4],start_time[5]))
