# testing connectivity Python - Node
import subprocess
import threading
import time

can_break = False

args = ['node', 'simple-echo.js']

DETACHED_PROCESS = 0x00000008


popen = subprocess.Popen(args, shell=False, stdout=subprocess.PIPE, stdin=subprocess.PIPE) # , creationflags=DETACHED_PROCESS

#popen = subprocess.Popen(args, shell=True) # , creationflags=DETACHED_PROCESS

i = 0


try:    
    while True:        
        i = i + 1
        #print('%i Main thread ...' % i)
        msg = "%dtest\n" % i
        popen.stdin.write(msg.encode('utf-8'))
        popen.stdin.flush()
        try:
            print(popen.stdout.readline())
        except:
            print("No data %d" % i)            

        time.sleep(1)
except KeyboardInterrupt:
    can_break = True