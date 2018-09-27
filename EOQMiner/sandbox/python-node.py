# testing connectivity Python - Node
# imports
import subprocess
import time

# prepare subprocess
args = ['node', 'simple-echo.js']
popen = subprocess.Popen(args, shell=False, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

# initialize the counter
i = 0

# main loop
try:    
    while True:        
        # send next message
        i = i + 1
        msg = "test %d\n" % i
        popen.stdin.write(msg.encode('utf-8'))
        # it is important to make the flush (otherwise nothing is received in NodeJS)
        popen.stdin.flush()
        try:
            print(popen.stdout.readline())
        except:
            print("No data - iteration %d" % i)            
        time.sleep(1)
except KeyboardInterrupt:
    can_break = True