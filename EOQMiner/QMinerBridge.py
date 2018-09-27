import subprocess

class QMinerBridge:
    """
    Constructor.
    """
    def __init__(self):
        self.startQMiner()

    """
    Class to build a bridge between NodeJS (QMiner) and Python (eo-learn).
    """
    def startQMiner(self):
        # prepare subprocess
        args = ['node', 'node/index.js']
        self.popen = subprocess.Popen(args, shell=False, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        # TODO: error handling

    def sendMsg(self, msg):
        """
        Send message to the QMiner.
        """
        msg = msg + '\n'
        self.popen.stdin.write(msg.encode('utf-8'))
        # it is important to make the flush (otherwise nothing is received in NodeJS)
        self.popen.stdin.flush()

    def readMsg(self):
        """
        Read message from QMiner.
        """
        try:
            msg = self.popen.stdout.readline()
            return(msg)
        except:
            return

    def stopQMiner(self):
        self.popen.terminate()