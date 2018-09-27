from QMinerBridge import QMinerBridge
import time

x = QMinerBridge()

x.sendMsg(u'test')
time.sleep(0.1)
print(x.readMsg())
time.sleep(0.1)
x.stopQMiner()