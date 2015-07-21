__author__ = 'Daguerreo'

from datetime import datetime
import pickle

class Logger:
    def __init__(self):
        self.startTime = -1
        self.round = -1
        pass

    def timerStart(self):
        self.startTime = datetime.now()
        self.round = datetime.now()
        return

    def timerRound(self):
        now = datetime.now()
        delta = now - self.round
        self.round = now
        return int(delta.total_seconds()*1000)

    def totalTime(self):
        now = datetime.now()
        delta = now - self.startTime
        return int(delta.total_seconds()*1000)

    def save(self, data, filename):
        with open(filename, 'wb') as file:
        # pickle.dump([obj0, obj1, obj2], f)
            pickle.dump(data, file, -1)
        return

    def load(self, filename):
        # Getting back the objects:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)

        return obj