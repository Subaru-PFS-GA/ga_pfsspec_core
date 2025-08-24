class Entry():
    def __init__(self, timestamp, message, data=None):

        self.timestamp = timestamp
        self.message = message
        self.data = data

    def copy(self):
        return Entry(self.timestamp, self.message, data=self.data)