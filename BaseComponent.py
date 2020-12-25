class BaseComponent:
    def sort_x_key(foo): return foo.x

    def __init__(self, box):
        self.x = box[0]
        self.y = box[2]
        self.pos = (box[0], box[2])
        self.width = box[1] - box[0]
        self.height = box[3] - box[2]

    def __repr__(self):
        return f"<BaseComponent pos = ({self.x}, {self.y}), dim = ({self.width}, {self.height})>"

    def get_ar(self):
        return self.width/self.height

    def get_slice(self):
        return (slice(self.y, self.y + self.height), slice(self.x, self.x + self.width))