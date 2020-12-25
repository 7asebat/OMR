
class BaseComponent:
    def sort_x_key(foo): return foo.x

    def __init__(self, box):
        self.x = box[0]
        self.y = box[2]
        self.pos = (box[0], box[2])
        self.width = box[1] - box[0]
        self.height = box[3] - box[2]

    def get_ar(self):
        return self.width/self.height

    def __repr__(self):
        return "<BaseComponent pos = (%i, %i), dim = (%i, %i)>" % (self.x, self.y, self.width, self.height)
