class BaseComponent:
    def sort_x_key(foo): return foo.x

    def __init__(self, box):
        self.box = box
        self.x = box[0]
        self.y = box[2]

        self.width = box[1] - box[0]
        self.height = box[3] - box[2]

        self.pos = (self.y, self.x)
        self.dim = (self.width, self.height)

        self.ar = self.width / self.height
        self.slice = (slice(self.y, self.y + self.height), slice(self.x, self.x + self.width))

    def __repr__(self):
        return f'<BaseComponent pos: {self.pos}, dim: {self.dim}>'


class Accidental(BaseComponent):
    def __init__(self, box):
        super().__init__(box)
        self.kind = None

    def __repr__(self):
        return f'<Accidental pos: {self.pos}, dim: {self.dim}, kind: {self.kind}>'

class Note(BaseComponent):
    def __init__(self, box):
        super().__init__(box)
        self.timing = None
        self.tone = None
        self.beamed = False
        self.filled = False

    def __repr__(self):
        return f'<Note pos: {self.pos}, dim: {self.dim}, beamed: {self.beamed}, filled: {self.filled}, timing: {self.timing}, tone: {self.tone}>'