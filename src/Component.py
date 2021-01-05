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
        self.slice = (slice(self.y, self.y + self.height),
                      slice(self.x, self.x + self.width))

    def __repr__(self):
        return f'<BaseComponent {self.pos}>'


class Accidental(BaseComponent):
    def __init__(self, box):
        super().__init__(box)
        self.kind = None

    def __repr__(self):
        return f'<Accidental {self.pos} kind: {self.kind}>'


class Note(BaseComponent):
    def __init__(self, box):
        super().__init__(box)
        self.timing = None
        self.tone = None
        self.beamed = False
        self.filled = False

    def __repr__(self):
        return f'<Note {self.pos} beamed: {self.beamed}, filled: {self.filled}, timing: {self.timing}, tone: {self.tone}>'


class Meter(BaseComponent):
    def __init__(self, box):
        super().__init__(box)
        self.time = None

    def __repr__(self):
        return f'<Meter {self.pos} time: {self.time}>'
