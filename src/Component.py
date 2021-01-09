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
        return ''

    def get_details(self):
        return f'<Accidental {self.pos} kind: {self.kind}>'


class Note(BaseComponent):
    def __init__(self, box):
        super().__init__(box)
        self.timing = None
        self.tone = None
        self.beamed = False
        self.filled = False
        self.accidental = ''
        self.artdots = ''

    def __repr__(self):
        t1 = 'None' if self.tone is None else self.tone[0]
        t2 = '' if self.tone is None else self.tone[1:]

        return f'{t1}{self.accidental}{t2}/{self.timing}{self.artdots}'

    def get_details(self):
        return f'<Note {self.pos} beamed: {self.beamed}, filled: {self.filled}, timing: {self.timing}, tone: {self.tone}>'


class Meter(BaseComponent):
    def __init__(self, box):
        super().__init__(box)
        self.time = None

    def __repr__(self):
        signature = self.time.replace('_', '/')[1:]
        return f'\meter<"{signature}">'

    def get_details(self):
        return f'<Meter {self.pos} time: {self.time}>'


class Chord(BaseComponent):
    def __init__(self, box):
        super().__init__(box)
        self.timing = None
        self.tones = None
        self.filled = False

    def __repr__(self):
        l = map(lambda x: f'{x}/{self.timing}', self.tones)
        s = ','.join(list(l))
        return f'{{{s}}}'

    def get_details(self):
        return f'<Note {self.pos} filled: {self.filled}, timing: {self.timing}, tones: {self.tones}>'
