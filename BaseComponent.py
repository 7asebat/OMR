
class BaseComponent:
    def __init__(self, box):
        self.x = box[0]
        self.y = box[2]
        self.pos = (box[0], box[2])
        self.width = box[1] - box[0]
        self.height = box[3] - box[2]

    def get_ar():
        return self.width/self.height

    def __repr__(self):
        return "<BaseComponent pos = (%i, %i), dim = (%i, %i)>" % (self.x, self.y, self.width, self.height)

# def get_bounding_boxes(image):
#     boundingBoxes = []
#     contours = find_contours(image, 0.8)
#     for c in contours:
#         xValues = np.round(c[:, 1]).astype(int)
#         yValues = np.round(c[:, 0]).astype(int)
#         ar = (xValues.max() - xValues.min()) / (yValues.max() - yValues.min())

#         if 0 <= ar:
#             boundingBoxes.append(
#                 [xValues.min(), xValues.max(), yValues.min(), yValues.max()])

#     return boundingBoxes
