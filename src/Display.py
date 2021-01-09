import matplotlib.pyplot as plt
import numpy as np


def show_images(images, titles=None, windowTitle=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    if windowTitle:
        fig.canvas.set_window_title(windowTitle)
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        a.xaxis.set_ticks([])
        a.yaxis.set_ticks([])
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    fig.tight_layout()
    plt.show()


def show_images_rows(images, titles=None, windowTitle=None):
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    if windowTitle:
        fig.canvas.set_window_title(windowTitle)
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(round(n_ims/2), 2, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        a.xaxis.set_ticks([])
        a.yaxis.set_ticks([])
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    fig.tight_layout()
    plt.show()


def show_images_columns(images, titles=None, windowTitle=None):
    n_ims = len(images)
    if not titles:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    if windowTitle:
        fig.canvas.set_window_title(windowTitle)
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(2, round(n_ims/2), n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        a.xaxis.set_ticks([])
        a.yaxis.set_ticks([])
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    fig.tight_layout()
    plt.show()


def get_guido_notation(components):
    guido = []
    for cmp in components:
        if str(cmp):
            guido.append(f'{cmp}')
    guidostring = ' '.join(guido)

    return f'[{guidostring}]'
