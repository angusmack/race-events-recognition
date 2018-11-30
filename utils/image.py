import matplotlib.pyplot as plt
import numpy as np
import cv2

def show_images(images, cols = 1, titles = None):
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.xticks([]), plt.yticks([])
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.tight_layout()
    plt.show()
    
    
def imprint(frames,values):
    for i,x in enumerate(frames):
        for j,z in enumerate(values):
            dx = 10+15*j
            clr = (255,0,0) if z<0.5 else (0,255,0)
            cv2.rectangle(x,(dx,10),(dx+10,90),clr,1)
            cv2.line(x,(dx+5,90),(dx+5,90-int(80*z)),clr,10)
    return frames