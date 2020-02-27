from scipy.ndimage import gaussian_filter


def gaussian_smoothing(img):
    return gaussian_filter(img[:, :], sigma=1)


def gradient(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    magnitude = np.hypot(Ix, Iy)
    magnitude = magnitude / magnitude.max() * 255
    direction = np.arctan2(Iy, Ix)
    return (magnitude, direction)


def non_maximum_suppression(magnitude, Direction):
    H, W = magnitude.shape
    Z = np.zeros((H, W), dtype=np.int32)
    angle = Direction * 180.0 / np.pi
    angle[angle < 0] += 180
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            q = 255
            r = 255
            # Gradient directions :
            # angle
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            # angle Pi/4
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            # angle Pi/2
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            # angle 3Pi/4
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]
            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                Z[i, j] = magnitude[i, j]
            else:
                Z[i, j] = 0
    return Z


def double_thresholding(img, thresh_lo, thresh_hi):

    highThreshold = img.max() * thresh_hi
    lowThreshold = highThreshold * thresh_lo

    H, W = img.shape
    res = np.zeros((H, W), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)


def connectivity(img, weak, strong):
    H, W = img.shape
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if img[i, j] == weak:
                if (
                    (img[i + 1, j - 1] == strong)
                    or (img[i + 1, j] == strong)
                    or (img[i + 1, j + 1] == strong)
                    or (img[i, j - 1] == strong)
                    or (img[i, j + 1] == strong)
                    or (img[i - 1, j - 1] == strong)
                    or (img[i - 1, j] == strong)
                    or (img[i - 1, j + 1] == strong)
                ):
                    img[i, j] = strong
                else:
                    img[i, j] = 0

    return img


def canny_edge_detector(img, thresh_lo=0.1, thresh_hi=0.2):
    """
    The Canny edge detector.
    
    Inputs:
        img              The input image
        thresh_lo        The fraction of the maximum gradient magnitude which will 
                         be considered the lo threshold. 
        thresh_hi        The fraction of the maximum gradient magnitude which will
                         be considered the hi threshold. Ideally should be 2x to 3x 
                         thresh_lo.
                         
    Outputs: 
        edge_img         A binary image, with pixels lying on edges marked with a 1, 
                         and others with a 0.
    """

    # Smooth the image first.
    smoothed = gaussian_smoothing(img / 255)

    # Find gradient magnitude and direction
    g_magnitude, g_dir = gradient(smoothed)

    # Non-maximum suppression
    g_max = non_maximum_suppression(g_magnitude, g_dir)

    # Double thresholding
    thresh_img, weak, strong = double_thresholding(g_max, thresh_lo, thresh_hi)

    # Final edge connectivity
    edge_img = connectivity(thresh_img, weak, strong)

    # Return the result
    return edge_img

