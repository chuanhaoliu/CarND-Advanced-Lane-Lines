import numpy as np
import cv2

def calculate_vertices(image, height_percentage=0.5,
                       top_width_percentage=0.04,
                       bottom_width_percentage=0.90):
    """
    Calculate vertices for a polygon
    """

    imshape = image.shape
    top_y = int(imshape[0] * (1 - height_percentage))
    top_left_x = int(imshape[1] * (1 - top_width_percentage) / 2)
    top_right_x = int(imshape[1] - imshape[1] * (1 - top_width_percentage) / 2)
    bottom_y = int(imshape[0])
    bottom_left_x = int(imshape[1] * (1 - bottom_width_percentage) / 2)
    bottom_right_x = int(imshape[1] - imshape[1] * (1 - bottom_width_percentage) / 2)
    vertices = np.array([[(bottom_left_x, bottom_y),
                          (top_left_x, top_y),
                          (top_right_x, top_y),
                          (bottom_right_x, bottom_y)]],
                        dtype=np.int32)
    return vertices


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.

    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def apply_perspective_transform(image):
    # hard-code src and dst coordinates for caculating the transform matrix
    src = np.float32(
        [[585, 460],
         [203, 720],
         [1127, 720],
         [695, 460]]
    )
    dst = np.float32(
        [[320, 0],
         [320, 720],
         [960, 720],
         [960, 0]]
    )

    # Get perspective transform matrix and also the reverse one
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Apply perspective transform
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    return warped, M, Minv

def perspective_transform(image):
    vertices = calculate_vertices(image)
    masked_edges_img = region_of_interest(image, vertices)
    warped_img, perspective_M, perspective_Minv = apply_perspective_transform(masked_edges_img)

    return warped_img, perspective_M, perspective_Minv