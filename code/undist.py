import cv2
def fetch_calib_argument():
    import pickle
    file_path_camera_calib = "/Users/chuanhl/Project/CarND-Advanced-Lane-Lines/camera_cal/mtx_dist_pickle.p"
    with open(file_path_camera_calib, 'rb') as f:
        camera_lib = pickle.load(f)

    mtx = camera_lib['mtx']
    dist = camera_lib['dist']

    return mtx, dist

def undist(image):
    mtx, dist = fetch_calib_argument()
    undist_img = cv2.undistort(image, mtx, dist, None, mtx)
    return undist_img, mtx, dist
