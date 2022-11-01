import torch
import cv2

def letterbox(img,height,width):
    h,w,_ = img.shape
    scale_h = float(height/h)
    scale_w = float(width/w)
    scale = min(scale_h,scale_w)
    fill_value = (127.5,127.5,127.5)
    new_size = round(scale*h),round(scale*w)
    dh,dw = (new_size[0]-h)/2,(new_size[1]-w)/2
    left,right = round(dw + 0.1),round(dw - 0.1)        # 左右padding值
    top,bottom = round(dh + 0.1),round(dh - 0.1)        # 上下padding值
    out_img = cv2.resize(img,new_size,interpolate = cv2.INTER_AREA)
    out_img = cv2.copyMakeBorder(out_img,top,bottom,left,right,cv2.BORDER_CONSTANT,value = fill_value)
    return out_img,scale,dh,dw

def random_affine(img,targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5))

    h,w,_ = img.shape
    border = 0
    # shear         相对剪切角 角度——> 弧度 ——> tan 值
    S = np.eye(3)
    S[0,1] = math.tan((random.random()*(shear[1]-shear[0]) + shear[0]) * math.pi/180)
    S[1,0] = math.tan((random.random()*(shear[1]-shear[0])+shear[0]) * math.pi/180)
    # translate     
    T = np.eye(3)
    T[0,2] = (random.random() * 2 - 1)*translate[0] * h + border
    T[1,2] = (random.random() * 2 - 1)*translate[1] * w + border

    # rotate and scale
    R = np.eye(3)
    r = random.random()*(degrees[1] - degrees[0]) + degrees[0]
    
    s = random.random()*(scale[1] - scale[0]) + scale[0]

    R[:2] = cv2.getRotationMatrix2D(angle=r, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    M = S @ T @ R
    imw = cv2.warpPerspective(img, M, dsize=(w, h), flags=cv2.INTER_LINEAR,borderValue = borderValue)
    
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 2:6].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            #np.clip(xy[:, 0], 0, width, out=xy[:, 0])
            #np.clip(xy[:, 2], 0, width, out=xy[:, 2])
            #np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            #np.clip(xy[:, 3], 0, height, out=xy[:, 3])
            width = xy[:, 2] - xy[:, 0]
            height = xy[:, 3] - xy[:, 1]
            area = width * height
            ar = np.maximum(width / (height + 1e-16), height / (width + 1e-16))
            i = (width > 4) & (height > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 2:6] = xy[i]
            targets = targets[targets[:, 2] < w]
            targets = targets[targets[:, 4] > 0]
            targets = targets[targets[:, 3] < h]
            targets = targets[targets[:, 5] > 0]

        return imw, targets, M
    else:
        return imw
