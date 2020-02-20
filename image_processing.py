from header import *
"""HSV_RANGES = {
    # red is a major color
    'red': [
        {
            'lower': np.array([0, 39, 64]),
            'upper': np.array([20, 255, 255])
        },
        {
            'lower': np.array([161, 39, 64]),
            'upper': np.array([180, 255, 255])
        }
    ],
    # yellow is a minor color
    'yellow': [
        {
            'lower': np.array([21, 39, 64]),
            'upper': np.array([40, 255, 255])
        }
    ],
    # green is a major color
    'green': [
        {
            'lower': np.array([41, 39, 64]),
            'upper': np.array([80, 255, 255])
        }
    ],
    # cyan is a minor color
    'cyan': [
        {
            'lower': np.array([81, 39, 64]),
            'upper': np.array([100, 255, 255])
        }
    ],
    # blue is a major color
    'blue': [
        {
            'lower': np.array([101, 39, 64]),
            'upper': np.array([140, 255, 255])
        }
    ],
    # violet is a minor color
    'violet': [
        {
            'lower': np.array([141, 39, 64]),
            'upper': np.array([160, 255, 255])
        }
    ],
    # next are the monochrome ranges
    # black is all H & S values, but only the lower 25% of V
    'black': [
        {
            'lower': np.array([0, 0, 0]),
            'upper': np.array([180, 255, 63])
        }
    ],
    # gray is all H values, lower 15% of S, & between 26-89% of V
    'gray': [
        {
            'lower': np.array([0, 0, 64]),
            'upper': np.array([180, 38, 228])
        }
    ],
    # white is all H values, lower 15% of S, & upper 10% of V
    'white': [
        {
            'lower': np.array([0, 0, 229]),
            'upper': np.array([180, 38, 255])
        }
    ]
}"""
def get_mask(_img, _colors):
    # _img = cv2.GaussianBlur(_img, (3, 3), 0)
    ycrcb_equalize(_img)
    hsv = cv2.cvtColor(_img, cv2.COLOR_BGR2HSV)

    mask = np.zeros(_img.shape[:2],np.uint8)
    if 'blue' in _colors:
        mask = mask | cv2.inRange(hsv, lower_blue, upper_blue)
    if 'red' in _colors:
        mask = mask | cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    if 'white' in _colors:
        mask = mask | cv2.inRange(hsv, lower_white, upper_white)
    if 'black' in _colors:
        mask = mask | cv2.inRange(hsv, lower_black, upper_black)
    if 'orange' in _colors:
        mask = mask | cv2.inRange(hsv,lower_orang, upper_orang)
    
    kernel = np.ones((5, 5),np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    mask = cv2.erode(mask, kernel, iterations = 1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)    
    return mask

def check_is_circles(_img):
    gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
    return (cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 
        minDist= gray.shape[0], param1=50, param2=30,
        minRadius=int(gray.shape[1]/2-5),maxRadius=int(gray.shape[0]/2+5)) is not None)

# def find_bounds(_mask, _img):
#     contours, hierarchy = cv2.findContours(_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     bounds = []
#     for contour in contours:
#         [x, y, w, h] = bound = cv2.boundingRect(contour)
#         contour_area = cv2.contourArea(contour)
#         ellipse_area = (math.pi * (w / 2) * (h / 2))

#         if (0.4 < w / h < 1.6) and (w > 20) and (h > 20):
#             if 0.8 < (contour_area / ellipse_area) < 1.2:
#                 if True:#check_is_circles(_img[y:y+h, x:x+w]):
#                     bounds.append(bound)

#     return sorted(bounds, key=lambda x: (x[2] * x[3]))

def image_fill(Binary_image):
    # Mask used to flood filling.
    im_th=Binary_image.astype('uint8').copy()
    h, w = im_th.shape[:2]
    im_floodfill = im_th.copy()
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 1);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    im_out[im_out==254]=0
    return im_out

def cnts_find(binary_image):
    cont_Saver=[]

    (cnts, _) = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#finding contours of conected component
    for d in cnts:
        if cv2.contourArea(d)>1000:
            (x, y, w, h) = cv2.boundingRect(d)
            if ((w/h)<1.21 and (w/h)>0.59 and w>35):
                cont_Saver.append([cv2.contourArea(d),x, y, w, h])
    return cont_Saver

def ycrcb_equalize(_img):
    ycrcb = cv2.cvtColor(_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4,4))
    y = clahe.apply(y)

    ycrcb = cv2.merge([y, cr, cb])
    _img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return _img

def classify_image(_image, _svm):
    descriptor = hog.compute(_image)
    return int(_svm.predict(np.array([descriptor]))[1][0])

def show_template(_img, _bound, _template):
    [x, y, w, h] = _bound
    _template = cv2.resize(_template, (h, h))
    # right
    if x + w + h < _img.shape[1]:
        _img[y : y + h, x + w: x + w + h] = _template
    # left
    else:
        _img[y : y + h, x - h: x] = _template
    
# def recognize_sign(_img, _mask, _svm, _id):
#     bounds = find_bounds(_mask, _img)
    
#     for bound in bounds:
#         [x, y, w, h] = bound
        
#         sign = _img[y:(y + h), x:(x + w)]
#         sign = cv2.resize(sign, (width, height))
#         class_id = classify_image(sign, _svm)
        
#         if class_id == _id:
#             return bound, class_id
#     return [], 11

def draw(_img, _bound, _template, _template_title):
    [x, y, w, h] = _bound
    # draw contour
    cv2.rectangle(_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    
    # show template of traffic sign
    show_template(_img, _bound, _template)
    # show traffic sign infomation
    # cv2.putText(_img, _templates_title, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    return _img

# def process_image(_img, _svm, _templates, _templates_title, _frame_id, _info):
#     img = ycrcb_equalize(_img)

#     '''
#     blue_mask = get_mask(img, ['blue'])
#     red_mask = get_mask(img, ['red'])
#     blue_red_mask = get_mask(img, ['blue', 'red'])
#     white_black_mask = get_mask(img, ['white', 'black'])

#     blue_bound, blue_predict_id = regconize_sign(img, blue_mask, _svm)
#     red_bound, red_predict_id = regconize_sign(img, red_mask, _svm)
#     blue_red_bound, blue_red_predict_id = regconize_sign(img, blue_red_mask, _svm)
#     white_black_bound, white_black_predict_id = regconize_sign(img, white_black_mask, _svm)
#     ''' 
#     orange_mask = get_mask(img, ['orange'])
#     #filled_img = image_fill(orange_mask)
#     orange_blur = cv2.medianBlur(orange_mask,3)

#     # cont_Saver=cnts_find(filled_img)
#     # if len(cont_Saver)>0:
#     #     cont_Saver=np.array(cont_Saver)

#     #     cont_Saver=cont_Saver[cont_Saver[:,0].argsort()].astype(int)
#     #     for conta in range(len(cont_Saver)):
#     #         cont_area,x, y, w, h=cont_Saver[len(cont_Saver)-conta-1]
#     #         cv2.rectangle(_img, (x, y), (x + w, y + h), (0, 255, 0), 4)
#             # #getting the boundry of rectangle around the contours.

#             # image_found=img[y:y+h,x:x+w]

#             # crop_image=image_found.copy()
#             # img0=cv2.cvtColor(image_found, cv2.COLOR_RGB2GRAY)
#             # img0 = cv2.medianBlur(img0,3)

#             # crop_image0=cv2.resize(img0, (64, 64))

#             # # Apply Hog from skimage library it takes image as crop image.Number of orientation bins that gradient
#             # # need to calculate.
#             # ret,crop_image0 = cv2.threshold(crop_image0,127,255,cv2.THRESH_BINARY)
#             # descriptor,imagehog  = hog(crop_image0, orientations=8,pixels_per_cell=(4,4),visualize=True)
#     wanted_id = 8
#     templates_id = blue_templates_id
    
#     bound, predict_id = recognize_sign(img, orange_mask, _svm, wanted_id)
#     if predict_id == wanted_id:
#         draw(_img, bound, _templates[predict_id - 1], _templates_title[predict_id - 1])
#         [x, y, w, h] = bound
#         _info.append([_frame_id, predict_id, x, y, x + w, y + h, '\n'])
            
#     cv2.imshow('result', _img)
#     #cv2.imshow('mask', orange_mask)
#     cv2.imshow('filled_img', orange_blur)
#     # cv2.imshow('mask', mask)
#     # cv2.imshow('blue_mask', blue_mask)
#     # cv2.imshow('red_mask', red_mask)
#     # cv2.imshow('white_black_mask', white_black_mask)
#     return orange_blur