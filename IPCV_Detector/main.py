import cv2
import numpy as np
import os
import sys
import xml.etree.ElementTree as ET

from cv2 import HoughLines, GaussianBlur, Sobel

from matplotlib import pyplot as plt

# PARAMETERS:
# cascadeDirName: NoEntrycascade
# vecFileName: no_entry.vec
# bgFileName: negatives.dat
# numPos: 500
# numNeg: 500
# numStages: 3
# precalcValBufSize[Mb] : 1024
# precalcIdxBufSize[Mb] : 1024
# acceptanceRatioBreakValue : -1
# stageType: BOOST
# featureType: HAAR
# sampleWidth: 20
# sampleHeight: 20
# boostType: GAB
# minHitRate: 0.999
# maxFalseAlarmRate: 0.05
# weightTrimRate: 0.95
# maxDepth: 1
# maxWeakCount: 100
# mode: ALL
# Number of unique features given windowSize [20,20] : 125199
#
# ===== TRAINING 0-stage =====
# <BEGIN
# POS count : consumed   500 : 500
# NEG count : acceptanceRatio    500 : 1
# Precalculation time: 0.836
# +----+---------+---------+
# |  N |    HR   |    FA   |
# +----+---------+---------+
#
# ===== TRAINING 1-stage =====
# <BEGIN
# POS count : consumed   500 : 500
# NEG count : acceptanceRatio    500 : 0.0164799
# Precalculation time: 0.895
# +----+---------+---------+
# |  N |    HR   |    FA   |
# +----+---------+---------+
# |   1|        1|        1|
# +----+---------+---------+
# |   2|        1|    0.138|
# +----+---------+---------+
# |   3|        1|     0.02|
# +----+---------+---------+
# END>
# Training until now has taken 0 days 0 hours 0 minutes 6 seconds.
#
# ===== TRAINING 2-stage =====
# <BEGIN
# POS count : consumed   500 : 500
# NEG count : acceptanceRatio    500 : 0.000523583
# Precalculation time: 0.815
# +----+---------+---------+
# |  N |    HR   |    FA   |
# +----+---------+---------+
# |   1|        1|        1|
# +----+---------+---------+
# |   2|        1|    0.232|
# +----+---------+---------+
# |   3|        1|    0.278|
# +----+---------+---------+
# |   4|        1|     0.09|
# +----+---------+---------+
# |   5|        1|    0.002|
# +----+---------+---------+
# END>
# Training until now has taken 0 days 0 hours 0 minutes 14 seconds.

## FPR - Acceptance Ratios: 0: 1, 1: 0.0164799, 2: 0.000523583
## TPR: 1, 1, 1

##### NOTES of Interest: Only takes 14 seconds. TP rate is always 1;

def get_ground_truths(imageName):
    ground_truths = []
    xml_tree = ET.parse("ground_truths/" + imageName + ".xml")
    for sign in (xml_tree.findall("object")):
        ground_truths.append([
        int(sign.find("bndbox").find("xmin").text),
        int(sign.find("bndbox").find("ymin").text),
        int(sign.find("bndbox").find("xmax").text),
        int(sign.find("bndbox").find("ymax").text)
            ]
        )
    # print("ground truths= ", ground_truths)
    return ground_truths

def calc_iou(arr1,arr2 ):
    pred_area = arr1[2] * arr1[3]
    ground_area = (arr2[0] - arr2[2]) * (arr2[1] - arr2[3])
    height = (max(0, min(arr1[1] + arr1[3], arr2[3]) - max(arr1[1], arr2[1])))
    width = max(0, min(arr1[0] + arr1[2], arr2[2]) - max(arr1[0], arr2[0]))
    intersection_area = (width * height)
    union_area = pred_area + ground_area - intersection_area
    iou = intersection_area / union_area
    return iou

def iou(imageName, predicted):
    count = 0
    # loading ground truths
    # print("PRED", predicted)
    ground_truths = get_ground_truths(imageName)
    # compute iou
    ious = []
    print(ground_truths)
    for pred in predicted:
        pred_iou = []
        for ground_truth in ground_truths:
            iou = calc_iou(pred, ground_truth)
            # print("iou = ", iou)
            pred_iou.append(iou)
        ious.append(pred_iou)


    return ious


def nms(detected, scores, scores_threshold, nms_threshold):
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    sorted_boxes = [detected[i] for i in indices]
    sorted_scores = [scores[i] for i in indices]

    final_boxes = []
    final_scores = []

    while sorted_boxes:
        current_box = sorted_boxes.pop(0)
        current_score = sorted_scores.pop(0)
        final_boxes.append(current_box)
        final_scores.append(current_score)
        filtered_boxes = []
        filtered_scores = []

        for box, score in zip(sorted_boxes, sorted_scores):
            if calc_iou(current_box, box) < scores_threshold:
                filtered_boxes.append(box)
                filtered_scores.append(score)

        sorted_boxes = filtered_boxes
        sorted_scores = filtered_scores

    return final_boxes, final_scores



def detect_imgs(path,model):
    scores = []

    all_objects = []
    for img in os.listdir(path):
        frame = cv2.imread(path + img, 1)

        detected_objects, rejectLevels, levelWeights = model.detectMultiScale3(frame,
                                                                               minNeighbors=1,
                                                                               scaleFactor=1.01,
                                                                               minSize=(50, 50),
                                                                               outputRejectLevels=True)
        print(levelWeights)
        print("detected", detected_objects)
        if len(detected_objects) != 0:
        # using non-maximum suppression remove duplicate detections

            indices, _ = nms(detected_objects, levelWeights, 0.5, 0.5)
            # print(indices)
            # detected_objects = detected_objects[indices]

            ious = iou(img.strip('.bmp'), detected_objects)
            scores,all_objects = display_and_eval(ious, detected_objects, levelWeights, frame, img,path, all_objects, scores)

    return all_objects, rejectLevels, levelWeights, scores

def display_and_eval(ious, detected_objects, levelWeights, frame, img, path, all_objects, scores):
    tp = 0  # True Positives
    fp = 0  # False Positives
    print("ious: ", ious)
    print("Number of detected: ", len(detected_objects), "coords", detected_objects)

    # Get ground truths for the current image
    ground_truths = get_ground_truths(img.strip('.bmp'))
    matched_truths = set()  # Track ground truths that have been matched

    for truth in ground_truths:
        cv2.rectangle(frame, (truth[0], truth[1]), (truth[2], truth[3]), (0, 0, 255), 2)

    for i, detection in enumerate(detected_objects):
        detection_tp = False
        for j, truth in enumerate(ground_truths):
            if j in matched_truths:
                continue

            # Check if IoU is above the threshold
            if ious[i][j] > 0.5:
                # Mark as matched and true positive
                detection_tp = True
                matched_truths.add(j)
                break

        if detection_tp:
            tp += 1
            cv2.rectangle(frame, (detection[0], detection[1]),(detection[0] + detection[2], detection[1] + detection[3]),(0, 255, 0), 2)
            cv2.putText(frame, str(np.round(levelWeights[i], 3)), (detection[0], detection[1] - 5),cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)
        else:
            fp += 1
            cv2.rectangle(frame, (detection[0], detection[1]),
                          (detection[0] + detection[2], detection[1] + detection[3]), (0, 255, 0), 2)
            cv2.putText(frame, str(np.round(levelWeights[i], 3)), (detection[0], detection[1] - 5), cv2.FONT_ITALIC,
                        0.5, (0, 255, 0), 1)

    # False negatives are unmatched ground truths
    fn = len(ground_truths) - len(matched_truths)

    # Save results
    # print("Saving...")
    # print(img)
    cv2.imwrite("detected.jpg", frame)
    all_objects.append(detected_objects)

    # Append scores: [True Positives, False Positives, False Negatives]
    scores.append([tp, fp, fn])
    print("TP, FP, FN", scores[-1])
    # cv2.imshow("Detection",frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return scores, all_objects



def task1():
    model = cv2.CascadeClassifier()
    model.load("NoEntrycascade/cascade.xml")
    path = "No_entry/"
    detected_objects, rejectLevels, levelWeights, scores = detect_imgs(path, model)
    print(scores)
    avg_tpr, avg_f1 = calculate_score_rates(scores)
    print("TPR = ", avg_tpr, "F1 = ", avg_f1)


def calculate_score_rates(scores):
    total_recall = 0
    total_f1 = 0

    for tp, fp, fn in scores:

        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0

        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        total_recall += recall
        total_f1 += f1

    avg_tpr = total_recall / len(scores)
    avg_f1 = total_f1 / len(scores)
    return avg_tpr, avg_f1





## subtask 2:
def generate_image_gradients(img):
    image_data = cv2.imread(img, cv2.IMREAD_COLOR)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    #


    image_data = cv2.GaussianBlur(image_data, (5, 5), 0)
    # cv2.imwrite("img_blurred.png", image_data)

    convolved_vert = Sobel(image_data,  cv2.CV_16S, 0, 1)
    convolved_horiz = Sobel(image_data,  cv2.CV_16S, 1, 0)

    convolved_vert = np.absolute(convolved_vert)
    convolved_horiz = np.absolute(convolved_horiz)
    # replace with sobel

    # cv2.imwrite("img_vert.png", convolved_vert)
    # cv2.imwrite("img_horiz.png", convolved_horiz)

    img_mag = convolved_vert + convolved_horiz
    # img_mag = np.sqrt(np.square(convolved_horiz) + np.square(convolved_vert))
    # cv2.imwrite("img_mag.png", img_mag)

    img_mag = np.clip(img_mag, 0, 255).astype(np.uint8)

    # print("saving gradient image...")
    # cv2.imwrite("img_mag_c.png", img_mag)

    _, img_mag = cv2.threshold(img_mag, 110, 255, cv2.THRESH_BINARY)


    # cv2.imwrite("img_mag_thres.png", img_mag)

    img_ang = np.arctan2(convolved_horiz, convolved_vert)

    # #adjust to 0 - 2pi
    img_ang += np.pi
    # normalise between 0 and 255
    img_ang = 255 * (img_ang / (2 * np.pi))
    img_ang = np.clip(img_ang, 0, 255).astype(np.uint8)
    img_ang[img_mag == 0] = 0

    # cv2.imwrite("img_ang.png", img_ang)
    return img_mag, img_ang






def task2():
    scores = []
    for image_name in os.listdir("No_entry/"):


        image_name = image_name.strip(".bmp")
        path = "No_entry/"+image_name+".bmp"
        box_size = 1
        image_data, _ = generate_image_gradients(path)
        # cv2.imwrite(image_name + " gradient_t2.png", image_data)
        print("oigjsofj")
        # accumulator, thetas, rhos = hough(image_data)
        lines = []
        ### Line detector
        # detected_threshold = np.mean(accumulator) + (3 * np.std(accumulator))
        # for i in range(accumulator.shape[0]):  # Iterate over rhos
        #     for j in range(accumulator.shape[1]):  # Iterate over thetas
        #         if accumulator[i, j] >= detected_threshold:  # Check if it meets the threshold, calculating robust threshold
        #             rho = rhos[i]
        #             theta = thetas[j]
        #             lines.append(np.array([rho, theta]))
        # # reduce it for faster hough circle
        # cv2.imwrite("1.png", image_data)
        # print(lines)
        # print(image_data.shape)
        # image_data = reduce_image_samples(image_data, box_size)

  

        # cv2.imwrite("2.png", image_data)

        output_image = cv2.imread(path)
        img_h, img_w = image_data.shape
        print("Detecting circles")
        print(image_data.shape)

        acc, circles = hough_circles(image_data, 0.7)
        print(circles.shape)
        print(acc.shape)
        # Output circles
        # for circle in circles:
        #     cv2.circle(output_image, (int(circle[0])*box_size, int(circle[1])*box_size), int(circle[2]*box_size), (0, 255, 0), 2)
        # # convert circle to real coordinates
        # cv2.imshow('Detections', output_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        hough_space_plot(acc)
        model = cv2.CascadeClassifier()
        model.load("NoEntrycascade/cascade.xml")
        image_data = cv2.imread(path, -1)
        all_detected = []
        all_weights = []
        # apply VJ detector to each detected hough circle to detect signs
        for x,y,r,_ in circles:
            # with padding of r//4
            left = max(0, int(x - r - r //4 ) )
            right = min(img_w, int(x + r + r//4))
            top = max(0, int(y - r - r//4))
            bottom = min(img_h, int(y + r + r//4))

            region = image_data[top*box_size:bottom*box_size, left*box_size:right*box_size]
            temp_output = detect_subset([left*box_size,top*box_size],region, model)
            if temp_output is not None:
                weight, coords = temp_output
                all_detected.append(coords)
                all_weights.append(weight)

        # ellipses = test_ellipses(image_name)
        # for x,y,a,b,_ in ellipses:
        #     # with padding of r//4
        #     left = max(0, int(x - a - a //4 ) )
        #     right = min(img_w, int(x + a + a//4))
        #     top = max(0, int(y - b - b//4))
        #     bottom = min(img_h, int(y + b + b//4))
        #     region = image_data[top*box_size:bottom*box_size, left*box_size:right*box_size]
        #     temp_output = detect_subset([left*box_size,top*box_size],region, model)
        #     if temp_output is not None:
        #         weight, coords = temp_output
        #         all_detected.append(coords)
        #         all_weights.append(weight)


        print(all_detected)
        # calculated ious of detected sigs
        ious = iou(image_name, all_detected)
        print("ious = ", ious)
        tp = 0
        fp = 0
        # cv2.imshow('Detections', image_data)
        # Calculating the tp,fp,fn for each image, given detections
        scores, all_objects = display_and_eval(ious,all_detected, all_weights, output_image, image_name + ".bmp","No_entry/",[], scores)
        print(scores)

    avg_tpr, avg_f1 = calculate_score_rates(scores)
    print("TPR = ", avg_tpr, "F1 = ", avg_f1)
    # calculate FP, TP, FN
    # number of ious that are zero
    # number of ious that are non-zeros

    ## ELLIPSES
#
# def test_ellipses(img_name):
#
#     path = "No_entry/" + img_name + ".bmp"
#     box_size = 2
#     image_data, _ = generate_image_gradients(path)
#     cv2.imwrite("1.png", image_data)
#     print("oigjsofj")
#     # accumulator, thetas, rhos = hough(image_data)
#     print("Detecting lines")
#     lines = []
#     output_image = cv2.imread(path)
#     cv2.imwrite("2.png", image_data)
#     print("Detecting Ellipses")
#
#     _, ellipses = hough_ellipses(image_data, 7)
#     for ellipse in ellipses:
#         cv2.ellipse(output_image, (int(ellipse[0]*box_size), int(ellipse[1]*box_size)),(int(ellipse[2]*box_size), int(ellipse[3]*box_size)),0,startAngle=0,endAngle=360,color= (0, 0, 255), thickness=5)
#     #take output of circle detector and put into line detector
#     print(ellipses)
#     output_image = output_hough(lines, output_image)
#     cv2.imwrite("3.png", output_image)
#     return ellipses



def detect_subset(region_offset,frame, model):

    detected_objects, rejectLevels, levelWeights = model.detectMultiScale3(frame,
                                                                           minNeighbors=1,
                                                                           scaleFactor=1.01,
                                                                           minSize=(25, 25),
                                                                           outputRejectLevels=True,
                                                                           flags=cv2.CASCADE_SCALE_IMAGE)
    # draw detected objects as rectangles
    for obj in detected_objects:
        cv2.rectangle(frame, (obj[0], obj[1]), (obj[0]+obj[2], obj[1]+obj[3]), (0, 0, 255), 2)
    # # print("Detected: ", detected_objects)
    # cv2.imshow("Image", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # calculating the maximum area of detections in the hough detected region
    index = -1
    max_area = 0

    for i in range(len(detected_objects)):
        area = detected_objects[i][2] * detected_objects[i][3]
        if area > max_area:
            index = i
            max_area = area

    if max_area < 0.3 * (frame.shape[0] * frame.shape[1]): # this is another threshold value which could be changed
        index = -1

    if index != -1:
        centre_x = detected_objects[index][0] + detected_objects[index][2] // 2
        centre_y = detected_objects[index][1] + detected_objects[index][3] // 2
        radius = (detected_objects[index][2] + detected_objects[index][3]) // 4
        centre = (centre_x, centre_y)

        cv2.circle(frame,centre,radius,(0, 255, 0),2)

        return levelWeights[index], np.array([detected_objects[index][0]+region_offset[0],detected_objects[index][1]+region_offset[1],
                        detected_objects[index][2], detected_objects[index][3]])

    return None

'''

Consider representing your Hough Space as a 3D array.
When you are displaying the Hough Space, you might want to take the logarithm of the image to make the image values more descriptive.
Hint

'''

def hough(img):
    # initialising variables to reduce computation
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = round(np.ceil(np.sqrt(width * width + height * height)))  # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(num_thetas):
            rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos


def hough_space_plot(H):
    # hough_2d = H.sum(axis=2)
    # plt.figure(figsize=(10, 8))
    # plt.imshow(hough_2d, cmap='gray', origin='upper')
    # plt.title('2D Hough Space (Summed over Radii)')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.colorbar(label='Vote Intensity')
    # plt.show()
    pass
    #
    #


def hough_circles(img, threshold_factor):
    # precomputing everything to improve runtime
    angles = np.arange(0, 360, step=5)
    cos_t = np.cos(np.radians(angles))
    sin_t = np.sin(np.radians(angles))
    height, width = img.shape
    max_r = 125
    min_r = 10
    radii = np.arange(min_r, max_r)
    accumulator = np.zeros((height, width, max_r - min_r), dtype=np.uint64)
    edge_pixels = np.argwhere(img >= 50)  # above threshold

    for y, x in edge_pixels:
        for r_idx, r in enumerate(radii):
            a_vals = (x - r * cos_t).astype(int)
            b_vals = (y - r * sin_t).astype(int)
            # filtering centre points outside image
            valid_indices = (0 <= a_vals) & (a_vals < width) & (0 <= b_vals) & (b_vals < height)
            a_vals = a_vals[valid_indices]
            b_vals = b_vals[valid_indices]
            np.add.at(accumulator[:, :, r_idx], (b_vals, a_vals), 1)


    print("Thresholding Circles... ")
    top_indices = np.argsort(accumulator.flatten())[::-1][:1000]
    ys, xs, rs = np.unravel_index(top_indices, accumulator.shape)
    scores = accumulator.flatten()[top_indices]
    num_circles = 100
    selected_detections = []
    for x, y, r_offset, score in zip(xs, ys, rs, scores):
        r = r_offset + min_r
        #leniency between detections
        if all(np.hypot(x - sx, y - sy) >= r // 1.5 for sx, sy, sr, score2 in selected_detections):
            selected_detections.append((x, y, r, score))
        if len(selected_detections) >= num_circles:
            break

    return accumulator, np.array(selected_detections)



def output_hough(lines, image):
    if lines is not None:
        print(lines)
        for line in lines:
            # print(line)
            rho, theta = line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return image

# # same as hough circles, but has vertical and horizontal radius.
# def hough_ellipses(img, threshold_factor):
#     angles = np.arange(0, 360, step=5)
#     cos_t = np.cos(np.radians(angles))
#     sin_t = np.sin(np.radians(angles))
#     height, width = img.shape
#     # vertical and horizontal radii..
#     min_horiz_r = 5
#     min_vert_r = 10
#     max_horiz_r = 10
#     max_vert_r = 20
#     a_vals = np.arange(min_horiz_r, max_horiz_r)
#     b_vals = np.arange(min_vert_r, max_vert_r)
#     accumulator = np.zeros((height, width, len(a_vals), len(b_vals)), dtype=np.uint64)
#     edge_pixels = np.argwhere(img >= 50)
#     for y, x in edge_pixels:
#         for a_idx, a in enumerate(a_vals):
#             for b_idx, b in enumerate(b_vals):
#                 x_vals = (x - a * cos_t).astype(int)
#                 y_vals = (y - b * sin_t).astype(int)
#                 # same as circles...
#                 valid_indices = (0 <= x_vals) & (x_vals < width) & (0 <= y_vals) & (y_vals < height)
#                 x_vals = x_vals[valid_indices]
#                 y_vals = y_vals[valid_indices]
#                 np.add.at(accumulator[:, :, a_idx, b_idx], (y_vals, x_vals), 1)
#
#     print("Thresholding Ellipses...")
#     num_ellipses = 100
#     selected_detections = []
#     threshold = threshold_factor * np.max(accumulator)
#     top_indices = np.argsort(accumulator.flatten())[::-1][:1000]
#     ys, xs, a_idxs, b_idxs = np.unravel_index(top_indices, accumulator.shape)
#     scores = accumulator.flatten()[top_indices]
#     for x, y, a_idx, b_idx, score in zip(xs, ys, a_idxs, b_idxs, scores):
#         a = a_vals[a_idx]
#         b = b_vals[b_idx]
#         if  all(np.hypot(x - sx, y - sy) >= max(a, b) // 1.5 for sx, sy, sa, sb, sscore in selected_detections):
#             selected_detections.append((x, y, a, b, score))
#         if len(selected_detections) >= num_ellipses:
#             break
#
#     return accumulator, np.array(selected_detections)


def normalise_image(img):
    image_data = cv2.imread(img, cv2.IMREAD_COLOR)
    # cv2.imwrite("default.png", image_data)
    image_data = cv2.convertScaleAbs(image_data, alpha=1.5, beta=0)
    # cv2.imwrite("bright_contra.png", image_data)

    ycrcb = cv2.cvtColor(image_data, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))

    image_data = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    # cv2.imwrite("equalise.png", image_data)
    return image_data

def contour_detection(edges):

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ellipses = []
    rectangles = []
    image_data = cv2.imread("equalise.png")
    for contour in contours:
        # eval contour
        if cv2.contourArea(contour) > 140:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (x_centre, y_centre), (minor_axis,major_axis), angle = ellipse
                if major_axis / minor_axis < 4:
                    print(major_axis, minor_axis, x_centre, y_centre)
                    ellipses.append([x_centre, y_centre, minor_axis, major_axis, angle])
                    cv2.ellipse(image_data, (int(x_centre), int(y_centre)),  (int(minor_axis//2), int(major_axis//2)), angle,0,360, (0, 255, 0), thickness=1)

                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                centre_x, centre_y = rect[0]
                angle = rect[2]
                angle_normalised = angle % 180
                if angle_normalised > 90:
                    angle_normalised -= 180
                    width, height = height, width

                print("Original rect: ", rect)
                print("Normalised angle:  ", angle_normalised)
                aspect_ratio = max(width, height) / min(width, height)
                area = width * height

                if -45 <= angle_normalised <= 45:
                    height = height * aspect_ratio * 1.5
                else:
                    width = width * aspect_ratio * 1.5
                new_rect = ((centre_x, centre_y), (width, height), angle)

                box = cv2.boxPoints(new_rect)
                box = np.intp(box)
                if aspect_ratio < 5 and 50 < area < 5000:
                    middle_bar = box
                    cv2.polylines(image_data, [box], isClosed=True, color=(0, 255, 0), thickness=2)
                    rectangles.append(new_rect)


    # cv2.imshow("jkaw", image_data)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ellipses = merge_similar_ellipses(ellipses,distance_threshold=50,size_ratio_threshold=0.2,angle_threshold=30)
    print(ellipses)
    return ellipses, rectangles


def merge_similar_ellipses(ellipses, distance_threshold, size_ratio_threshold, angle_threshold):

    if not ellipses:
        return []
    merged = []
    used = set()

    for i, ellipse1 in enumerate(ellipses):
        if i in used:
            continue
        x1, y1, minor1, major1, angle1 = ellipse1
        similar = []
        for j, ellipse2 in enumerate(ellipses):
            if j <= i or j in used:
                continue

            x2, y2, minor2, major2, angle2 = ellipse2

            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            area1 = np.pi * minor1 * major1
            area2 = np.pi * minor2 * major2
            size_ratio = abs(area1 - area2) / max(area1, area2)

            angle_diff = abs(angle1 - angle2) % 180
            if angle_diff > 90:
                angle_diff = 180 - angle_diff
            print(size_ratio, angle_diff, distance)
            if (distance < distance_threshold and size_ratio < size_ratio_threshold and angle_diff < angle_threshold):
                similar.append(j)

        if similar:
            used.update(similar)
            used.add(i)

            all_indices = [i] + similar
            x_avg = sum(ellipses[k][0] for k in all_indices) / len(all_indices)
            y_avg = sum(ellipses[k][1] for k in all_indices) / len(all_indices)
            minor_avg = sum(ellipses[k][2] for k in all_indices) / len(all_indices)
            major_avg = sum(ellipses[k][3] for k in all_indices) / len(all_indices)
            major_avg = 2 * major_avg
            minor_avg = 2 * minor_avg
            angle_avg = sum(ellipses[k][4] for k in all_indices) / len(all_indices)

            merged.append([x_avg, y_avg, minor_avg + minor_avg//8, major_avg + major_avg//8, angle_avg])

        elif i not in used:
            merged.append(ellipse1)

    return merged



def detect_subset_task3(region_offset,frame, model, adjusted_frame):

    detected_objects, rejectLevels, levelWeights = model.detectMultiScale3(adjusted_frame,
                                                                           minNeighbors=1,
                                                                           scaleFactor=1.01,
                                                                           minSize=(15, 15),
                                                                           outputRejectLevels=True,
                                                                           flags=cv2.CASCADE_SCALE_IMAGE)
    # draw detected objects as rectangles
    for obj in detected_objects:
        cv2.rectangle(frame, (obj[0], obj[1]), (obj[0]+obj[2], obj[1]+obj[3]), (0, 0, 255), 2)
    print("Detected: ", detected_objects)
    # cv2.imshow("Image", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # calculating the maximum area of detections in the hough detected region
    index = -1
    max_area = 0

    for i in range(len(detected_objects)):
        area = detected_objects[i][2] * detected_objects[i][3]
        if area > max_area:
            index = i
            max_area = area

    if max_area < 0.3 * (frame.shape[0] * frame.shape[1]): # this is another threshold value which could be changed
        index = -1

    if index != -1:
        centre_x = detected_objects[index][0] + detected_objects[index][2] // 2
        centre_y = detected_objects[index][1] + detected_objects[index][3] // 2
        radius = (detected_objects[index][2] + detected_objects[index][3]) // 4
        centre = (centre_x, centre_y)

        cv2.circle(frame,centre,radius,(0, 255, 0),2)

        return levelWeights[index], np.array([detected_objects[index][0]+region_offset[0],detected_objects[index][1]+region_offset[1],
                        detected_objects[index][2], detected_objects[index][3]])

    return None




def generate_image_gradients_task3(image_data, image_name):
    # print(image_data.shape)
    blue_channel = image_data[:, :, 0].astype(np.float32)
    green_channel = image_data[:, :, 1].astype(np.float32)
    red_channel = image_data[:, :, 2].astype(np.float32)

    red = (red_channel > green_channel + 75) & (red_channel > blue_channel + 75)
    red = np.clip(red, 0, 1) * 255

    red = np.clip(red, 0, 1) * 255
    red = red.astype(np.uint8)

    _, red = cv2.threshold(red, 100, 255, cv2.THRESH_BINARY)

    # Save the output image
    # cv2.imwrite("RED.png", red)
    image_data = red

    #
    # vertical_prewitt = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    # horizontal_prewitt = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

    image_data = cv2.GaussianBlur(image_data, (3, 3), 0)
    # cv2.imwrite("img_blurred.png", image_data)

    img_mag = image_data
    # cv2.imwrite("img_mag.png", img_mag)

    img_mag = np.clip(img_mag, 0, 255).astype(np.uint8)

    print("saving gradient image...")
    # cv2.imwrite("img_mag_c.png", img_mag)

    img_mag = cv2.adaptiveThreshold(img_mag,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

    img_mag = cv2.morphologyEx(img_mag, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    img_mag = cv2.morphologyEx(img_mag, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # cv2.imwrite("img_mag_thres.png", img_mag)
    img_mag = cv2.Canny(img_mag, 50, 150)


    return img_mag, 0


def task3(image_name):
    scores = []
    # for image_name in os.listdir("No_entry/"):
    # image_name = "NoEntry10.bmp"
    image_name = image_name.strip(".bmp")
    path = "No_entry/" + image_name + ".bmp"

    enhanced_image = normalise_image(path)
    image_data, _ = generate_image_gradients_task3(enhanced_image, image_name)
    # cv2.imshow("fwf", image_data)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ellipses, _ = contour_detection(image_data)

    output_image = cv2.imread(path)
    img_h, img_w = image_data.shape

    _, circles = hough_circles(image_data, 0.7)

    model = cv2.CascadeClassifier()
    model.load("NoEntrycascade/cascade.xml")
    image_data = cv2.imread(path, -1)

    circle_detections = []
    circle_weights = []
    ellipse_detections = []
    ellipse_weights = []

    # for (x,y),(a,b),_ in rectangles:
    #     padding = b // 8
    #     left = max(0, int(x - b//2 - padding))
    #     right = min(img_w, int(x + b//2 + padding))
    #     top = max(0, int(y - b//2 - padding))
    #     bottom = min(img_h, int(y + b//2 + padding))
    #     if left != right and top != bottom:
    #         regions.append({
    #             'coords': [left, top, right, bottom],
    #             'type': 'rectangle'
    #         })

    print("processing circles...")
    for x, y, r, _ in circles:
        padding = r // 4
        left = max(0, int(x - r - padding))
        right = min(img_w, int(x + r + padding))
        top = max(0, int(y - r - padding))
        bottom = min(img_h, int(y + r + padding))
        if left != right and top != bottom:
            region = enhanced_image[top:bottom, left:right]
            temp_output = detect_subset_task3([left, top], region, model,
                                              image_data[top:bottom, left:right])
            if temp_output is not None:
                weight, coords = temp_output
                circle_detections.append(coords)
                circle_weights.append(weight)

    print("ellipses..")
    for x, y, b, a, theta in ellipses:
        padding = b // 8
        left = max(0, int(x - b // 2 - padding))
        right = min(img_w, int(x + b // 2 + padding))
        top = max(0, int(y - b // 2 - padding))
        bottom = min(img_h, int(y + b // 2 + padding))
        if left != right and top != bottom:
            region = enhanced_image[top:bottom, left:right]
            temp_output = detect_subset_task3([left, top], region, model,
                                              image_data[top:bottom, left:right])
            if temp_output is not None:
                weight, coords = temp_output
                ellipse_detections.append(coords)
                ellipse_weights.append(weight)

    all_detected = []
    all_weights = []

    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    for det, weight in zip(circle_detections, circle_weights):
        all_detected.append(det)
        all_weights.append(weight)

    for det, weight in zip(ellipse_detections, ellipse_weights):
        should_add = True
        for existing_det in all_detected:
            if calculate_iou(det, existing_det) > 0.5:
                should_add = False
                break
        if should_add:
            all_detected.append(det)
            all_weights.append(weight)
    print("calculating scores")
    ious = iou(image_name, all_detected)
    scores, all_objects = display_and_eval(ious, all_detected, all_weights,
                                           output_image, image_name + ".bmp",
                                           "No_entry/", [], scores)

    # avg_tpr, avg_f1 = calculate_score_rates(scores)
    # print("TPR = ", avg_tpr, "F1 = ", avg_f1)



def main():
    # task1()
    # task2()
    # task3()
    return -1

if __name__ == "__main__":
    if len(sys.argv) != 2:

        sys.exit(1)

    input_filename = sys.argv[1]
    task3(input_filename)
