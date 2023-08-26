import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from video_reader import Video


def get_frames(video_path):
    video = Video(video_path)
    return video.return_list()


def pred_frame_pose(classifier, numpy_image):
    # Load the input image from a numpy array.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
    # Perform image classification on the provided single image.
    classification_result = classifier.classify(mp_image)
    return classification_result.classifications[0].categories[0].category_name


# Shot phase classification
def pose_mean_algo(model, video_path):
    frames = get_frames(video_path)
    phases = ['P1', 'P2', 'P3', 'P4', 'P5', 'P7', 'P8', 'P10']
    phd = dict([(ph, []) for ph in phases])
    # ids = []
    for i in range(0, len(frames), 2):
        # model returns 'P<num>'
        pred = pred_frame_pose(model, frames[i])
        phd[pred] += [i]
    ids = []
    for i, pn in enumerate(list(phd.values())):
        if len(pn) == 0:
            if i == 0:
                ids.append(0)
            else:
                ids.append(ids[i - 1] + 2)
        else:
            if len(ids) == 0:
                ids.append(sum(pn) // len(pn) + 1)
            else:
                if ids[i - 1] < sum(pn) // len(pn) + 1:
                    ids.append(sum(pn) // len(pn) + 1)
                else:
                    ids.append(ids[i - 1] + 2)
    # if len(ids) > len(phases):
    #     print('ERROR', ids)
    # if len(ids) < len(phases):
    #     ids += [len(frames)] * (len(phases) - len(ids))
    return ids


####################################
# Deprecated! Use pose_mean_algo() #
####################################
def pose_order_algo(model, video_path):
    frames = get_frames(video_path)
    phases = ['P1', 'P2', 'P3', 'P4', 'P5', 'P7', 'P8', 'P10']
    ids = []
    for i in range(0, len(frames), 2):
        # model returns 'P<num>'
        # print('pred:', pred_frame_pose(model, frames[i]))
        # print(phases[len(ids)])
        if pred_frame_pose(model, frames[i]) == phases[len(ids)]:
            ids.append(i)
            # print(ids)
    if len(ids) > len(phases):
        print('ERROR', ids)
    if len(ids) < len(phases):
        ids += [len(frames)] * (len(phases) - len(ids))
    return ids
