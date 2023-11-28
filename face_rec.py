import numpy as np
import pandas as pd
import cv2

import redis


# insight face
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

# Connect to Redis Client
hostname = 'redis-14249.c239.us-east-1-2.ec2.cloud.redislabs.com'
portnumber = 14249
password = '5ASm8yEK2Bu8o4aqqXzkM1otOfhyM57z'

r = redis.StrictRedis(host=hostname,
                      port=portnumber,
                      password=password)


# configure face analysis
faceapp = FaceAnalysis(name='buffalo_sc',root='insightface_model', providers = ['CPUExecutionProvider'])
faceapp.prepare(ctx_id = 0, det_size=(640,640), det_thresh = 0.5)

def ml_search_algorithm(dataframe, feature_column, test_vector,
                        name_role=['Name', 'Role'], thresh=0.5):
    dataframe = dataframe.copy()
    X_list = dataframe[feature_column].tolist()

    # Find the maximum length among all facial features
    max_length = max(len(x) for x in X_list)

    # Pad or truncate each facial feature to the maximum length
    X_list = [np.pad(x, (0, max_length - len(x)), 'constant') if len(x) < max_length else x[:max_length] for x in X_list]
    x = np.asarray(X_list)

    # Resize or pad the test vector to match the length of facial features
    test_vector = np.pad(test_vector, (0, max_length - len(test_vector)), 'constant') if len(test_vector) < max_length else test_vector[:max_length]

    similar = pairwise.cosine_similarity(x, test_vector.reshape(1, -1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr

    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:
        data_filter.reset_index(drop=True, inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'

    return person_name, person_role


def face_prediction(test_image, dataframe,feature_column,
                        name_role=['Name','Role'],thresh=0.5):
    # step-1: take the test image and apply to insight face
    results = faceapp.get(test_image)
    test_copy = test_image.copy()
    # step-2: use for loop and extract each embedding and pass to ml_search_algorithm

    for res in results:
        x1, y1, x2, y2 = res['bbox'].astype(int)
        embeddings = res['embedding']
        person_name, person_role = ml_search_algorithm(dataframe,
                                                       feature_column,
                                                       test_vector=embeddings,
                                                       name_role=name_role,
                                                       thresh=thresh)
        if person_name == 'Unknown':
            color =(0,0,255) # bgr
        else:
            color = (0,255,0)


        cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)

        text_gen = person_name
        cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)

    return test_copy
