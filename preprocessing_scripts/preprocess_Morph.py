import pandas as pd
import numpy as np
import cv2
# import json # to serialize objects, so can be stored as string in pandas's feather file
# import matplotlib.pyplot as plt
from pathlib import Path
import dlib
from pose import get_rotation_angle

# process it to detect faces, detect landmarks, align, & make 3 sub boxes which will be used in next step to feed into network
# save dataset as pandas,feather & imencode for size efficiency


def gen_boundbox(box, landmark):
    # getting 3 boxes for face, as required in paper... i.e feed 3 different sized images to network (R,G,B) 
    xmin, ymin, xmax, ymax = box # box is [ymin, xmin, ymax, xmax]
    w, h = xmax - xmin, ymax - ymin
    nose_x, nose_y = (landmark.parts()[30].x, landmark.parts()[30].y) # calculating nose center point, so the triple boxes will be cropped according to nose point
    w_h_margin = abs(w - h)
    top2nose = nose_y - ymin
    # Contains the smallest frame
    return np.array([
        [(xmin - w_h_margin, ymin - w_h_margin), (xmax + w_h_margin, ymax + w_h_margin)],  # out
        [(nose_x - top2nose, nose_y - top2nose), (nose_x + top2nose, nose_y + top2nose)],  # middle
        [(nose_x - w//2, nose_y - w//2), (nose_x + w//2, nose_y + w//2)]  # inner box
    ])



def init_dataset_meta_csv():
  image_dir = dataset_base_path.joinpath('morph')

  dataset_meta_list = []
  for file_path in image_dir.glob('*.jpg'):
    img_name = str(file_path).split('/')[-1]
    temp_name = img_name.split('_')
    temp_name = temp_name[1].split('.')
    isMale = temp_name[0].find('M')
    isFemale = temp_name[0].find('F')

    if isMale > -1:
        gender = 0
        age = temp_name[0].split('M')
        age = age[1]
    elif isFemale > -1:
        gender = 1
        age = temp_name[0].split('F')
        age = age[1]

    age = int(float(age))

    dataset_meta_list.append([file_path,age,gender])

  # make a dataframe dataset
  dataset_meta_df = pd.DataFrame(dataset_meta_list, columns=['full_path','age','gender'])
  # save dataframe as csv
  dataset_meta_df.to_csv(dataset_base_path.joinpath(dataset_name+'_meta.csv'),index=False)


def detect_faces_and_landmarks(image):
  face_rect_list = detector(image)
  img_face_count = len(face_rect_list) # number of faces found in image
  if img_face_count < 1:
    return 0,[],[] # no face found, so return 

  xmin, ymin, xmax, ymax = face_rect_list[0].left() , face_rect_list[0].top(), face_rect_list[0].right(), face_rect_list[0].bottom() # face_rect is dlib.rectangle object, so extracting values from it
  
  # make a landmarks_list of all faces detected in image
  lmarks_list = dlib.full_object_detections()
  for face_rect in face_rect_list:
    lmarks_list.append(predictor(image, face_rect)) # getting landmarks as a list of objects
  
  return img_face_count,np.array([xmin, ymin, xmax, ymax]), lmarks_list


def loadData_preprocessData_and_makeDataFrame():
  meta_dataframe = pd.read_csv(dataset_base_path.joinpath(dataset_name+'_meta.csv'))
  properties_list = [] # init lists of all properties gonna be saved
  # loop through meta.csv for all images
  for index,series in meta_dataframe.iterrows():
    image_path = series.full_path # get image path
    try:
      image = cv2.imread(image_path, cv2.IMREAD_COLOR)
      # image = cv2.copyMakeBorder(image, self.extra_padding, self.extra_padding, self.extra_padding, self.extra_padding, cv2.BORDER_CONSTANT)
      face_count,_,lmarks_list = detect_faces_and_landmarks(image) # Detect face & landmarks
      if face_count != 1:
        raise Exception("more than 1 or no face found in image ",image_path )
      # found exactly 1 face, so now process it
      #extract_image_chips will crop faces from image according to size & padding and align them in upright position and return list of them
      cropped_faces = dlib.get_face_chips(image, lmarks_list, padding=extra_padding)  # aligned face with padding 0.4 in papper
      image = cropped_faces[0] # must be only 1 face, so getting it.
      # detect face landmarks again from cropped & align face.  (as positions of lmarks are changed in cropped image)
      _,face_rect_box, lmarks_list = detect_faces_and_landmarks(image) # Detect face from cropped image
      first_lmarks = lmarks_list[0] # getting first face's rectangle box and landmarks 
      trible_box = gen_boundbox(face_rect_box, first_lmarks) # get 3 face boxes for nput into network, as reauired in paper
      if (trible_box < 0).any():
        print(index,'Some part of face is out of image ',series.full_path)
        raise Exception("more than 1 or no face found in image ",image_path )
      face_pitch, face_yaw, face_roll = get_rotation_angle(image, first_lmarks) # gen face rotation for filtering
    except Exception as ee:        
      # print('index ',index,': exption ',ee)
      properties_list.append([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]) # add null dummy values to current row & skill this iteration
      continue
      
    # everything processed succefuly, now serialize values and save them
    status, buf = cv2.imencode(".jpg", image)
    image_buffer = buf.tostring()
    #dumping with `pickle` much faster than `json` (np.dumps is pickling)
    face_rect_box_serialized = face_rect_box.dumps()  # [xmin, ymin, xmax, ymax] : Returns the pickle(encoding to binary format (better than json)) of the array as a string. pickle.loads or numpy.loads will convert the string back to an array
    trible_boxes_serialized = trible_box.dumps() # 3 boxes of face as required in paper
    landmarks_list = np.array([[point.x,point.y] for point in first_lmarks.parts()]) # Same converting landmarks (face_detection_object) to array so can be converted to json
    face_landmarks_serialized = landmarks_list.dumps()#json.dumps(landmarks_list,indent = 2)  # y1..y5, x1..x5
    
    # adding everything to list
    properties_list.append([image_path,series.age,series.gender,image_buffer,face_rect_box_serialized,trible_boxes_serialized,face_yaw,face_pitch,face_roll,face_landmarks_serialized])
    if index%500 == 0:
      print(index,'images preprocessed')
  processed_dataset_df = pd.DataFrame(properties_list,columns=['image_path','age','gender','image','org_box','trible_box','yaw','pitch','roll','landmarks'])
  # some filtering on df
  processed_dataset_df = processed_dataset_df.dropna()
  processed_dataset_df = processed_dataset_df[(processed_dataset_df.age >= 0) & (processed_dataset_df.age <= 100)]
  # processed_dataset_df.to_csv('/content/Dataset.csv',index=False)
  #Dataset_DF = processed_dataset_df # putting it into global Dataset_DF variable
  return processed_dataset_df # returning now (just in case need to return), maybe later remove...



# save processed dataset_df to feather format
def save(chunkSize=5000):
    print('df ssize save  ',len(Dataset_DF))
    dataframe = Dataset_DF.reset_index()
    chunk_start = 0
    while(chunk_start < len(Dataset_DF)):
        dir_path = dataset_base_path.joinpath(dataset_name + "_" + str(int(chunk_start / chunkSize)) + ".feather")
        tmp_pd = dataframe[chunk_start:chunk_start + chunkSize].copy().reset_index()
        tmp_pd.to_feather(dir_path)
        chunk_start += chunkSize
        print('succesfully saved as feather to ',dir_path)



def rectify_data():
    sample = []
    max_nums = 500.0
    for x in range(100):
        age_set = Dataset_DF[Dataset_DF.age == x]
        cur_age_num = len(age_set)
        if cur_age_num > max_nums:
            age_set = age_set.sample(frac=max_nums / cur_age_num, random_state=2007)
        sample.append(age_set)
    Dataset_DF = pd.concat(sample, ignore_index=True)
    Dataset_DF.age = Dataset_DF.age 
    print(Dataset_DF.groupby(["age", "gender"]).agg(["count"]))

################################## GLOBAL PARAMS ##############################################

#initiate face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/content/C3AE_keras/detector/shape_predictor_68_face_landmarks.dat")

# creating dummy DF, later will process all images to make it real df
Dataset_DF = pd.DataFrame(columns=["age", "gender", "image", "org_box", "trible_box", "landmarks", "roll", "yaw", "pitch"])

# define all parameters here
dataset_base_path = Path('/content/C3AE_keras/datasets/aligned/')
dataset_name = 'morph' # different dataset name means different sequence for loading etc
# image transform params (if require)
extra_padding = 0.55

if __name__ == "__main__":

    init_dataset_meta_csv() # convert meta.mat to meta.csv
    Dataset_DF = loadData_preprocessData_and_makeDataFrame()
    save() # save preprocessed dataset as .feather in  dataset_directory_path
