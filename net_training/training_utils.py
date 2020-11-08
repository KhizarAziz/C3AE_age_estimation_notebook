import numpy as np
import cv2
import pickle
import random
import keras.backend as K


# adding random box in image with random colored pixels, it makes model generic????
def random_erasing(img, dropout=0.3, aspect=(0.5, 2), area=(0.06, 0.10)):
    # https://arxiv.org/pdf/1708.04896.pdf
    if 1 - random.random() > dropout:
        return img
    img = img.copy()
    height, width = img.shape[:-1]
    aspect_ratio = np.random.uniform(*aspect)
    area_ratio = np.random.uniform(*area)
    img_area = height * width * area_ratio
    dwidth, dheight = np.sqrt(img_area * aspect_ratio), np.sqrt(img_area * 1 / aspect_ratio) 
    xmin = random.randint(0, height)
    ymin = random.randint(0, width)
    xmax, ymax = min(height, int(xmin + dheight)), min(width, int(ymin + dwidth))
    img[xmin:xmax,ymin:ymax,:] = np.random.random_integers(0, 256, (xmax-xmin, ymax-ymin, 3))
    return img



def two_point(age_label, category, interval=10, elips=0.000001):
    def age_split(age):
        embed = [0 for x in range(0, category)]
        right_prob = age % interval * 1.0 / interval
        left_prob = 1 - right_prob
        idx = age // interval
        if left_prob:
            embed[idx] = left_prob
        if right_prob and idx + 1 < category:
            embed[idx+1] = right_prob
        return embed
    return np.array(age_split(age_label))



def image_transform(row,dropout,target_img_shape,require_augmentation):
  # read image from buffer then decode
  img = np.frombuffer(row["image"], np.uint8)
  img = cv2.imdecode(img, cv2.IMREAD_COLOR)
  #add random noise
  if require_augmentation:
    img = random_erasing(img,dropout=dropout)
  #add padding, incase any face location is negative (face is not full)
  padding = 50
  img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
  # get trible box (out,middle,inner) and crop image from these boxes then
  tripple_cropped_imgs = []
  for box in pickle.loads(row['trible_box'],encoding="bytes"): # deserializing object which we converted to binary format using myNumArray.dump() method
    h_min, w_min = box[0] # xmin,ymin
    h_max, w_max = box[1] #xmax, ymax
    # print('img shape {} & trible box {} '.format(img.shape,box))
    # crop image according to box size and add to list
    triple_box_cropped = img[w_min+padding:w_max+padding, h_min+padding: h_max+padding] # cropping image
    triple_box_cropped = cv2.resize(triple_box_cropped, (64,64)) # resize according to size we want
    tripple_cropped_imgs.append(triple_box_cropped)
    # image augmentaion (hue, contrast,rotation etc) if needed
    cascad_imgs = tripple_cropped_imgs
    if require_augmentation:
       flag = random.randint(0, 3)
       contrast = random.uniform(0.5, 2.5)
       bright = random.uniform(-50, 50)
       rotation = random.randint(-15, 15)
       cascad_imgs = [image_enforcing(x, flag, contrast, bright, rotation) for x in cascad_imgs]
       
  return cascad_imgs    


def img_and_age_data_generator(dataset_df,category,interval,imgs_shape, batch_size,augmentation,dropout):
  dataset_df = dataset_df.reset_index(drop=True)
  df_count = len(dataset_df)
  while True:
    idx = np.random.permutation(df_count) # it will return a list of numbrs (0-df_count), in randomnly arranged
    start = 0
    while start+batch_size < df_count:
      idx_to_get = idx[start:start+batch_size] # making a list of random indexes, to get them from dataset
      current_batch = dataset_df.iloc[idx_to_get] # fetching some list, which is our batch
      #load imgs, transform& create a list
      img_List = []
      two_point_ages = [] # list for 2_point_rep of ages
      for index,row in current_batch.iterrows(): #iterate over batch to load & transform each img
        # load and transform image
        img = image_transform(row, dropout=dropout,target_img_shape=imgs_shape,require_augmentation=augmentation)
        img_List.append(img)
        # make 2_point_represenation(list) of age
        two_point_rep = two_point(int(row.age), category, interval)
        two_point_ages.append(two_point_rep)    

      img_nparray = np.array(img_List) # converting image list to np
      two_point_ages_nparray = np.array(two_point_ages) # converting to np
      out = [current_batch.age.to_numpy(),two_point_ages_nparray] # making list of age_array & 2point_reprseation_array

      # print(len(two_point_ages_nparray[0]))

      yield [img_nparray[:,0], img_nparray[:,1], img_nparray[:,2]], out # return batch
      start += batch_size # update start point, for next batch



def image_enforcing(img, flag, contrast, bright, rotation):
    if flag & 1:  # trans hue
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=bright)
    elif flag & 2:  # rotation
        height, width = img.shape[:-1]
        matRotate = cv2.getRotationMatrix2D((height // 2, width // 2), rotation, 1) # mat rotate 1 center 2 angle 3 缩放系数
        img = cv2.warpAffine(img, matRotate, (height, width))
    elif flag & 4:  # flp 翻转
        img = cv2.flip(img, 1)
    return img

############### Model Related things
def focal_loss(classes_num, gamma=2., alpha=.25, e=0.1):
    # classes_num contains sample number of each classes
    # copy from https://github.com/maozezhong/focal_loss_multi_class/blob/master/focal_loss.py
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor
        '''
        import tensorflow as tf
        from tensorflow.python.ops import array_ops

        #1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        one_minus_p = array_ops.where(tf.greater(target_tensor,zeros), target_tensor - prediction_tensor, zeros)
        FT = -1 * (one_minus_p ** gamma) * tf.math.log(tf.clip_by_value(prediction_tensor, 1e-6, 1.0))

        #2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [ (total_num / ff if ff != 0 else 0.0) for ff in classes_num ]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #scale
        print(classes_w_t2)
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)

        #3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_mean(balanced_fl)

        #4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1-e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(prediction_tensor) / nb_classes, prediction_tensor)
        return fianal_loss
    return focal_loss_fixed








