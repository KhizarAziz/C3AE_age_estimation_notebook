from m_net_training import training_utils
import tensorflow as tf
from keras.layers import Conv2D,Lambda,Input,BatchNormalization,Activation,AveragePooling2D,GlobalAveragePooling2D,Flatten,ReLU,Dense,multiply,Reshape,Concatenate
from keras.activations import sigmoid
from keras.models import Model
from keras.utils import plot_model
from keras import regularizers

def preprocessing(dataframes,batch_size = 32, category=12, interval=10,input_imgs_shape =(64,64), augmentation=True, dropout = 0.2):
  # category: bin + 2 due to two side
  # interval: age interval
  import imp
  imp.reload(training_utils)
  return training_utils.img_and_age_data_generator(dataset_df=dataframes,category=category,interval=interval,imgs_shape=input_imgs_shape, batch_size=batch_size,augmentation=augmentation,dropout=dropout)
  
def white_norm(input): # this is used for normalizing whitish of image, kind of works as increase saturation, contrast & reduce brightness,.... Only included in first layer
  return (input - tf.constant(127.5)) / 128.0

def BRA(input):
  bn = BatchNormalization()(input)
  activtn = Activation('relu')(bn)
  return AveragePooling2D(pool_size=(2,2),strides=(2,2))(activtn)

def SE_BLOCK(input,using_SE=True,r_factor=2):
  channels_count = input.get_shape()[-1]
  act = GlobalAveragePooling2D()(input)
  fc1 = Dense(channels_count//r_factor,activation='relu')(act)
  scale = Dense(channels_count,activation='sigmoid')(fc1)
  return multiply([scale,input])


def build_shared_plain_network(input_height,input_width,input_channels,using_white_norm=True, using_SE=True):
  # design base model here
  input_shape = (input_height,input_width,input_channels)
  input_image = Input(shape=input_shape)
  if using_white_norm:
    wn = Lambda(white_norm,name='white_norm')(input_image)
    conv1 = Conv2D(32,(3,3),use_bias=False)(wn)
  else:
    conv1 = Conv2D(32,(3,3),use_bias=False)(input_image)

  block1 = BRA(input=conv1) # img/filters size reduced by half cuz Avgpooling2D with stride=(2,2)
  block1 = SE_BLOCK(input=block1, using_SE=using_SE)

  conv2 = Conv2D(32, (3, 3), padding="valid", strides=1, name="conv2")(block1)  # param 9248 = 32 * 32 * 3 * 3 + 32
  block2 = BRA(conv2) # img size half
  block2 = SE_BLOCK(block2, using_SE)  # put the se_net after BRA which achived better!!!!

  conv3 = Conv2D(32, (3, 3), padding="valid", strides=1, name="conv3")(block2)  # 9248
  block3 = BRA(conv3)# img size half
  block3 = SE_BLOCK(block3, using_SE)

  #at this point img/filters size is 4x4 so cannot reduce more, so using only B(BN) & R(relu) & excluded A (avgpool)
  conv4 = Conv2D(32,(3,3),use_bias=False,name='conv4')(block3)
  block4 = BatchNormalization()(conv4)
  block4 = Activation(activation='relu')(block4)
  block4 = SE_BLOCK(block4, using_SE)


  conv5 = Conv2D(32, (1, 1), padding="valid", strides=1, name="conv5")(block4)  # 1024 + 32
  conv5 = SE_BLOCK(conv5, using_SE)  # r=16 Not as effective as conv5

  # understand this -1 in reshape
  flat_conv = Reshape((-1,))(conv5)
  # cant find the detail how to change 4*4*32->12, you can try out all dims reduction
  # fc or pooling or any ohter operation
  #shape = map(int, conv5.get_shape()[1:])
  #shrinking_op = Lambda(lambda x: K.reshape(x, (-1, np.prod(shape))))(conv5)

  baseModel = Model(inputs=input_image, outputs=[flat_conv])
  return baseModel



def build_net(Categories=12, input_height=64, input_width=64, input_channels=3, using_white_norm=True, using_SE=True):
    #building basic plain compact model for basic feature extrction
    base_model = build_shared_plain_network(input_height=input_height,input_width=input_width,input_channels=input_channels, using_white_norm=using_white_norm, using_SE=using_SE)

    x1 = Input(shape=(input_height, input_width, input_channels))
    x2 = Input(shape=(input_height, input_width, input_channels))
    x3 = Input(shape=(input_height, input_width, input_channels))

    y1 = base_model(x1)
    y2 = base_model(x2)
    y3 = base_model(x3)

    cfeat = Concatenate(axis=-1)([y1, y2, y3])
    bulk_feat = Dense(Categories, use_bias=True, activity_regularizer=regularizers.l1(0), activation='softmax', name="W1")(cfeat)
    age = Dense(1, name="age")(bulk_feat)
    #gender = Dense(2, activation=softmax, activity_regularizer=regularizers.l2(0), name="gender")(cfeat)

    model = Model(inputs=[x1, x2, x3], outputs=[age, bulk_feat]) 
    #age = Lambda(lambda a: tf.reshape(tf.reduce_sum(a * tf.constant([[x * 10.0 for x in range(12)]]), axis=-1), shape=(-1, 1)), name="age")(bulk_feat)
    return Model(inputs=[x1, x2, x3], outputs=[age, bulk_feat])
