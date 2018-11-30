from keras.regularizers import l2
from keras.layers import Input, Dense, AveragePooling2D, Conv2D, Conv3D, MaxPooling2D, Dropout, Flatten, concatenate
from keras.models import Model, Sequential


def RetinaFlowNN(input_shape):
    model1 = Sequential()
    model1.add(AveragePooling2D((2, 2), input_shape=input_shape))
    model1.add(Conv2D(64, (5,2), activation='relu', kernel_initializer='glorot_uniform'))
    model1.add(Conv2D(32, (2,4), activation='relu', kernel_initializer='glorot_uniform'))
    model1.add(MaxPooling2D(pool_size=(3, 3)))
    model1.add(Dropout(0.3))

    model1.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='glorot_uniform'))
    model1.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='glorot_uniform'))
    model1.add(Dropout(0.2))
    model1.add(Flatten())

    model1.add(Dense(100, activation='relu', kernel_initializer='glorot_uniform'))
    model1.add(Dense(10, activation='relu', kernel_initializer='glorot_uniform'))
    model1.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform'))
    return model1


def EmbeddingsNN(input_shape):
    model2 = Sequential()
    model2.add(AveragePooling2D((4, 4),input_shape=input_shape))
    model2.add(Conv2D(8, (3, 3), 
                  data_format='channels_last',
                  activation='relu',
                  kernel_initializer='glorot_uniform',
                  kernel_regularizer=l2(0.01))
          )
    model2.add(AveragePooling2D((4, 4)))
    model2.add(Conv2D(16, (3, 3) ,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))
    model2.add(AveragePooling2D((4, 4)))
    model2.add(Flatten())
    model2.add(Dropout(0.5))
    model2.add(Dense(100,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))
    return model2


def DenseFlowNN(input_shape):
    model3 = Sequential()
    model3.add(AveragePooling2D((2, 2),input_shape=input_shape))
    model3.add(Conv2D(8, (3, 3), 
                  data_format='channels_last',
                  activation='relu',
                  kernel_initializer='glorot_uniform',
                  kernel_regularizer=l2(0.01))
          )
    model3.add(AveragePooling2D((2, 2)))
    model3.add(Conv2D(16, (3, 3) ,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))
    model3.add(AveragePooling2D((2, 2)))
    model3.add(Flatten())
    model3.add(Dropout(0.5))
    model3.add(Dense(100,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))
    return model3


def combine_models(input_shapes, models):
    inputs = [Input(shape=input_shape) for input_shape in input_shapes]
    concat = concatenate([model(inp)for inp, model in zip(inputs, models)])
    output = Dense(1, activation='sigmoid',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01))(concat)  
    model = Model(input=inputs,output=output)
    return model