from keras.regularizers import l2
from keras.layers import Input, Dense, concatenate
from keras.models import Model


def combine_models(input_shapes, models):
    inputs = [Input(shape=input_shape) for input_shape in input_shapes]
    concat = concatenate([model(inp)for inp, model in zip(inputs, models)])
    output = Dense(1, activation='sigmoid',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01))(concat)  
    model = Model(input=inputs,output=output)
    return model