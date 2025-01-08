import pandas as pd



import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, MaxPooling2D, Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
# Check for GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print('GPU device found and configured')
else:
    print('No GPU devices found, using CPU')


def load_train(path):
    
    """
    It loads the train part of dataset from path
    """
    
    # place your code here
    datagen = ImageDataGenerator(rescale=1./255)
    train_gen_flow = datagen.flow_from_dataframe(
        dataframe=pd.read_csv('/datasets/faces/labels.csv'),
        directory='/datasets/faces/final_files/',
    
        #X represents the input features (images)
        x_col='file_name',
    
        #Y represents the target values (ages)
        y_col='real_age',
    
        target_size=(224, 224),
        batch_size=128,
        class_mode='raw',
        seed=12345)

    return train_gen_flow


def load_test(path):
    
    """
    It loads the train part of dataset from path
    """
    
    # place your code here
    datagen = ImageDataGenerator(rescale=1./255)
    
    test_gen_flow = datagen.flow_from_dataframe(
        dataframe=pd.read_csv('/datasets/faces/labels.csv'),
        
        directory='/datasets/faces/final_files/',
    
        #X represents the input features (images)
        x_col='file_name',
    
        #Y represents the target values (ages)
        y_col='real_age',
    
        target_size=(224, 224),
        batch_size=128,
        class_mode='raw',
        seed=12345)

    return test_gen_flow
    
    """
    It loads the validation/test part of dataset from path
    """
    
    # place your code here
    val_datagen_flow = datagen.flow_from_directory(
    '/datasets/faces/final_files/',
    target_size=(150,150),
    batch_size=32,
    class_mode='other',
    subset='validation',
    seed=12345,
)

    return test_gen_flow


def create_model(input_shape):
    
    """
    It defines the model
    """
    
    # place your code here
    model = Sequential()
    
    
    model.add(
    Conv2D(
        6, (5, 5), padding='same', activation='tanh', input_shape=input_shape)
    )
    
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    
    #64 neurons
    model.add(Dense(64, activation='tanh'))
    
    #1 neuron
    model.add(Dense(1, activation='linear'))
    
    
    optimizer = Adam(learning_rate=0.0005)
    
    # Compile the model with the custom optimizer
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    
    return model


def train_model(model, train_data, test_data, epochs=20, batch_size=32,
                steps_per_epoch=None, validation_steps=None):
    """
    Trains the model given the parameters
    """
    model.fit(
        train_data,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=test_data,
        validation_steps=validation_steps
    )
    return model




if __name__ == '__main__':
    train_data = load_train('/content/drive/My Drive/Colab Notebooks/Sprint15_project/datasets/faces/final_files/labels.csv')
    test_data = load_test('/content/drive/My Drive/Colab Notebooks/Sprint15_project/datasets/faces/final_files/labels.csv')
    model = create_model((224, 224, 3))
    trained_model = train_model(model, train_data, test_data, epochs=20, batch_size=64)
