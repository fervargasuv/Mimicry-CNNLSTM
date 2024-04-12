
import numpy as np
from keras.models import Sequential
from keras.layers import TimeDistributed, Conv1D, LSTM, Dropout, MaxPooling1D, Flatten, Dense



# Cargar los datos desde los archivos .npy
X_train= np.load('LSTM/X_train.npy')
Y_train = np.load('LSTM/Y_train.npy')
X_test= np.load('LSTM/X_test.npy')
Y_test= np.load('LSTM/Y_test.npy')


# Define el modelo CNN-LSTM
def define_model(input_shape, output_shape):
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(output_shape, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Función para evaluar el modelo
def evaluate_model(trainX, trainy, testX, testy,epochs,batch_size):
    verbose = 0
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    # reshape data en saltos de tiempos de sub secuencias osea si la secuencia 1 es de 200 frames ahora tengo 4 de 50 frames (4segs)
    n_steps, n_length = 4, 50
    trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
    n_timesteps, n_length, n_features = trainX.shape[1], trainX.shape[2], trainX.shape[3]
    model = define_model((None, n_length, n_features), trainy.shape[1])
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy,model

# Función principal para ejecutar el experimento con Random Search
def run_random_search(trainX, trainy, testX, testy, n_configs, epochs_range, batch_sizes_range):
    best_accuracy = 0.0
    best_config = None
    best_model=None
    
    for _ in range(n_configs):
        epochs = np.random.randint(epochs_range[0], epochs_range[1] + 1)
        batch_size = np.random.choice(batch_sizes_range)
        
        accuracy,model = evaluate_model(trainX, trainy, testX, testy, epochs, batch_size)
        print(f'Config: epochs={epochs}, batch_size={batch_size}, accuracy={accuracy}')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_config = {'epochs': epochs, 'batch_size': batch_size}
            best_model = model  # Guarda el modelo actual como el mejor
    
    print(f'\nBest configuration: epochs={best_config["epochs"]}, batch_size={best_config["batch_size"]}, accuracy={best_accuracy}')
    return best_config, best_accuracy,best_model

# Definir los rangos de búsqueda para epochs y batch_size
epochs_range = (700,800)  
batch_sizes_range = [32,64,128] 

# Ejecutar el Random Search
best_config, best_accuracy, best_model = run_random_search(X_train, Y_train, X_test, Y_test, n_configs=10, epochs_range=epochs_range, batch_sizes_range=batch_sizes_range) #n_configs es para la cantidad de veces que va a probar diferentes tipos de epochs y batch


print("la mejor acurracy es",best_accuracy)
print("la mejor config es",best_config)

# Guardar el mejor modelo
best_model.save('mejor_modelo.keras')

