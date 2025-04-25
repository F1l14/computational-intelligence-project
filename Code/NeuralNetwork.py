import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.regularizers import l2
import matplotlib.pyplot as plt
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)
filepath = '/content/drive/MyDrive/AI/'
dataset = pd.read_csv(filepath + "preprocessed.csv")


# Split into input and output
X = dataset.iloc[:, :-1].values  # Select all rows and all columns except the last one
Y = dataset.iloc[:, -1].values 

# Initialize StratifiedKFold
strat_kfold = StratifiedKFold(n_splits=5, shuffle=True)

# Lists to store results
ceList = []
mseList = []
accList = []

epoch_ce_list = []
epoch_mse_list = []  
epoch_acc_list = [] 

r=0.01

# Cross-validation loop
for i, (train, test) in enumerate(strat_kfold.split(X, Y)):
    # Create model
    model = Sequential()
    model.add(Dense(78, activation="relu", input_dim=X.shape[1], kernel_regularizer=l2(r)))  # First hidden layer
    model.add(Dense(1, activation="sigmoid", kernel_regularizer=l2(r)))                      # Output layer for binary classification

    # Early stopping on validation loss
    early_stop = EarlyStopping(
        monitor='loss',          
        patience=10,
        restore_best_weights=True,
        mode='min'
    )

    # Compile model
    optimizer = SGD(learning_rate=0.1, momentum=0.6)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['mse', 'accuracy']
    )

    # Fit model
    history = model.fit(
        X[train], Y[train],
        epochs=500,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0
    )


    epoch_ce_list.append(history.history['loss'])  # Cross-entropy loss for each epoch
    epoch_mse_list.append(history.history['mse'])  # MSE for each epoch
    epoch_acc_list.append(history.history['accuracy'])  # Accuracy for each epoch

    # Evaluate model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    # scores = [loss, mse, accuracy]
    ceList.append(scores[0])   # Cross-entropy loss
    mseList.append(scores[1])  # MSE
    accList.append(scores[2])  # Accuracy

    print(f"Fold {i} - CE: {scores[0]:.4f}, MSE: {scores[1]:.4f}, ACC: {scores[2]:.4f}")

# Print average scores
print("\nAverage Results:")
print("Cross-Entropy Loss (CE):", np.mean(ceList))
print("Mean Squared Error (MSE):", np.mean(mseList))
print("Accuracy (ACC):", np.mean(accList))


# Plotting the metrics for each fold
for i in range(len(epoch_ce_list)):
    # Plot Cross-Entropy Loss per epoch for each fold
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(epoch_ce_list[i], label=f'Fold {i}')
    plt.title('Cross-Entropy Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot MSE per epoch for each fold
    plt.subplot(1, 3, 2)
    plt.plot(epoch_mse_list[i], label=f'Fold {i}')
    plt.title('Mean Squared Error per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    # Plot Accuracy per epoch for each fold
    plt.subplot(1, 3, 3)
    plt.plot(epoch_acc_list[i], label=f'Fold {i}')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Show the plots for each fold
    plt.tight_layout()
    plt.show()
