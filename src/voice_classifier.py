import librosa
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import keras
import datetime
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_selection import mutual_info_classif as mi
import warnings
import tts
import logging

logging.getLogger('tensorflow').disabled = True
warnings.filterwarnings("ignore")
plt.style.use("ggplot")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.serif"] = "Ubuntu"
plt.rcParams["font.monospace"] = "Ubuntu Mono"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["figure.titlesize"] = 12
plt.rcParams["image.cmap"] = "jet"
plt.rcParams["image.interpolation"] = "none"
plt.rcParams["figure.figsize"] = (10, 10)
plt.rcParams["axes.grid"] = True
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["lines.markersize"] = 8

class VoiceClassifier:
    def __init__(self, debug=True):
        self.debug = debug
        
    def train_model(self, mp3, csv, save_model_path="model"):
        """
        Public method
        Preparing data & trains CNN model
        Params:
        mp3 - path to mp3 data
        csv - path to csv data
        save_model_path - url for saving model
        """
        self.number_of_components = 4
        self.build_spectrogram(mp3)
        self.read_csv(csv)
        self.process_pca_data()
        self.select_most_informative_component()
        self.process_time_series_data()
        self.prepare_model()
        self.keras_train_model(30, 50)
        self.save_model(save_model_path)
        
    def exploit_model(self, load_model_path):
        """
        Public method
        Use model
        Params:
        load_model_path - url for loading model
        """
        self.model_predict(load_model_path)
        
    def build_spectrogram(self, mp3):
        """
        Private method
        Build Mel-spectrogram, convert amplitude square to db
        Params:
        mp3 - path to mp3 data
        """
        y, sr = librosa.load(mp3)
        self.duration = librosa.get_duration(y, sr=sr)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20, fmax=8000)
        self.S_dB = librosa.power_to_db(S, ref=np.max)
        if self.debug:
            fig, ax = plt.subplots()
            img = librosa.display.specshow(self.S_dB, x_axis="time", y_axis="mel", sr=sr, fmax=8000, ax=ax)
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            ax.set(title="Mel-frequency spectrogram")
            plt.show()
        
    def read_csv(self, csv):
        """
        Private method
        Reading csv data and do some transformations
        Params:
        csv - path to csv data
        """
        data = pd.read_csv(csv)
        if self.debug:
            print("\n")
            print("Csv data DF")
            display(data.head())
        self.new_data = data[data.seconds<=np.linspace(0, round(self.duration), len(self.S_dB[0])).max()]
        self.new_data.drop_duplicates(subset="speaker")
        if self.debug:
            print("\n")
            sns.countplot(self.new_data.speaker, palette="plasma")
            plt.show()
            
    def process_pca_data(self):
        """
        Private method
        Use PCA to reduce number of features
        """
        audio_data = pd.DataFrame(self.S_dB)
        audio_data = audio_data.T
        if self.debug:
            print("\n")
            print("Audio data DF")
            display(audio_data.head(10))
        pca = PCA(n_components=self.number_of_components)
        pca.fit(audio_data)
        self.pca_data = pd.DataFrame(pca.transform(audio_data))
        audio_time = np.array(self.pca_data.index.tolist()) * self.new_data.seconds.max() / np.array(self.pca_data.index.tolist()).max()
        nd_time_list = [0] + self.new_data.seconds.tolist()
        CLASS = []
        for j in range(len(audio_time)):
            time_j = audio_time[j]
            for i in range(1, len(nd_time_list)):
                start_i = nd_time_list[i - 1]
                end_i = nd_time_list[i]
                if time_j >= start_i and time_j <= end_i:
                    CLASS.append(self.new_data.loc[i - 1].speaker)
        self.pca_data["speaker"] = CLASS[0:len(self.pca_data)]
        self.pca_data["Time"] = audio_time
        if self.debug:
            print("\n")
            print("PCA data with speaker")
            display(self.pca_data.head(10))
            print("\n")
            print("Components pairplot")
            sns.pairplot(self.pca_data, hue="speaker", plot_kws={"s": 1}, palette="plasma")
            plt.show()
        
    def select_most_informative_component(self):
        """
        Private method
        Select the most informative component via mutual information from sklearn
        """
        components = list(range(self.number_of_components))
        components.insert(0, "Time")
        mi_array = mi(X=self.pca_data[components], y=self.pca_data["speaker"])
        self.mic = mi_array[1:].tolist().index(max(mi_array[1:]))
        if self.debug:
            print("\n")
            print("Mutual information for features")
            for i in range(len(mi_array[1:])):
                print("Feature #%d: %f" % (i, mi_array[1:][i]))
            print("The most informative is %d component" % (self.mic))
        
    def process_time_series_data(self):
        """
        Private method
        Using labelencoder for speaker class
        """
        self.time_series_data = self.pca_data[["Time", self.mic, "speaker"]]
        self.time_series_data = self.time_series_data.rename(columns={self.mic: "X"})
        if self.debug:
            print("\n")
            print("Time series scatterplot for #%d component" % (self.mic))
            sns.scatterplot(x='Time', y='X', hue='speaker', data=self.time_series_data, s=10, palette='plasma')
            plt.show()
        label_encoder = LabelEncoder()
        self.time_series_data['speaker'] = label_encoder.fit_transform(self.time_series_data.speaker)
        if self.debug:
            sns.scatterplot(x='Time',y='X',hue='speaker',data=self.time_series_data,s=10,palette='plasma')
            plt.show()

    def prepare_model(self):
        """
        Private method
        Prepare data for CNN model, split data to test and train, mix train data
        """
        self.speaker_list = self.new_data.speaker.unique().tolist()
        self.X = self.time_series_data[['Time','X']]
        self.y = self.time_series_data.speaker
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)
        y_train = np.array(y_train)
        x_train = np.array(X_train).reshape((np.array(X_train).shape[0], np.array(X_train).shape[1], 1))
        self.y_test = np.array(y_test)
        self.x_test = np.array(X_test).reshape((np.array(X_test).shape[0], np.array(X_test).shape[1], 1))
        number_of_classes = len(np.unique(y_train))
        idx = np.random.permutation(len(x_train))
        self.x_train = x_train[idx]
        self.y_train = y_train[idx]
        self.X_test = X_test
        self.model = self.make_model(self.x_train.shape[1:], number_of_classes)

    def make_model(self, shape, number_of_classes):
        """
        Private method
        Create 3-layer CNN model
        Params:
        shape - shape of input data
        number_of_classes - number of data classes
        """
        inputl = keras.layers.Input(shape)
        layer_1 = keras.layers.Conv1D(50, 3, padding="same")(inputl)
        layer_1 = keras.layers.BatchNormalization()(layer_1)
        layer_1 = keras.layers.ReLU()(layer_1)
        layer_2 = keras.layers.Conv1D(50, 3, padding="same")(layer_1)
        layer_2 = keras.layers.BatchNormalization()(layer_2)
        layer_2 = keras.layers.ReLU()(layer_2)
        layer_3 = keras.layers.Conv1D(50, 3, padding="same")(layer_2)
        layer_3 = keras.layers.BatchNormalization()(layer_3)
        layer_3 = keras.layers.ReLU()(layer_3)
        global_average_pooling = keras.layers.GlobalAveragePooling1D()(layer_3)
        outputl = keras.layers.Dense(number_of_classes, activation="softmax")(global_average_pooling)
        return keras.models.Model(inputs=inputl, outputs=outputl)

    def keras_train_model(self, epochs, batch_size):
        """
        Private method
        Train CNN model via keras with some callbacks and adam optimizer
        Params:
        epochs - number of epochs
        batch_size - size of the batch
        """
        epochs = epochs
        batch_size = batch_size
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                "best_model.h5", 
                save_best_only=True, 
                monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", 
                factor=0.5, 
                patience=20, 
                min_lr=0.0001
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=50, 
                verbose=1
            ),
        ]
        print("\n")
        print("Training model. Epochs: %d. Batch size: %d" % (epochs, batch_size))
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=0.2,
            verbose=1,
        )
        if self.debug:
            loss, acc = self.model.evaluate(self.x_test, self.y_test)
            print("\n")
            print("Test accuracy: %.4f" % acc)
            print("Test loss: %.4f" % loss)
            
    def save_model(self, save_model_path):
        """
        Private method
        Save model to disk storage
        Params:
        save_model_path - url for saving model
        """
        self.model.save(save_model_path)
        if self.debug:
            print("Model was saved to ./%s" % save_model_path)
            
        
    def model_predict(self, load_model_path):
        """
        Private method
        Make prediction on test data via CNN trained model
        Params:
        load_model_path - url for loading model
        """
        self.model = keras.models.load_model(load_model_path)
        pred_test = pd.DataFrame(self.model.predict(self.x_test))
        for i in range(len(self.speaker_list)):   
            pred_test = pred_test.rename(columns={i: self.speaker_list[i]})
        if self.debug:
            display(pred_test.head(10))
        test_data = self.X_test.reset_index().drop('index', axis=1)
        test_data['target'] = self.y_test
        for i in range(len(self.speaker_list)):   
            test_data[self.speaker_list[i]] = pred_test[self.speaker_list[i]]
        target_list = test_data.target.tolist()
        for t in range(len(target_list)):
            target_list[t] = self.speaker_list[target_list[t]]
        test_data['target']=target_list
        test_data = test_data.sort_values(by='Time')
        for i in range(len(self.speaker_list)):   
            plt.title('%s Probability' % self.speaker_list[i])
            sns.scatterplot(test_data.Time, test_data.X, s=10, hue=test_data[self.speaker_list[i]])
            plt.show()
        if self.debug:
            self.create_classiffication_report()
            
            
    def create_classiffication_report(self):
        """
        Private method
        Generate confusion matrix and classification report from sklearn
        Params:
        load_model_path - url for loading model
        """
        conf_matrix = confusion_matrix(self.prepare_data_for_report(self.x_test), self.y_test)
        conf_matrix_data = pd.DataFrame(conf_matrix, columns=self.speaker_list)
        conf_matrix_data.index = self.speaker_list
        sns.heatmap(conf_matrix_data.astype(int), fmt='d', annot=True, cmap='plasma')
        plt.yticks(rotation=0)
        plt.xticks(rotation=0, fontsize=10)
        plt.show()
        report = classification_report(self.y_test,
                                   self.prepare_data_for_report(self.x_test),
                                   labels=[0, 1, 2],
                                   target_names=self.speaker_list,
                                   output_dict=True)
        sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='plasma')
        plt.show()
    
    def prepare_data_for_report(self, data):
        """
        Private method
        Prepare input data for classification report
        Params:
        data - data for preparing
        """
        predictions = self.model.predict(data)
        data_for_report = []
        for p in predictions:
            data_for_report.append(np.argmax(p))
        return data_for_report