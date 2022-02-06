#!/usr/bin/env python
# coding: utf-8

# ## Notebook Requirements

# In[ ]:


get_ipython().system('pip install tensorflow_io')
get_ipython().system('apt-get install rar')


# In[ ]:


import os
from os import path
from google.colab import drive
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
from tqdm import tqdm
from IPython.display import Audio, display
import IPython
from IPython import display
import librosa
import librosa.display
import soundfile as sf
from heapq import merge
from itertools import chain
from collections import namedtuple
import soundfile as sf
import re


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


drive.mount('/content/drive')
if os.path.exists('/content/Data') == False:
  os.mkdir("/content/Data/")
  get_ipython().system('unrar x "/content/drive/MyDrive/Autism Detection/Data/Autism/Cry sounds.rar" "/content/Data"')


# In[ ]:


@tf.function
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

@tf.function
def load_wav_mono(filename):
    """ Load a WAV file, convert it to a float tensor, single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    return wav


# # Voice Activity Detection Using Energy Thresholds

# In[ ]:


SR = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 60
SAMPLE_LENGTH = 0.5 #s
SAMPLE_SIZE = int(np.ceil(SR*SAMPLE_LENGTH))
NOISE_RATIO = 0.25

LABELS = ["ASD", "TD"]


# In[ ]:


from heapq import merge
from itertools import chain
from collections import namedtuple

def remaining(intervals, deleted):
    Event = namedtuple('Event', ['position', 'toggle'])

    int_iter = (Event(position=pos, toggle='in_interval') for pos in chain.from_iterable(intervals))
    del_iter = (Event(position=pos, toggle='in_deleted') for pos in chain.from_iterable(deleted))

    state = {'in_interval': False, 'in_deleted': False}
    start = None
    out = []

    for event in merge(int_iter, del_iter):
        state[event.toggle] = not state[event.toggle]
        if state['in_interval'] and not state['in_deleted']:
            # start a new interval
            start = event.position
        elif start is not None:
            # end an interval. If it's not empty, we append it to the output
            if event.position > start:
                out.append((start, event.position))
            start = None
    return out


# In[ ]:


def oneseq (a,n):
    idx = [i for i, v in enumerate(a)  if not i or a[i-1] != v] + [len(a)]
    res= [r for r in zip(idx, idx[1:]) if r[1] >= r[0]+n]
    for ele in res:
      if a[ele[0]] ==0 :
        res.remove(ele)
    return res

def write_ones(one_intervals, original_duration, length):
  res = []
  with open('/content/Data/times.txt', 'w') as writefile:
    for ele in one_intervals:
        # print(line)
        writefile.write(str(ele[0]/length*original_duration) +',' +str((ele[1]-1)/length*original_duration) + '\n')
        res.append((ele[0]/length*original_duration,(ele[1]-1)/length*original_duration))
  return res


def calc_diff(list1,list2):

  rem1 = remaining(list1, list2)
  rem2 = remaining(list2, list1)
  s1= 0
  s2 =0
  for i in range(len(rem1)):
    s1 = s1+  rem1[i][1] - rem1[i][0]

  for i in range(len(rem2)):
    s2  = s2+ rem2[i][1] - rem2[i][0]

  return s1,s2


# In[ ]:


def read_file(e):
  with open(e, "r") as file1:
    FileList = file1.readlines()

  res = []
  for i in range(len(FileList)):
    res.append( ( float(FileList[i].split(',')[0]) , float((FileList[i].split(',')[1]).split('\n')[0])  ) )
  return res


# In[ ]:


SILENCE = 0.01
SILENCE_DOWN = 0.007
global deleted
deleted = []


# In[ ]:


def envelope(signal, rate, thresh):
    mask = []
    y = pd.Series(signal).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20), min_periods=1, center=True).mean()
    for i in range(len(y_mean)):
      mask.append((y_mean[i] > thresh) or (y_mean[i] > SILENCE_DOWN and y_mean[i] < thresh and ((i > 0 and y_mean[i-1] > SILENCE_DOWN )  or (i < len(y_mean)-1 and y_mean[i+1] > SILENCE_DOWN ))))

    return mask

def load_audio(root, name):
    path = os.path.join(root, name)
    global diff
    signal, rate = librosa.load(path, sr=SR)
    original_duration = librosa.get_duration(y=signal, sr=rate)


    mask = envelope(signal, rate, SILENCE)
    one_intervals = oneseq(mask ,0)
    # print(one_intervals)
    # energy_intervals =write_ones(one_intervals, original_duration,len(mask))
    # our_intervals = read_file('/content/drive/MyDrive/cry_samples/ASD21-5.txt')
    # a,b = calc_diff(energy_intervals,our_intervals)
    # print("First Percentage: ", a*100/original_duration)
    # print("First Percentage: ", b*100/original_duration)
    # print("Total Percentage: ", a*100/original_duration + b*100/original_duration)

    signal = signal[mask]
    if not os.path.exists(root.replace('original' , 'processed')):
      os.mkdir(root.replace('original' , 'processed'))
    sf.write(path.replace('original' , 'processed'), signal, rate)
    return signal


# In[ ]:


root = './Data/Cry sounds/original'
if path.exists("/content/Data/Cry sounds/processed"):
  shutil.rmtree("/content/Data/Cry sounds/processed")
if not path.exists("/content/Data/Cry sounds/processed"):
  os.mkdir("/content/Data/Cry sounds/processed")
for root, directories, files in os.walk(root, topdown=False):
  for name in files:
    testing_wav_file_name = os.path.join(root, name)
    #if root == "./Data/Cry sounds/original/ASD1":
    load_audio(root, name)
    print(testing_wav_file_name, ": Done!")


# In[ ]:


# if not path.exists("/content/drive/MyDrive/Autism Detection/Data/Autism/processed"):
#   os.mkdir("/content/drive/MyDrive/Autism Detection/Data/Autism/processed")
# !cp -av  "/content/Data/Cry sounds/processed" "/content/drive/MyDrive/Autism Detection/Data/Autism"


# # Clean Cry Detection Using YAMNet

# ## Initiailization

# In[ ]:


yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)
my_classes = ["TD", "ASD"]
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names = list(pd.read_csv(class_map_path)['display_name'])


# In[ ]:


if not os.path.exists("/content/Data/Cry sounds/processed"):
  os.mkdir("/content/Data/Cry sounds/processed")
get_ipython().system('cp -av "/content/drive/MyDrive/Autism Detection/Data/Autism/processed" "/content/Data/Cry sounds/processed"')


# ## Visualization

# In[ ]:


persons = ['ASD1', 'ASD7', 'ASD11','ASD21', 'TD1', 'TD4']
folders = [os.path.join('./Data/Cry sounds/processed/processed', person) for person in persons]
for folder in folders:
  for root, directories, files in os.walk(folder, topdown=False):
    for name in files:
      testing_wav_file_name = os.path.join(root, name)
      print(testing_wav_file_name)
      wav_data = load_wav_mono(testing_wav_file_name)
      voice_data = load_wav_16k_mono(testing_wav_file_name)
      scores, embeddings, spectrogram = yamnet_model(voice_data)
      class_scores = tf.reduce_mean(scores, axis=0)
      top_class = tf.argmax(class_scores)
      inferred_class = class_names[top_class]

      print(f'The main sound is: {inferred_class}')
      print(f'The embeddings shape: {embeddings.shape}')
      IPython.display.display(IPython.display.Audio(voice_data,rate=16000))
      plt.figure(figsize=(20, 16))

      # Plot the waveform.
      plt.subplot(3, 1, 1)
      plt.plot(voice_data)
      plt.xlim([0, len(voice_data)])
      for root2, directories2, files2 in os.walk(folder.replace("original", "cleaned"), topdown=False):
        for name2 in files2:
          if (name2.split(".")[0])[0] == (name.split(".")[0])[0]:
            segment_data = load_wav_mono(os.path.join(root2, name2))
            indices = np.where(wav_data == segment_data[0])
            for i in indices[0]:
              
              flag = True
              for j in range(10):
                if segment_data[j] != wav_data[i+j]:
                  flag = False
              if flag:
                index = i
                r = random.random()
                b = random.random()
                g = random.random()
                color = (r, g, b)
                start = int(index*(16/44.1))
                end = int((index+len(segment_data))*(16/44.1))
                print("start:", start, "- end:", end)
                print(start/16000, end/16000)
                plt.axvline(x=start, color=color)
                plt.axvline(x=end, color=color)
                break
      # Plot the log-mel spectrogram (returned by the model).
      plt.subplot(3, 1, 2)
      plt.imshow(spectrogram.numpy().T, aspect='auto', interpolation='nearest', origin='lower')

      # Plot and label the model output scores for the top-scoring classes.
      mean_scores = np.mean(scores, axis=0)
      top_n = 10
      top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
      plt.subplot(3, 1, 3)
      plt.imshow(scores.numpy()[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')

      # patch_padding = (PATCH_WINDOW_SECONDS / 2) / PATCH_HOP_SECONDS
      # values from the model documentation
      patch_padding = (0.025 / 2) / 0.01
      plt.xlim([-patch_padding-0.5, scores.shape[0] + patch_padding-0.5])
      # Label the top_N classes.
      yticks = range(0, top_n, 1)
      plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
      _ = plt.ylim(-0.5 + np.array([top_n, 0]))
      plt.show()
      cry_probability = scores[:,19] + scores[:,20]
      plt.plot(cry_probability)
      plt.show()


# In[ ]:


def segmentize_audio(root, name, cry_threshold):
  path = os.path.join(root, name)
  voice_data = load_wav_16k_mono(path)
  scores, embeddings, spectrogram = yamnet_model(voice_data)
  cry_probability = scores[:,19] + scores[:,20]
  cry_or_not = cry_probability > cry_threshold
  cry_or_not = np.array(cry_or_not)
  first = -1
  num_segments = 0
  for segment in range(len(cry_or_not)):
    if first == -1 and cry_or_not[segment] == 1:
      first = segment
    if first != -1 and (cry_or_not[segment] != 1 or segment == (len(cry_or_not) - 1)):
      signal, rate = librosa.load(path, sr=16000)
      start = first * 0.48
      end = segment * 0.48 + 0.48
      if segment == (len(cry_or_not) - 1):
        signal = signal[int(start * 16000):]
      else:
        signal = signal[int(start * 16000):int(end * 16000)]
      if not os.path.exists(root.replace('processed' , 'segments')):
        os.mkdir(root.replace('processed' , 'segments'))
      segment_name = path.replace('processed' , 'segments')
      segment_name = segment_name.replace('.wav', '')
      segment_name = segment_name.replace('.WAV', '')
      segment_name = segment_name + '_' + str(num_segments) + '.wav'
      sf.write(segment_name, signal, rate)
      num_segments += 1
      first = -1
  return num_segments


# ## Processing

# In[ ]:


cry_threshold = 0.2

root = './Data/Cry sounds/processed'
if os.path.exists("/content/Data/Cry sounds/segments"):
  shutil.rmtree("/content/Data/Cry sounds/segments")
os.mkdir("/content/Data/Cry sounds/segments")
total_segments = 0
for root, directories, files in os.walk(root, topdown=False):
  for name in files:
    testing_wav_file_name = os.path.join(root, name)
    total_segments += segmentize_audio(root, name, cry_threshold)


# In[ ]:


# if not os.path.exists("/content/drive/MyDrive/Autism Detection/Data/Autism/segments"):
#   os.mkdir("/content/drive/MyDrive/Autism Detection/Data/Autism/segments")
# !cp -av  "/content/Data/Cry sounds/segments" "/content/drive/MyDrive/Autism Detection/Data/Autism"


# In[ ]:


total_segments


# # Autism Classification

# In[ ]:


if os.path.exists("/content/Data/Cry sounds/segments"):
  shutil.rmtree("/content/Data/Cry sounds/segments")
if not os.path.exists("/content/Data/Cry sounds/segments"):
  os.mkdir("/content/Data/Cry sounds/segments")
get_ipython().system('cp -av "/content/drive/MyDrive/Autism Detection/Data/Autism/segments" "/content/Data/Cry sounds"')


# ## Without K-fold

# In[ ]:


path = "/content/Data/Cry sounds/segments"
filenames = []
targets = []
folds = []
ASD_counter = 0
TD_counter = 0
for root, directories, files in os.walk(path, topdown=False):
  for name in files:
    if (sf.SoundFile(os.path.join(root, name))).subtype == "PCM_U8":
      continue
    if "ASD" in os.path.join(root, name):   
      targets.append(1)
      filenames.append(os.path.join(root, name))
    elif "TD" in os.path.join(root, name):
      for i in range(2):
        targets.append(0)
        filenames.append(os.path.join(root, name))
for root, directories, files in os.walk(path, topdown=False):
  for name in files:
    if (sf.SoundFile(os.path.join(root, name))).subtype == "PCM_U8":
      continue
    if "ASD" in os.path.join(root, name):
      if ASD_counter < len(filenames) * 0.4:
        folds.append((ASD_counter%4) + 1)
      else:
        folds.append(5)
      ASD_counter += 1
    elif "TD" in os.path.join(root, name):
      for i in range(2):
        if TD_counter < len(filenames) * 0.4:
          folds.append((TD_counter%4) + 1)
        else:
          folds.append(5)
        TD_counter += 1


print("Number of ASD samples:", ASD_counter)
print("Number of TD samples:", int(TD_counter / 2))
print(len(filenames))
list_of_tuples = list(zip(filenames, folds, targets)) 
df = pd.DataFrame(list_of_tuples, columns = ['filename', 'fold', 'target'])


# In[ ]:


(unique, counts) = np.unique(folds, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)


# In[ ]:


filenames = df['filename']
targets = df['target']
folds = df['fold']

main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))
main_ds.element_spec


# In[ ]:


def load_wav_for_map(filename, label, fold):
  return load_wav_16k_mono(filename), label, fold

main_ds = main_ds.map(load_wav_for_map)
main_ds.element_spec


# In[ ]:


# applies the embedding extraction model to a wav data
def extract_embedding(wav_data, label, fold):
  ''' run YAMNet to extract embedding from the wav data '''
  scores, embeddings, spectrogram = yamnet_model(wav_data)
  num_embeddings = tf.shape(embeddings)[0]
  return (embeddings,
            tf.repeat(label, num_embeddings),
            tf.repeat(fold, num_embeddings))

# extract embedding
main_ds = main_ds.map(extract_embedding).unbatch()
main_ds.element_spec


# In[ ]:


cached_ds = main_ds.cache()
train_ds = cached_ds.filter(lambda embedding, label, fold: fold <= 4)
val_ds = cached_ds.filter(lambda embedding, label, fold: fold == 5)

# remove the folds column now that it's not needed anymore
remove_fold_column = lambda embedding, label, fold: (embedding, label)

train_ds = train_ds.map(remove_fold_column)
val_ds = val_ds.map(remove_fold_column)

train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)


# ### YAMNet

# In[ ]:


my_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                          name='input_embedding'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2)
], name='my_model')

my_model.summary()


# In[ ]:


my_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 optimizer="adam",
                 metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                            patience=5,
                                            restore_best_weights=True)


# In[ ]:


history = my_model.fit(train_ds,
                       epochs=100,
                       validation_data=val_ds,
                       callbacks=callback)


# In[ ]:


y_pred = []
for f in filenames:
  scores, embeddings, spectrogram = yamnet_model(load_wav_16k_mono(f))
  result = my_model(embeddings).numpy()
  inferred_class = result.mean(axis=0).argmax()
  y_pred.append(inferred_class)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(targets, y_pred)


# ### VGGish

# In[ ]:


VGGish_model_handle = 'https://tfhub.dev/google/vggish/1'
VGGish_model = hub.load(VGGish_model_handle)
my_classes = ["TD", "ASD"]


# In[ ]:


filenames = df['filename']
targets = df['target']
folds = df['fold']

main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))
main_ds.element_spec
main_ds = main_ds.map(load_wav_for_map)
main_ds.element_spec


# In[ ]:


# applies the embedding extraction model to a wav data
def extract_embedding(wav_data, label, fold):
  ''' run VGGish to extract embedding from the wav data '''
  embeddings = VGGish_model(wav_data)
  num_embeddings = tf.shape(embeddings)[0]
  return (embeddings,
            tf.repeat(label, num_embeddings),
            tf.repeat(fold, num_embeddings))

# extract embedding
main_ds = main_ds.map(extract_embedding).unbatch()
main_ds.element_spec


# In[ ]:


cached_ds = main_ds.cache()
train_ds = cached_ds.filter(lambda embedding, label, fold: fold <= 4)
val_ds = cached_ds.filter(lambda embedding, label, fold: fold == 5)

# remove the folds column now that it's not needed anymore
remove_fold_column = lambda embedding, label, fold: (embedding, label)

train_ds = train_ds.map(remove_fold_column)
val_ds = val_ds.map(remove_fold_column)

train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)


# In[ ]:


my_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128), dtype=tf.float32,
                          name='input_embedding'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2)
], name='my_model')

my_model.summary()


# In[ ]:


my_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 optimizer="adam",
                 metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                            patience=10,
                                            restore_best_weights=True)


# In[ ]:


history = my_model.fit(train_ds,
                       epochs=100,
                       validation_data=val_ds,
                       callbacks=callback)


# In[ ]:


y_pred = []
for f in filenames:
  embeddings = VGGish_model(load_wav_16k_mono(f))
  result = my_model(embeddings).numpy()
  inferred_class = result.mean(axis=0).argmax()
  y_pred.append(inferred_class)
from sklearn.metrics import confusion_matrix
confusion_matrix(targets, y_pred)


# ## With K-fold

# In[ ]:


path = "/content/Data/Cry sounds/segments"
filenames = []
targets = []
folds = []
TD_counter = 0
ASD_counter = 0
for root, directories, files in os.walk(path, topdown=False):
  for name in files:
    folder_name = root.split("/")[-1]
    folder_number = folder_name.replace("TD", "")
    folder_number = folder_number.replace("ASD", "")
    if (sf.SoundFile(os.path.join(root, name))).subtype == "PCM_U8":
      continue
    if int(folder_number) > 10:
      continue
    if "ASD" in os.path.join(root, name):   
      targets.append(1)
      filenames.append(os.path.join(root, name))
      folds.append(int(folder_number) - 1)
      ASD_counter += 1
    elif "TD" in os.path.join(root, name):
      for i in range(2):
        targets.append(0)
        filenames.append(os.path.join(root, name))
        folds.append(int(folder_number) - 1)
      TD_counter += 1
    

print("Number of ASD samples:", ASD_counter)
print("Number of TD samples:", TD_counter)
print(len(filenames))
list_of_tuples = list(zip(filenames, folds, targets)) 
df = pd.DataFrame(list_of_tuples, columns = ['filename', 'fold', 'target'])


# In[ ]:


(unique, counts) = np.unique(folds, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)


# In[ ]:


filenames = df['filename']
targets = df['target']
folds = df['fold']

main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))
main_ds.element_spec


# In[ ]:


def load_wav_for_map(filename, label, fold):
  return load_wav_16k_mono(filename), label, fold

main_ds = main_ds.map(load_wav_for_map)
main_ds.element_spec


# In[ ]:


yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)
my_classes = ["TD", "ASD"]
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names = list(pd.read_csv(class_map_path)['display_name'])


# In[ ]:


# applies the embedding extraction model to a wav data
def extract_embedding(wav_data, label, fold):
  ''' run YAMNet to extract embedding from the wav data '''
  scores, embeddings, spectrogram = yamnet_model(wav_data)
  num_embeddings = tf.shape(embeddings)[0]
  return (embeddings,
            tf.repeat(label, num_embeddings),
            tf.repeat(fold, num_embeddings))

# extract embedding
main_ds = main_ds.map(extract_embedding).unbatch()
main_ds.element_spec


# ### YAMNet

# In[ ]:


models = []
train_accuracies = []
val_accuracies = []
for i in range(10):
    print("Test set is", i)
    cached_ds = main_ds.cache()
    train_ds = cached_ds.filter(lambda embedding, label, fold: fold != i)
    val_ds = cached_ds.filter(lambda embedding, label, fold: fold == i)

    # remove the folds column now that it's not needed anymore
    remove_fold_column = lambda embedding, label, fold: (embedding, label)

    train_ds = train_ds.map(remove_fold_column)
    val_ds = val_ds.map(remove_fold_column)

    train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
    regularizer = tf.keras.regularizers.l2(l2=0.5)
    my_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                              name='input_embedding'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax', kernel_regularizer = regularizer)
    ], name='my_model')

    my_model.summary()
    my_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    optimizer="adam",
                    metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=10,
                                                restore_best_weights=True)
    history = my_model.fit(train_ds,
                          epochs=100,
                          validation_data=val_ds,
                          callbacks=callback)
    history = history.history
    acc_val = history['val_accuracy']
    acc_tra = history['accuracy']
    # plt.figure()
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.plot(acc_val, 'darkred')
    # plt.plot(acc_tra, 'darkblue')
    # plt.legend(['Validation','Training'])
    # plt.grid(c='lightgrey')
    # plt.show()
    train_accuracies.append(acc_tra[-11])
    val_accuracies.append(acc_val[-11])
    models.append(my_model)


# In[ ]:


for i in range(10):
    print("test fold is", i)
    print("train :", train_accuracies[i])
    print("test :", val_accuracies[i])


# In[ ]:


print(sum(train_accuracies) / len(train_accuracies))
print(sum(val_accuracies) / len(val_accuracies))


# In[ ]:


y_pred = []
for i in range(len(filenames)):
  f = filenames[i]
  scores, embeddings, spectrogram = yamnet_model(load_wav_16k_mono(f))
  result = models[folds[i]](embeddings).numpy()
  inferred_class = result.mean(axis=0).argmax()
  y_pred.append(inferred_class)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(targets, y_pred)


# In[ ]:


predicted = []
for c in ["TD", "ASD"]:
  print(c, ":")
  for i in range(1, 11):
    ASD_votes = 0
    TD_votes = 0
    for j in range(len(filenames)):
      if (c + str(i)) in filenames[j]:
        if y_pred[j]:
          ASD_votes += 1
        else:
          TD_votes += 1
    print(ASD_votes + TD_votes, ASD_votes, TD_votes)
    if ASD_votes >= (0.4 * (ASD_votes + TD_votes)):
      predicted.append(1)
    else:
      predicted.append(0)
print("TD:", predicted[:10])
print("ASD:", predicted[10:])


# In[ ]:





# In[ ]:


models = []
train_accuracies = []
val_accuracies = []
for i in range(10):
  print("Test set is", i)
  cached_ds = main_ds.cache()
  train_ds = cached_ds.filter(lambda embedding, label, fold: fold != i)
  val_ds = cached_ds.filter(lambda embedding, label, fold: fold == i)

  # remove the folds column now that it's not needed anymore
  remove_fold_column = lambda embedding, label, fold: (embedding, label)

  train_ds = train_ds.map(remove_fold_column)
  val_ds = val_ds.map(remove_fold_column)

  train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
  val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
  regularizer = tf.keras.regularizers.l2(l2=0.001)
  my_model = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                            name='input_embedding'),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dense(2, activation='softmax', kernel_regularizer = regularizer)
  ], name='my_model')

  my_model.summary()
  my_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer="adam",
                  metrics=['accuracy'])

  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=10,
                                              restore_best_weights=True)
  history = my_model.fit(train_ds,
                        epochs=100,
                        validation_data=val_ds,
                        callbacks=callback)
  history = history.history
  acc_val = history['val_accuracy']
  acc_tra = history['accuracy']
  # plt.figure()
  # plt.xlabel('Epochs')
  # plt.ylabel('Accuracy')
  # plt.plot(acc_val, 'darkred')
  # plt.plot(acc_tra, 'darkblue')
  # plt.legend(['Validation','Training'])
  # plt.grid(c='lightgrey')
  # plt.show()
  train_accuracies.append(acc_tra[-11])
  val_accuracies.append(acc_val[-11])
  models.append(my_model)


# In[ ]:


for i in range(10):
  print("test fold is", i)
  print("train :", train_accuracies[i])
  print("test :", val_accuracies[i])


# In[ ]:


print(sum(train_accuracies) / len(train_accuracies))
print(sum(val_accuracies) / len(val_accuracies))


# In[ ]:


def most_frequent(List):
  return max(set(List), key = List.count)
y_pred = []
for f in filenames:
  scores, embeddings, spectrogram = yamnet_model(load_wav_16k_mono(f))
  temp = []
  for i in range(10):
    result = models[i](embeddings).numpy()
    inferred_class = result.mean(axis=0).argmax()
    temp.append(inferred_class)
  y_pred.append(most_frequent(temp))


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(targets, y_pred)


# In[ ]:


predicted = []
for c in ["TD", "ASD"]:
  for i in range(1, 11):
    ASD_votes = 0
    TD_votes = 0
    for j in range(len(filenames)):
      if (c + str(i)) in filenames[j]:
        if y_pred[j]:
          ASD_votes += 1
        else:
          TD_votes += 1
    print(ASD_votes + TD_votes, ASD_votes, TD_votes)
    if ASD_votes >= (0.23 * (ASD_votes + TD_votes)):
      predicted.append(1)
    else:
      predicted.append(0)
print("TD:", predicted[:10])
print("ASD:", predicted[10:])


# In[ ]:


path = "/content/Data/Cry sounds/segments"
filenames = []
targets = []
folds = []
TD_counter = 0
ASD_counter = 0
for root, directories, files in os.walk(path, topdown=False):
  for name in files:
    folder_name = root.split("/")[-1]
    folder_number = folder_name.replace("TD", "")
    folder_number = folder_number.replace("ASD", "")
    if (sf.SoundFile(os.path.join(root, name))).subtype == "PCM_U8":
      continue
    if "ASD" in os.path.join(root, name):   
      targets.append(1)
      filenames.append(os.path.join(root, name))
      folds.append(int(folder_number) - 1)
      ASD_counter += 1
    elif "TD" in os.path.join(root, name):
      for i in range(2):
        targets.append(0)
        filenames.append(os.path.join(root, name))
        folds.append(int(folder_number) - 1)
      TD_counter += 1
    

print("Number of ASD samples:", ASD_counter)
print("Number of TD samples:", TD_counter)
print(len(filenames))
list_of_tuples = list(zip(filenames, folds, targets)) 
df = pd.DataFrame(list_of_tuples, columns = ['filename', 'fold', 'target'])


# In[ ]:


def most_frequent(List):
  return max(set(List), key = List.count)
y_pred = []
for f in filenames:
  scores, embeddings, spectrogram = yamnet_model(load_wav_16k_mono(f))
  temp = []
  for i in range(10):
    result = models[i](embeddings).numpy()
    inferred_class = result.mean(axis=0).argmax()
    temp.append(inferred_class)
  y_pred.append(most_frequent(temp))


# In[ ]:


predicted = []
for c in ["TD", "ASD"]:
  for i in range(1, 32):
    ASD_votes = 0
    TD_votes = 0
    for j in range(len(filenames)):
      if (c + str(i)) in filenames[j]:
        if y_pred[j]:
          ASD_votes += 1
        else:
          TD_votes += 1
    #print(ASD_votes + TD_votes, ASD_votes, TD_votes)
    if ASD_votes >= (0.17 * (ASD_votes + TD_votes)):
      predicted.append(1)
    else:
      predicted.append(0)
#print(predicted)
counter = 0
correct_TD = 0
correct_ASD = 0
for i in range(len(predicted)):
  if counter <= 30 and predicted[i] == 0:
    correct_TD += 1
  elif counter > 30 and predicted[i] == 1:
    correct_ASD += 1
  counter += 1
print(correct_TD, correct_ASD)
# print("TD :", predicted[:30])
# print("ASD :", predicted[31:])


# ### VGGish

# In[ ]:


VGGish_model_handle = 'https://tfhub.dev/google/vggish/1'
VGGish_model = hub.load(VGGish_model_handle)
my_classes = ["TD", "ASD"]


# In[ ]:


filenames = df['filename']
targets = df['target']
folds = df['fold']

main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))
main_ds.element_spec
main_ds = main_ds.map(load_wav_for_map)
main_ds.element_spec


# In[ ]:


# applies the embedding extraction model to a wav data
def extract_embedding(wav_data, label, fold):
  ''' run VGGish to extract embedding from the wav data '''
  embeddings = VGGish_model(wav_data)
  num_embeddings = tf.shape(embeddings)[0]
  return (embeddings,
            tf.repeat(label, num_embeddings),
            tf.repeat(fold, num_embeddings))

# extract embedding
main_ds = main_ds.map(extract_embedding).unbatch()
main_ds.element_spec


# In[ ]:


train_accuracies = []
val_accuracies = []
for i in range(10):
  print("Test set is", i)
  cached_ds = main_ds.cache()
  train_ds = cached_ds.filter(lambda embedding, label, fold: fold != i)
  val_ds = cached_ds.filter(lambda embedding, label, fold: fold == i)

  # remove the folds column now that it's not needed anymore
  remove_fold_column = lambda embedding, label, fold: (embedding, label)

  train_ds = train_ds.map(remove_fold_column)
  val_ds = val_ds.map(remove_fold_column)

  train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
  val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
  for j in range(3):
    my_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128), dtype=tf.float32,
                              name='input_embedding'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2)
    ], name='my_model')

    my_model.summary()
    my_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    optimizer="adam",
                    metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                patience=5,
                                                restore_best_weights=True)
    history = my_model.fit(train_ds,
                          epochs=100,
                          validation_data=val_ds,
                          callbacks=callback)
    history = history.history
    acc_val = history['val_accuracy']
    acc_tra = history['accuracy']
    train_accuracies.append(acc_tra[-6])
    val_accuracies.append(acc_val[-6])


# In[ ]:


for i in range(10):
  print("test fold is", i)
  temp = train_accuracies[i * 3: (i + 1) * 3]
  print("train :", sum(temp) / len(temp))
  temp = val_accuracies[i * 3: (i + 1) * 3]
  print("test :", sum(temp) / len(temp))


# In[ ]:


print(sum(train_accuracies) / len(train_accuracies))
print(sum(val_accuracies) / len(val_accuracies))

