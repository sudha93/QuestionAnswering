import keras
from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, Dense, Embedding
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import io
from nltk import word_tokenize 
import numpy as np
import pickle
import sys 
import json
from keras import backend as K
print K.tensorflow_backend._get_available_gpus()
# This is a code for Question Answering. I'm training a siamese network for sentence 
# similarity. I'm using this network to get the question encodings for all the questions.
# when an new question is given , based on the encodings , it finds the nearest neighbour 
# question and displays its answer 

glove = "/DATA1/USERS/anirudh/glove6B/glove.6B.50d.txt"
trainData = "/DATA1/USERS/anirudh/miniprojects/QA/quora_duplicate_questions.tsv"
VECTOR_LEN = 50
THRESHOLD = 20
# no.of partitions of training data 
NUMBER = 4 
# no.of samples to test 
TEST = 5000

def preprocess(trainData,THRESHOLD) :
    # Preprocess the text and store in a list format
    # Each item in the list is a list containing score,sentence1,sentence2 in that order
    train = []
    with open(trainData) as f:
        lines = f.readlines() #decode("latin-1")
        #print lines 
        #lines = lines[:10000] # need to be commneted 
        for i in range(len(lines)):
            linesplit = lines[i].split('\t')
            for j in range(len(linesplit)):
                #linesplit[j] = linesplit[j].decode('ascii','ignore').strip()
                linesplit[j] = linesplit[j].strip()
            train.append(linesplit[3:6]) 
    #print train[:10]
    #print len(train)
    del train[0]
    #print train 
    # Removing the list elements which have both sentences' length above a threshold 
    count = 0
    new = []
    # iterating through the list 
    for i in range(len(train)) :
        #print i 
        #print train[i][0]
        try:
            #print train[i][0]
            sent1 = word_tokenize(train[i][0])
            #print train[i][1]
            sent2 = word_tokenize(train[i][1])
        except:
            continue
        if (len(sent1) <= THRESHOLD and len(sent2)<=THRESHOLD):
            output = float(train[i][2])
            new.append([output, sent1, sent2])

    return new  


# Create a dictionary with ids for all the words/tokens in the corpus
# Giving a common id for all the words not found in glove 
def dictionary(train,embedding_index):
    idDict = {}
    for item in train:
        #print item 
        for token in item:
            #print token 
            #token = token.encode('utf-8')
            # if token doesnt exist in the dictionary
            if idDict.get(token) is None:
                # if its an UNKNOWN word, assign id -1 to it 
                if embedding_index.get(token) is None:
                    idDict[token] = -1
                # if its a word that exists in glove  
                else :
                    if len(idDict) == 0:
                        idDict[token] = 1
                    else:
                        k = max(idDict, key=idDict.get)
                        # k is the key of the highest value in the list 
                        highestId = idDict[k]
                        if highestId >= 1 :
                            idDict[token] = highestId + 1
                        else :
                            idDict[token] = 1

    # All the unknown words are given id -1 , now lets replace it with highestvalue+1
    maxKey = max(idDict, key=idDict.get)
    highest = idDict[maxKey]
    for word,value in idDict.items():
        if value == -1:
            idDict[word] = highest+1

    return idDict

def convert(train,idDict):
    for i,item in enumerate(train):
        #print item 
        for j,token in enumerate(item):
            #print token 
            train[i][j] = idDict[token]
    return train

def partition(lst,n):
    div = len(lst)/float(n)
    return [ lst[int(round(div * i)): int(round(div * (i + 1)))] for i in xrange(n) ]

def extract(lst):
    for item in lst:
        qList.append(item['question'])
        aList.append(item['answer'])

# Main code
# This commented code has been run,and the data is stored using pickle.So, its commented
'''
train = preprocess(trainData,THRESHOLD)
print 'length ',len(train)
testOP = [item[0] for item in train[:TEST]]
test_a = [item[1] for item in train[:TEST]]
test_b = [item[2] for item in train[:TEST]]
#print test_a
trainOP = [item[0] for item in train[TEST:]]
train_a = [item[1] for item in train[TEST:]]
train_b = [item[2] for item in train[TEST:]]

# Loading the glove embeddings into a dictionary with word as key and embedding as value
f = open(glove)
embedding_index = {}
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32') 
    embedding_index[word] = coefs 
f.close()
#creating a tokenlist 
tokenlist = train_a+train_b + test_a + test_b
#print tokenlist 
idDict = dictionary(tokenlist,embedding_index)
# convert the tokens to ids 
idtrain_a = convert(train_a,idDict)
idtrain_b = convert(train_b,idDict)
idtest_a = convert(test_a, idDict)
idtest_b = convert(test_b,idDict)
# padding 
idtrain_a = pad_sequences(idtrain_a,maxlen=THRESHOLD,padding='pre',truncating= 'post', value=0.0)
idtrain_b = pad_sequences(idtrain_b,maxlen=THRESHOLD,padding='pre',truncating= 'post', value=0.0)
idtest_a = pad_sequences(idtest_a,maxlen=THRESHOLD,padding='pre',truncating= 'post', value=0.0)
idtest_b = pad_sequences(idtest_b,maxlen=THRESHOLD,padding='pre',truncating= 'post', value=0.0)

# getting the list of partitioned training data
totalOP = partition(trainOP,NUMBER)
total_a = partition(idtrain_a,NUMBER)
total_b = partition(idtrain_b,NUMBER)
# Here each sublist is a partition
#print idtest_a

# Creating the embedding matrix 
maxKey = max(idDict, key=idDict.get)
highestValue = idDict[maxKey]
VOCAB_SIZE = highestValue +1 
embedMatrix = np.zeros((VOCAB_SIZE,VECTOR_LEN))
vector = np.random.rand(VECTOR_LEN)
for key,value in idDict.items():
    if value != 0 :
        embed = embedding_index.get(key)
        if embed is None :
            embedMatrix[value] = vector 
        else :
            embedMatrix[value] = embed 

# Using pickle to store lists
pickle.dump(totalOP,open('totalOP.pkl','w'))
pickle.dump(total_a,open('total_a.pkl','w'))
pickle.dump(total_b,open('total_b.pkl','w'))

pickle.dump(testOP,open('testOP.pkl','w'))
pickle.dump(idtest_a,open('idtest_a.pkl','w'))
pickle.dump(idtest_b,open('idtest_b.pkl','w'))

pickle.dump(embedMatrix,open('embedMatrix.pkl','w'))
pickle.dump(VOCAB_SIZE,open('vocab_size.pkl','w'))
sys.exit("Finished!")
'''

# Retrieving data from pickle 
totalOP = pickle.load(open('totalOP.pkl'))
total_a = pickle.load(open('total_a.pkl'))
total_b = pickle.load(open('total_b.pkl'))

testOP = pickle.load(open('testOP.pkl'))
idtest_a = pickle.load(open('idtest_a.pkl'))
idtest_b = pickle.load(open('idtest_b.pkl'))

embedMatrix = pickle.load(open('embedMatrix.pkl'))
VOCAB_SIZE = pickle.load(open('vocab_size.pkl'))


#Model comstruction using keras funcitonal api shared layers  

input_a = Input(shape=(THRESHOLD,))
input_b = Input(shape=(THRESHOLD,))
# This embedding layer will encode the input sequence
# into a sequence of dense THRESHOLD-dimensional vectors.
x = Embedding(output_dim=VECTOR_LEN, input_dim=VOCAB_SIZE, input_length=THRESHOLD,weights =[embedMatrix])
# Getting the embeddings of both sentences
embed_a = x(input_a)
embed_b = x(input_b)
# This layer can take as input a matrix and will return a vector of size 64
shared_lstm = LSTM(128)
# Getting the sentence encoding of both the sentences
encode_a = shared_lstm(embed_a)
encode_b = shared_lstm(embed_b)
# We can then concatenate the two vectors:
merged_vector:wq = keras.layers.concatenate([encode_a, encode_b], axis=-1)
# And add a logistic regression on top
dense1 = Dense(70, activation='relu')(merged_vector)
dense2 = Dense(40,activation='relu')(dense1)
dense3 = Dense(10,activation='relu')(dense2)
predictions = Dense(1,activation='sigmoid')(dense3)
# We define a trainable model linking the sentence inputs to the predictions
model = Model(inputs=[input_a, input_b], outputs=predictions)
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

# Concatenating all the subparts and making a final train list
finalTrain_a = total_a[0].tolist() + total_a[1].tolist() + total_a[2].tolist() + total_a[3].tolist()
finalTrain_b = total_b[0].tolist() + total_b[1].tolist() + total_b[2].tolist() + total_b[3].tolist()
finalTrain_OP = totalOP[0] + totalOP[1] + totalOP[2] + totalOP[3]

finalTrain_a = np.asarray(finalTrain_a,dtype=float)
finalTrain_b = np.asarray(finalTrain_b,dtype=float)

# early stoppign callback 
# here we are giving a validation set of size 5000 and early stopping 
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
model.fit([finalTrain_a, finalTrain_b], finalTrain_OP, epochs=10,callbacks=[es],validation_data= ([idtest_a,idtest_b],testOP))
#loss,score = model.evaluate([idtest_a,idtest_b],testOP, verbose=1)
#print 'accuracy : ',score*100

# over 82 thousand training samples are in each partition and testing is done on 5 thousand samples

# Preprcoessing medical QA data

ehealth = "/DATA1/USERS/anirudh/miniprojects/QA/medical-question-answer-data/ehealthforumQAs.json"
icliniq = "/DATA1/USERS/anirudh/miniprojects/QA/medical-question-answer-data/icliniqQAs.json"
qdoctor = "/DATA1/USERS/anirudh/miniprojects/QA/medical-question-answer-data/questionDoctorQAs.json"
webmd = "/DATA1/USERS/anirudh/miniprojects/QA/medical-question-answer-data/webmdQAs.json"
healthtap = "/DATA1/USERS/anirudh/miniprojects/QA/medical-question-answer-data/healthtapQAs.json"

qList = []
aList = []
data1 = json.load(open(ehealth))
data2 = json.load(open(icliniq))
data3 = json.load(open(qdoctor))
data4 = json.load(open(webmd))
daat5 = json.load(open(healthtap))
extract(data1)
extract(data2)
extract(data3)
extract(data4)
extract(data5)

# Assuming we got the question representation of all the questions in the medical dataset 
# Also assuming we got the user input n got its vector rep as well 
# Now find the nearest neighbour 




























