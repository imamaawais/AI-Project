#libraries and utilities
#load data
import json
#remove accents
import unicodedata
#vectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#data splitting
from sklearn.model_selection import train_test_split
#classifiers
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier


#-----------------------------------------------------------------------------------------------------#
#                                          Preprocessing                                              #
#-----------------------------------------------------------------------------------------------------#

#load the training data file
def loadTrainingData():

    with open('train.json', 'r') as file:
        trainingData = json.load(file)

    return trainingData

#remove any accents/ fonts
def removeAccents(trainingString):
    normalizedData = unicodedata.normalize('NFKD', trainingString)
    #filter and concatenate each char and return the string
    return ''.join([char for char in normalizedData if not unicodedata.combining(char)])
    

def normalizeData(trainingData):

    ingredients = []
    cuisines = []

    #for each recipe/ data point thats in the trainingData
    for recipe in trainingData:
        #remove the accented characters from each ingredient
        ingr = [removeAccents(ingredient.lower()) for ingredient in recipe['ingredients']]
        #extract the cuisine for this recipe
        cuisine = recipe['cuisine']

        #append the ingredient in the ingredients list
        ingredients.append(' '.join(ingr))
        #append the cuisine in the cuisine list
        cuisines.append(cuisine)

    return ingredients, cuisines

#load the training data then extract ingredients and cuisines
def preprocessData():
    #load the training data
    trainingData = loadTrainingData()
    #normalize and extract the data
    ingredients, cuisines = normalizeData(trainingData)

    return ingredients, cuisines

#-----------------------------------------------------------------------------------------------------#
#                                          Model Training                                             #
#-----------------------------------------------------------------------------------------------------#

#split the data into 35k points for training and remaining 4774 for validation
def splitData(ingredients, cuisines): 

    #split the ingredients and cuisine into training and validation sets
    ingrTrain, ingrValid, cuiTrain, cuiValid = train_test_split(ingredients, cuisines, test_size=4774, random_state=42)

    return ingrTrain, ingrValid, cuiTrain, cuiValid

#form vectors from the ingredients
def vectorizer(ingrTrain, ingrValid):
    
    #use bag-of-words model to vectorize 
    vectorizer = CountVectorizer()
    #apply the tf-idf weighting 
    tfIdfTransformer =TfidfTransformer()

    #make ingredient vectors from training set
    trainVectors = vectorizer.fit_transform(ingrTrain)
    transTrainVectors = tfIdfTransformer.fit_transform(trainVectors)

    #make ingredient vectors from validation set using transform not fit-transform to ensure transformation in same way as training set
    validVectors = vectorizer.transform(ingrValid)
    transValidVectors = tfIdfTransformer.transform(validVectors)

    return transTrainVectors, transValidVectors, vectorizer, tfIdfTransformer

#Naive Bayes classifier
def NBClassifier(ingrTrain, ingrValid, cuiTrain, cuiValid):

    #initialize Naive Bayes classifier
    clf = MultinomialNB().fit(ingrTrain, cuiTrain)
    
    empericalAcc = clf.score(ingrTrain, cuiTrain)
    genAcc = clf.score(ingrValid, cuiValid)

    #print("Naive Bayes\nEA: ",round(empericalAcc, 2),",GA:", round(genAcc, 2))

    return clf

#SGD classifier
def SGDclassifier(ingrTrain, ingrValid, cuiTrain, cuiValid):
    # initialize the  SDG classifier
    clf = SGDClassifier(loss='modified_huber', penalty='l2',alpha=0.0008, random_state=42).fit(ingrTrain, cuiTrain)
    
    empericalAcc = clf.score(ingrTrain, cuiTrain)
    genAcc = clf.score(ingrValid, cuiValid)
    
    #print("SGD\nEA: ",round(empericalAcc, 2),",GA:", round(genAcc, 2))
    
    return clf

#random forest classifier
def RFClassifier(ingrTrain, ingrValid, cuiTrain, cuiValid):
    #initialise random forest classifier
    clf = RandomForestClassifier(max_depth=20,n_estimators=40,random_state=42).fit(ingrTrain,cuiTrain)
    
    empericalAcc = clf.score(ingrTrain, cuiTrain)
    genAcc = clf.score(ingrValid, cuiValid)
    
    #print("RF\nEA: ",round(empericalAcc, 2),",GA:", round(genAcc, 2))
    
    return clf

def votingClassifier(ingrTrain, ingrValid, cuiTrain, cuiValid, SGDclf, RFclf):
    #define estimators
    estimators = [('SGD', SGDclf), ('RF',RFclf)]
    #initialize and fit voting classifier
    clf = VotingClassifier(estimators=estimators, voting='soft', weights=[3,1]).fit(ingrTrain,cuiTrain)

    empericalAcc = clf.score(ingrTrain, cuiTrain)
    genAcc = clf.score(ingrValid, cuiValid)

    #print("Voting\nEA: ",round(empericalAcc, 2),",GA:", round(genAcc, 2))
    
    return clf

def KNNclassifier(ingrTrain, ingrValid, cuiTrain, cuiValid):
    #initialize and fit the classifier
    clf = KNeighborsClassifier(n_neighbors=5, weights='uniform').fit(ingrTrain,cuiTrain)

    empericalAcc = clf.score(ingrTrain, cuiTrain)
    genAcc = clf.score(ingrValid, cuiValid)

    #print("KNN\nEA: ",round(empericalAcc, 2),",GA:", round(genAcc, 2))
    
    return clf

def trainModel(transTrainVectors, transValidVectors, cuiTrain, cuiValid):
    #NB classifier
    NBclf = NBClassifier(transTrainVectors, transValidVectors, cuiTrain, cuiValid)
    #SGD classifier
    SGDclf = SGDclassifier(transTrainVectors, transValidVectors, cuiTrain, cuiValid)
    #Random Forest Classifier
    RFclf = RFClassifier(transTrainVectors, transValidVectors, cuiTrain, cuiValid)
    #Voting Classifier
    Votingclf = votingClassifier(transTrainVectors, transValidVectors, cuiTrain, cuiValid, SGDclf, RFclf)
    #K nearesr neighbors classifier
    KNNclf = KNNclassifier(transTrainVectors, transValidVectors, cuiTrain, cuiValid)
    
    return NBclf, SGDclf, RFclf, Votingclf, KNNclf
    #return SGDclf

#-----------------------------------------------------------------------------------------------------#
#                                           Testing                                                   #
#-----------------------------------------------------------------------------------------------------#

def loadTestingData():
    with open('test.json', 'r') as file:
        testingData = json.load(file)
    
    return testingData

def loadlabeledData():
    with open('recipe.json', 'r') as file:
        labeledData = json.load(file)

    return labeledData


def processTestData(vectorizer, transformer):
    testingData = loadTestingData()

    ingredients = []
    #for each recipe/ data point thats in the trainingData
    for recipe in testingData:
        #remove the accented characters from each ingredient
        ingr = [removeAccents(ingredient.lower()) for ingredient in recipe['ingredients']]
        #append the ingredient in the ingredients list
        ingredients.append(' '.join(ingr))

    #needs vectorizer that was used on the training set
    ingrVectors = vectorizer.transform(ingredients)
    ingrVectors = transformer.transform(ingrVectors)

    return testingData, ingrVectors

def processLabeledData(vectorizer, transformer):
    labeledData = loadlabeledData()

    #normalize and extract the data
    ingredients, cuisines = normalizeData(labeledData)
    ingrVectors = vectorizer.transform(ingredients)
    ingrVectors = transformer.transform(ingrVectors)

    return labeledData, ingrVectors, cuisines
    

def predictCuisine(clf, testingData, ingrVectors, cuiLabels):

    cuiPredict = clf.predict(ingrVectors)
    print("classification report:\n",classification_report(cuiLabels, cuiPredict))

    for i in range(len(testingData)):
        print(f"Recipe ID {testingData[i]['id']}: {cuiPredict[i]}")


#-----------------------------------------------------------------------------------------------------#
#                                          Driver code                                                #
#-----------------------------------------------------------------------------------------------------#


if __name__ == '__main__':
    
    #call the preprocessor
    ingredients, cuisines = preprocessData()

    #split the data into training and validation
    ingrTrain, ingrValid, cuiTrain, cuiValid = splitData(ingredients, cuisines)

    #make ingredients vectors
    transTrainVectors, transValidVectors, vectorizer, transformer = vectorizer(ingrTrain, ingrValid)

    
    #train the model
    #NBclf, SGDclf, RFclf, Votingclf, KNNclf = trainModel(transTrainVectors, transValidVectors, cuiTrain, cuiValid)
    SGDclf = trainModel(transTrainVectors, transValidVectors, cuiTrain, cuiValid)

    #process the test data
    testingData, ingrVectors = processTestData(vectorizer, transformer)

    #processed labeled testdata
    labeledData, labeledVectors, cuiLabels = processLabeledData(vectorizer, transformer)

    #predict the cuisine
    
    '''print("NB: \n")
    predictCuisine(NBclf, testingData, ingrVectors, cuiLabels)
   
    print("RF: \n")
    predictCuisine(RFclf, testingData, ingrVectors, cuiLabels)
    print("Voting: \n")
    predictCuisine(Votingclf, testingData, ingrVectors, cuiLabels)
    print("KNN: \n")
    predictCuisine(KNNclf, testingData, ingrVectors, cuiLabels)'''
    
    print("SGD: \n")
    predictCuisine(SGDclf, testingData, ingrVectors, cuiLabels)

    #print(set(cuisines))

    