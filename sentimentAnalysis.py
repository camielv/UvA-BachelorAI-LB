import LP as langProc
import perceptron as p
import csv

def main( iterations = 1 ):
    file1 = csv.reader(open('DataCSV.csv', 'rb'), delimiter=',', quotechar='"')     

    LP = langProc.LanguageProcessor(file1)
    p1 = p.Perceptron()
    p2 = p.Perceptron()

    # accuracy, precision, recall for both classifications
    Opinion = [0,0,0, 0, 0, 0]
    
    for i in range(iterations):
        print 'Iteration ', i, ':' 
        sentiment = LP.getSentiment()
        sentence = LP.getSentence()

        # bagofwords approach
        bow = LP.makeBagOfWords()
        
        bagOfWordsTrain = bow[0]
        bagOfWordsTest  = bow[1]

        # bagofwords[sentencenumber][token] = 1 or 0
        
        gottenInput  = dict()
        for j in bagOfWordsTrain.keys():
            gottenInput[j] = bagOfWordsTrain[j].values()
        
        wantedOutput1 = [x != 0 for x in sentiment.values()]
        wantedOutput2 = [x >  0 for x in sentiment.values()]

        trainingSet1 = dict()
        trainingSet2 = dict()
        
        k = 0
        # for all trainingset
        for j in bow[2]:
            trainingSet1[j] = (gottenInput[j], wantedOutput1[j])
            if sentiment[j]:
                k+=1
                trainingSet2[k] = (gottenInput[j], wantedOutput2[j])
        p1.train(trainingSet1)  
        p2.train(trainingSet2)

        
        confusion = {}
        confusion['PosNeg'] = {'tp':0,'tn':0,'fp':0,'fn':0}
        confusion['Opinion'] = {'tp':0,'tn':0,'fp':0,'fn':0}
        # use the bag of words for the test set to test found perceptron

        for j in bow[3]:
            # get the output from inputting a word vector of sentence j

            sentimental = p1.output( bagOfWordsTest[j].values() )
            if sentiment[j]:
                positive = p2.output( bagOfWordsTest[j].values() )

            
            if sentimental:
                if sentiment[j] == 0:
                    confusion['Opinion']['fp'] += 1
                    #print sentence[i], ' Distance = ', probSent[i], '-', sentiment[i], ' = ', probSent[i]- sentiment[i]
                else:
                    confusion['Opinion']['tp'] += 1
            else:
                if sentiment[j] == 0:
                    confusion['Opinion']['tn'] += 1
                else:
                    confusion['Opinion']['fn'] += 1
            
            # only test pos/neg for sentimental sentences
            if sentiment[j] != 0:
                if positive:
                    if sentiment[j] < 0 :
                        confusion['PosNeg']['fp'] += 1
                    else:
                        confusion['PosNeg']['tp'] += 1
                else:
                    if sentiment[j] < 0:
                        confusion['PosNeg']['tn'] += 1
                    else:
                        confusion['PosNeg']['fn'] += 1
                    
        try:
            accNeu = float(confusion['Opinion']['tp'] + confusion['Opinion']['tn']) / (confusion['Opinion']['tp'] + confusion['Opinion']['tn'] + confusion['Opinion']['fp'] + confusion['Opinion']['fn'])
        except:
            accNeu = 0
        try:
            preNeu = float(confusion['Opinion']['tp']) / (confusion['Opinion']['tp'] + confusion['Opinion']['fp'] )
        except:
            preNeu = 0
        try:
            recNeu = float(confusion['Opinion']['tp']) / (confusion['Opinion']['tp'] + confusion['Opinion']['fn'] )
        except:
            recNeu = 0
        try:   
            accPos = float(confusion['PosNeg']['tp'] + confusion['PosNeg']['tn']) / (confusion['PosNeg']['tp'] + confusion['PosNeg']['tn'] + confusion['PosNeg']['fp'] + confusion['PosNeg']['fn'])
        except:
            accPos = 0
        try:
            prePos = float(confusion['PosNeg']['tp']) / (confusion['PosNeg']['tp'] + confusion['PosNeg']['fp'] )
        except:
            prePos = 0
        try:
            recPos = float(confusion['PosNeg']['tp']) / (confusion['PosNeg']['tp'] + confusion['PosNeg']['fn'] )
        except:
            recPos = 0
        print confusion

        results = (accNeu, preNeu, recNeu, accPos, prePos, recPos)        
        
        ''' 
        # linear approach
        probs = LP.makeCorpus()
        (trainSet, testSet, probWord, probSent) = probs

 
        # create training set based on sentiments
        ssv1 = [x != 0 for x in sentiment.values()]
        ssv2 = [x >  0 for x in sentiment.values()]
        trainingSet1 = dict()
        trainingSet2 = dict()

        j = 0 
        for j in trainSet:
            trainingSet1[j] = ((probSent['Opinion'][j],), ssv1[j])
            if sentiment[j]:
                k+=1
                trainingSet2[k] = ((probSent['PosNeg'][k],), ssv2[k])
        p1.train(trainingSet1)  
        p2.train(trainingSet2)
        
        thresholds = (p1.threshold / p1.weights[0], p2.threshold / p2.weights[0])
        results = printResults(testSet, thresholds, sentiment, probSent)
        '''

        for o in range(6):
            Opinion[o] += results[o]
    for o in range(6):
        Opinion[o] /= iterations
    print Opinion
        
def printResults(testSet, thresholds, sentiment, probSent):
        # dictionary containing number of true postives etc. for classifiers Positive and Neutral 
        confusion = {}
        confusion['PosNeg'] = {'tp':0,'tn':0,'fp':0,'fn':0}
        confusion['Opinion'] = {'tp':0,'tn':0,'fp':0,'fn':0}
        
        for i in testSet:
                if probSent['Opinion'][i] >= thresholds[0]:
                    if sentiment[i] == 0:
                        confusion['Opinion']['fp'] += 1
                        #print sentence[i], ' Distance = ', probSent[i], '-', sentiment[i], ' = ', probSent[i]- sentiment[i]
                    else:
                        confusion['Opinion']['tp'] += 1
                elif probSent['Opinion'][i] < thresholds[0]:
                    if sentiment[i] == 0:
                        confusion['Opinion']['tn'] += 1
                    else:
                        confusion['Opinion']['fn'] += 1
                
                # only test pos/neg for sentimental sentences
                if sentiment[i] != 0:
                    if probSent['PosNeg'][i] >= thresholds[1]:
                        if sentiment[i] < 0 :
                            confusion['PosNeg']['fp'] += 1
                        else:
                            confusion['PosNeg']['tp'] += 1
                    elif probSent['PosNeg'][i] < thresholds[1]:
                        if sentiment[i] < 0:
                            confusion['PosNeg']['tn'] += 1
                        else:
                            confusion['PosNeg']['fn'] += 1
                        
        accNeu = float(confusion['Opinion']['tp'] + confusion['Opinion']['tn']) / (confusion['Opinion']['tp'] + confusion['Opinion']['tn'] + confusion['Opinion']['fp'] + confusion['Opinion']['fn'])
        try:
            preNeu = float(confusion['Opinion']['tp']) / (confusion['Opinion']['tp'] + confusion['Opinion']['fp'] )
        except:
            preNeu = 0
        recNeu = float(confusion['Opinion']['tp']) / (confusion['Opinion']['tp'] + confusion['Opinion']['fn'] )

        accPos = float(confusion['PosNeg']['tp'] + confusion['PosNeg']['tn']) / (confusion['PosNeg']['tp'] + confusion['PosNeg']['tn'] + confusion['PosNeg']['fp'] + confusion['PosNeg']['fn'])
        try:
            prePos = float(confusion['PosNeg']['tp']) / (confusion['PosNeg']['tp'] + confusion['PosNeg']['fp'] )
        except:
            prePos = 0
        recPos = float(confusion['PosNeg']['tp']) / (confusion['PosNeg']['tp'] + confusion['PosNeg']['fn'] )

        print confusion

        return (accNeu, preNeu, recNeu, accPos, prePos, recPos)        


main()
