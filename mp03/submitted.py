'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

# define your epsilon for laplace smoothing here
def create_frequency_table(train):
        wordFrequency = {}
        tagcount = Counter()

        for sentence in train:   
                for word, tag in sentence:
                        tagcount.update([tag])

                        if word not in wordFrequency:
                                wordFrequency[word] = Counter()
                                wordFrequency[word].update([tag])
                        else:
                                wordFrequency[word].update([tag])

        wordFrequency['OOV'] = tagcount.most_common(1)[0][0]

        return wordFrequency


def baseline(test, train):
        '''
        Implementation for the baseline tagger.
        input:  test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
                training data (list of sentences, with tags on the words)
        output: list of sentences, each sentence is a list of (word,tag) pairs.
                E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        '''

        freq = create_frequency_table(train)

        print(freq)
    
        hypotheses = []

        for sentence in test:
                sentencelist = []
                for word in sentence:
                        currtag = None
                        
                        try:
                                currtag = freq[word].most_common(1)[0][0]
                        except:
                                currtag = freq['OOV']
                        
                        sentencelist.append((word, currtag))
                hypotheses.append(sentencelist)

        #raise NotImplementedError("You need to write this part!")
        return hypotheses 

def create_viterbi_frequency_table(train):
        tagFrequency = {}
        tagTransition = {}
        tagStart = Counter()

        for sentence in train:   
                prevTag = None

                for word, tag in sentence:
                        if tag not in tagFrequency:
                                tagFrequency[tag] = Counter()
 
                        tagFrequency[tag].update([word])

                        if (prevTag != None and prevTag != 'START'):
                                if prevTag not in tagTransition:
                                        tagTransition[prevTag] = Counter()

                                tagTransition[prevTag].update([tag])

                        if prevTag == 'START':
                                tagStart.update([tag])
                        prevTag = tag

        return tagFrequency, tagTransition, tagStart

def log_laplace_viterbi_smoothing(tagFrequency, tagTransition, tagStart, smoothness):

        lsFrequency = {}
        lsTransition = {}
        lsStart = {}

        for t in tagFrequency:
                lsFrequency[t] = {}
                wordTag = sum(tagFrequency[t].values())
                wordType = len(tagFrequency[t].keys()) + 1

                lsFrequency[t]['OOV'] = math.log(smoothness/(wordTag + smoothness * wordType))

                for w in tagFrequency[t]:
                        lsFrequency[t][w] = math.log((tagFrequency[t][w] + smoothness)/(wordTag + smoothness * wordType))

        for pt in tagTransition:
                        lsTransition[pt] = {}
                        tags = sum(tagTransition[pt].values())
                        tagType = len(tagTransition[pt].keys()) + 1
                        #lsTransition[pt]['OOV'] = math.log(smoothness/((tags + smoothness * tagType)))

                        for currt in tagTransition:
                                lsTransition[pt][currt]= math.log((tagTransition[pt][currt] + smoothness)/(tags + smoothness * tagType))

        srtTags = sum(tagStart.values())
        srtTagType = len(tagStart.keys()) + 1
        for t in tagTransition:
                lsStart[t] = math.log((tagStart[t] + smoothness) / (srtTags + smoothness * srtTagType)) 

        #lsStart['OOV'] = math.log(smoothness / (srtTags + smoothness * (len(tagStart) + 1)))

        return lsFrequency, lsTransition, lsStart

def viterbi(test, train):
        '''
        Implementation for the viterbi tagger.
        input:  test data (list of sentences, no tags on the words)
                training data (list of sentences, with tags on the words)
        output: list of sentences with tags on the words
                E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        '''

        tagFrequency, tagTransition, tagStart = create_viterbi_frequency_table(train)
        lsFrequency, lsTransition, lsStart = log_laplace_viterbi_smoothing(tagFrequency, tagTransition, tagStart, 0.000001)

        '''for i in lsFrequency:
                print(i, lsFrequency[i])'''

        #print(lsStart)
        
        #print ("Frequency: ", tagFrequency)
        #print ("Transition: ", lsTransition)
        #print ("Start Trans: ", lsStart)

        #print(lsTransition.keys())

        output = []

        for sentence in test:
                trellis = [{}]
                backpointer = [{}]

                for tag in lsStart:
                        emissProb = lsFrequency[tag].get(sentence[0],lsFrequency[tag]['OOV'])
                        initProb = lsStart[tag]

                        '''if (sentence == test[0]):
                                print(tag, ": ", initProb, " + ", emissProb, " = ", initProb + emissProb)
                                if(tag == 'NOUN'):
                                        print(sentence[0])
                                        print('NOUN:', lsFrequency[tag]['appointment'])'''

                        trellis[0][tag] = initProb + emissProb
                        backpointer[0][tag] = None

                for i, word in enumerate(sentence[1:]):
                        trellis.append({})
                        backpointer.append({})

                        for tag in lsTransition:
                                trellis[i+1][tag] = float('-inf')

                                for prevTag in lsTransition:
                                        transProb = lsTransition[prevTag][tag]
                                        emissProb = lsFrequency[tag].get(word,lsFrequency[tag]['OOV'])

                                        currProb = trellis[i][prevTag] + transProb + emissProb
                                        if (currProb > trellis[i+1][tag]):
                                                trellis[i+1][tag] = currProb
                                                backpointer[i+1][tag] = prevTag
                '''if (sentence == test[0]):
                        for i in range(len(trellis)):
                                print("Word: ", sentence[i], "\n")
                                print(trellis[i], "\n")'''
                #print(backpointer[-1])


                best_tags = []
                last_tag = max(trellis[-1], key=trellis[-1].get)
                for i in range(len(sentence) - 1, -1, -1):
                        best_tags.insert(0, last_tag)
                        #print(last_tag)
                        last_tag = backpointer[i][last_tag]
                
                output.append(list(zip(sentence, best_tags)))
        print(output[0])
        
        return output

        #raise NotImplementedError("You need to write this part!")
        return test




def viterbi_ec(test, train):
    '''
    Implementation for the improved viterbi tagger.
    input:  test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
            training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    raise NotImplementedError("You need to write this part!")



