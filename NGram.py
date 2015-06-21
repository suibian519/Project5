import operator

# part 2.2
#
# we create unsmoothed n-gram models first by determining
# frequencies of each n-gram
# then using these frequencies we compute probabilities
# for each n-gram
# returns: dictionaries of key:n-gram value:counts

def countNGram(ngrams,ngramsDict):

	for item in ngrams:

		if item in ngramsDict:

			ngramsDict[item]+=1

		else:

			ngramsDict[item]=1

	return ngramsDict


def getNGram(data):

	# dictionaries to store ngrams with counts
	unigrams={}
	bigrams={}
	trigrams={}

 
	for emailText in data:

		for sentence in emailText:
			#Unigram
			unigrams=countNGram(sentence,unigrams)

			#Bigram
			generated_bigrams=zip(sentence, sentence[1:])
			bigrams=countNGram(generated_bigrams,bigrams)

			#Trigram
			generated_trigrams=zip(sentence, sentence[1:], sentence[2:])
			trigrams=countNGram(generated_trigrams,trigrams)

	# create lists of n-grams in order of decreasing frequency
	sortedTrigram = sorted(trigrams.items(), key=operator.itemgetter(1),reverse=True)

	sortedBigram = sorted(bigrams.items(), key=operator.itemgetter(1),reverse=True)
	for bgram in sortedBigram:
		# find the bigram in sortedBigram for <s><s> and remove it
		# for later processing
		if bgram[0][0] == '<s>' and bgram[0][1] == '<s>':
			sortedBigram.remove(bgram)
			break
	
	sortedUnigram = sorted(unigrams.items(), key=operator.itemgetter(1),reverse=True)
	for ugram in sortedUnigram:
		# find the unigram in sortedUnigram for <s> and remove it
		# for later processing
		if ugram[0] == '<s>':
			sortedUnigram.remove(ugram)
			break

	unigramProbs = getUnigramProbs(sortedUnigram) # compute probs
	print('---===<Top 10 unigrams by frequency>===---')
	for j in range(0,10):
		#print(unigramProbs[j]) #top 10 by probability
		print(sortedUnigram[j]) #top 10 by frequency
	
	
	bigramProbs = getBigramProbs(sortedBigram,unigrams) # compute probs
	print('---===<Top 10 bigrams by frequency>===---')
	for k in range(0,10):
		#print(bigramProbs[k]) #top 10 by probability
		print(sortedBigram[k]) #top 10 by frequency
	
	
	trigramProbs = getTrigramProbs(sortedTrigram,bigrams) # compute probs
	print('---===<Top 10 trigrams by frequency>===---')
	for l in range(0,10):
		#print(trigramProbs[l]) #top 10 by probability
		print(sortedTrigram[l]) #top 10 by frequency

	return unigrams,bigrams,trigrams # dictionaries containing all tokens/freq. counts
	
def getUnigramProbs(sortedUnigrams):
	print('\ncomputing unigram unsmoothed probabilities...')
	
	unigramProbs = []
	
	# first get total token count in corpus
	totalNumTokens = 0.0
	for ngram in sortedUnigrams: # sortedUnigrams does not contain <s>
		totalNumTokens += ngram[1]

	for curUnigram in sortedUnigrams:
		unigramProbs.append([curUnigram[0],(curUnigram[1]/totalNumTokens)]) # update list with [token,prob] tuple
	
	return unigramProbs
	
def getBigramProbs(sortedBigrams,unigrams):
	print('\ncomputing bigram unsmoothed probabilities...')
	
	bigramProbsDict = {}
	bigramProbs = []
	
	# construct dict of bigram probabilities
	# sortedBigrams does not contain <s><s>
	for curBigram in sortedBigrams:
		# compute probability for each bigram using
		# (C(bigram)/C(first token in bigram))
		bigramProbsDict[curBigram[0][0],curBigram[0][1]] = (float(curBigram[1])/unigrams[curBigram[0][0]])

	# sort by probability
	bigramProbs = sorted(bigramProbsDict.items(), key=operator.itemgetter(1),reverse=True)	
	
	return bigramProbs
	
def getTrigramProbs(sortedTrigrams,bigrams):
	print('\ncomputing trigram unsmoothed probabilities...')
	
	trigramProbsDict = {}
	trigramProbs = []
	
	# construct dict of trigram probabilities
	for curTrigram in sortedTrigrams:
		# compute probability for each trigram using
		# (C(trigram)/C(first two tokens in trigram))
		trigramProbsDict[curTrigram[0][0],curTrigram[0][1],curTrigram[0][2]] = (float(curTrigram[1])/bigrams[curTrigram[0][0],curTrigram[0][1]])
	
	# sort by probability
	trigramProbs = sorted(trigramProbsDict.items(), key=operator.itemgetter(1),reverse=True)
	
	return trigramProbs
