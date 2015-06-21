import math

# for part 2.5:
# we compute perplexity by using the tokenized validation set
# and the respective laplace-smoothed model (unigram/bigram/trigram)
#
# we rewrote the given perplexity formula to work with our data using
# the following mathematical equalities, and the fact that our smoothed
# models already contain log probabilities:
# starting with the PP equation given in the assignment
# log PP = log[product(i=0,N):smProb]^(1/N)
# log PP = (-1/N)sum(i=0,N):log(smProb) --> log(smProb) is in our models
# PP = exp((-1/N)sum(i=0,N):log(smProb)

def computeUniPP(val_set,smoothedUniModel):
	
	ppProduct = 0.0 # the large product (i=0 to N) in the general formula
	# ppProduct starts at zero since we are summing
	tokenCount = 0.0 # this is N in the general formula
	for emailText in val_set:
		for sentence in emailText:
			for token in sentence:
				if(token=='<s>'):
					# don't count <s> as instructed on assignment
					continue

				# we are summing as explained above; smoothed model
				# contains log probability values
				ppProduct = ppProduct + float(smoothedUniModel[token])
				tokenCount += 1 # increment token count for everything except <s>

	return(math.exp((-1.0/tokenCount)*ppProduct))

def computeBiPP(val_set,smoothedBiModel):

	ppProduct = 0.0 # the large product (i=0 to N) in the general formula
	# ppProduct starts at zero since we are summing
	tokenCount = 0.0 # this is N in the general formula
	for emailText in val_set:
		for sentence in emailText:
			# iterating over sentences to avoid the </s><s> bigram
			for token in range(len(sentence)):
				# token represents an integer index into the sentence
				if(token==0 or token==1):
					#skip first two <s> tokens and don't count them
					continue

				# we are summing as explained above; smoothed model
				# contains log probability values.
				# find the key in the smoothed bigram model for
				# the previous token/current token bigram
				ppProduct = ppProduct + float(smoothedBiModel[sentence[token-1],sentence[token]])

				tokenCount += 1 # increment token count for everything except <s>

	return(math.exp((-1.0/tokenCount)*ppProduct))

def computeTriPP(val_set,smoothedTriModel):
	
	ppProduct = 0.0 # the large product (i=0 to N) in the general formula
	# ppProduct starts at zero since we are summing
	tokenCount = 0.0 # this is N in the general formula
	for emailText in val_set:
		for sentence in emailText:
			# iterating over sentences to avoid the </s><s> bigram
			for token in range(len(sentence)):
				# token represents an integer index into the sentence
				if(token==0 or token==1):
					#skip first two <s> tokens and don't count them
					continue

				# we are summing as explained above; smoothed model
				# contains log probability values.
				# find the key in the smoothed trigram model for
				# the previous two tokens&current token trigram
				ppProduct = ppProduct + float(smoothedTriModel[sentence[token-2],sentence[token-1],sentence[token]])

				tokenCount += 1

	return(math.exp((-1.0/tokenCount)*ppProduct))