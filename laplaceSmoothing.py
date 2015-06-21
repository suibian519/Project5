import operator
import random
import bisect
import math

# part 2.4
#
# Laplace smoothing and unseen word handling
#
# Input:
# Dictionary of the 6 model
# Test data set
# Delta coefficient, set to 1 in case of Laplace smoothing
#
# Output:
# A list of smoothed log probability of each email in the set
# A dictionary of word and smoothed log probability pair

def uniLaSmooth(unigrams,testData,delta):

	print('\ncomputing unigram laplace smoothing probabilities...')

	# Handling <s>. Count the number of <s> in the unigram model and remove the <s> in the unigram dictionary
	ssCount=unigrams["<s>"]
	unigrams.pop("<s>", None)

	# Compute the number of tokens(N), and number of word types in vocabulary
	N = sum(unigrams.values())
	V = len(unigrams)

	# A list to store the smoothed probabilities of each email
	emailProbs=[]
	# A dictionary to store the unigram and smoothed probability pair
	unigramDic = {}

	# Loop through each word in each sentence of emails in the test data
	for email in testData:
		# Initialized the email probability
		emailProb=0.0

		for sentence in email:

			for word in sentence:
				# Skip <s>
				if word != '<s>':

					if word in unigrams:
						# Compute the numerator of Laplace formula
						unigramCount = unigrams[word] + delta

					else:
						# If meet unseen words 
						unigramCount = delta

					# Compute the smoothed log probability
					uniProb = math.log(float(unigramCount)/(N + V*delta))
					# Store the (word, smoothed probability) pair into the dictionary
					unigramDic[word] = uniProb
					# Sum up to get the smoothed log probability of each email
					emailProb=emailProb+math.log(float(unigramCount)/(N + V*delta))

		# Append email probablity to the list
		emailProbs.append(emailProb)

	# Add the <s> back to the unigram model
	unigrams["<s>"]=ssCount

	return emailProbs,unigramDic

def biLaSmooth(unigrams, bigrams, testData,delta):

	print('\ncomputing bigram laplace smoothing probabilities...')

	# Handling <s>. Count the number of "<s>, <s>" in the bigram model and remove the "<s>, <s>" in the bigram dictionary
	ssCount=bigrams[("<s>", "<s>")]
	bigrams.pop(("<s>", "<s>"), None)

	# Compute the number of word types in vocabulary
	V = len(unigrams)

	# A list to store the smoothed probabilities of each email
	emailProbs=[]
	# A dictionary to store the bigram and smoothed probability pair
	bigramDic={}


	# Loop through each word in each sentence of emails in the test data
	for email in testData:
		# Initialized the email probability
		emailProb=0.0

		for sentence in email:
			# Get the bigrams in each sentence
			generated_bigrams = zip(sentence, sentence[1:])

			for bigram in generated_bigrams:

				if (bigram[0][0]=='<s>') and (bigram[0][1]=='<s>'):
					continue
				else:

					if bigram in bigrams:
						# Get the word count adding one
						bigramCount = bigrams[bigram] + delta

					else:
						# If meet unseen words
						bigramCount = delta

					if bigram[0] in unigrams.keys():
						# Compute the count of w(n-1)
						unigramCount = unigrams[bigram[0]]

					else:
						# if meet unseen words
						unigramCount = 0

					# Compute the smoothed log probability
					biProb = math.log(float(bigramCount)/(unigramCount + V*delta))
					# Store the (word, smoothed probability) pair into the dictionary
					bigramDic[bigram] = biProb

					# Sum up to get the smoothed log probability of each email
					emailProb = emailProb + math.log(float(bigramCount)/(unigramCount + V*delta))

		# Append email probablity to the list
		emailProbs.append(emailProb)

	# Add the <s><s> back to the bigram model
	bigrams[("<s>", "<s>")]=ssCount

	return emailProbs,bigramDic

def triLaSmooth(unigrams, bigrams, trigrams, testData,delta):

	print('\ncomputing trigram laplace smoothing probabilities...')

	V = len(unigrams)

	emailProbs = []
	trigramDic = {}

	# Loop through each word in each sentence of emails in the test data
	for email in testData:
		# Initialized the email probability
		emailProb=0.0

		for sentence in email:
			# Get the trigrams in each sentence
			generated_trigrams=zip(sentence, sentence[1:], sentence[2:])

			for trigram in generated_trigrams:

				if trigram in trigrams:
					# Get the word count adding one
					trigramCount = trigrams[trigram] + delta

				else:
					# if meet unseen words
					trigramCount = delta

				if trigram[0:2] in bigrams.keys():
					# Compute the count of w(n-1)
					bigramCount = bigrams[trigram[0:2]]

				else:
					# if meet unseen words
					bigramCount = 0

				# Compute the smoothed log probability
				triProb = math.log(float(trigramCount)/(bigramCount+ V*delta))
				# Store the (word, smoothed probability) pair into the dictionary
				trigramDic[trigram] = triProb

				# Sum up to get the smoothed log probability of each email
				emailProb = emailProb + math.log(float(trigramCount)/(bigramCount+ V*delta))

		# Append email probablity to the list
		emailProbs.append(emailProb)

	return emailProbs,trigramDic

