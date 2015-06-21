import random
import bisect

def randomSentence(n, nGrams):
	'''
	Generate 5 random sentences for the selected model
	n: order of the ngrams
	nGrams: list of ngrams and their count
	'''

	# Unigrams
	if n == 1:
		print ('\n===== Generate random sentence with unigrams... =====\n')

		# Print 5 sentences
		for i in range(0, 5):

			# Initialize parameters
			sentence = '<s> '
			word = ''
			x = 0.0

			# Remove '<s>' token from the list of unigrams
			nGrams = [val for val in nGrams if val[0] != '<s>']

			# Get the unigrams and their weights
			unigrams, weights = zip(*nGrams)

			# Cumulative weights
			cumWeights = accumulate(weights)

			# Continue until we get to the end of the sentence
			while word != '</s>':
				# Pick a random number in range(0, cumWeights[-1]), then find the corresponding index in cumWeights
				# and take the word corresponding to that index. Therefore it picks the word according to the probability distribution
				x = random.random() * cumWeights[-1]
				word = unigrams[bisect.bisect(cumWeights, x)]
				sentence += word + ' '

			print(sentence)

	# Bigrams
	elif n == 2:
		print ('\n===== Generate random sentence with bigrams... =====\n')

		# Print 5 sentences
		for i in range(0, 5):

			# Initialize parameters
			sentence = '<s> '
			word = ''
			x = 0.0

			# Remove ('<s>','<s>') token from the list of bigrams
			nGrams = [val for val in nGrams if val[0] != ('<s>','<s>')]

			# Current word, initialized to a start of sentence
			current = '<s>'
			
			# Continue until we get to the end of the sentence
			while word != '</s>':
				# Keep only bigrams starting with current
				currentBigrams = [val for val in nGrams if val[0][0] == current]
				bigrams, weights = zip(*currentBigrams)
				cumWeights = accumulate(weights)
				x = random.random() * cumWeights[-1]
				word = bigrams[bisect.bisect(cumWeights, x)][1]
				sentence += word + ' '
				current = word

			print(sentence)

	# Trigrams
	elif n == 3:
		print ('\n===== Generate random sentence with trigrams... =====\n')

		# Print 5 sentences
		for i in range(0, 5):

			# Initialize parameters
			sentence = '<s> '
			word = ''
			x = 0.0

			# Current words, initialized so a start of sentence
			current = ['<s>', '<s>']
			
			# Continue until we get to the end of the sentence
			while word != '</s>':
				# Keep only bigrams starting with current
				currentTigrams = [val for val in nGrams if (val[0][0] == current[0] and val[0][1] == current[1])]
				trigrams, weights = zip(*currentTigrams)
				cumWeights = accumulate(weights)
				x = random.random() * cumWeights[-1]
				word = trigrams[bisect.bisect(cumWeights, x)][2]
				sentence += word + ' '
				current[0] = current[1]
				current[1] = word

			print(sentence)

def accumulate(init_list):
	'''
	Return a new list where new_list[i] = sum(k=0..i) init_list[k]
	'''

	total = 0
	new_list = []

	for item in init_list:
		total += item
		new_list.append(total)

	return new_list