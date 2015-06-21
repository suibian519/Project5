import math

def bigramsBackOff(unigrams, bigrams, testData):
	'''
	Compute bigram Katz Back-Off probabilities
	unigrams: dictionary {unigram: count} from training set
	bigrams: dictionary {bigram: count} from training set
	testData: tokenized emails from test/validation set
	'''

	print('\ncomputing bigram Katz back-off smoothing probabilities...')

	N = sum(unigrams.values())
	V = len(unigrams)

	# Dictionary that contains Laplace bigram probabilities
	bigramsProbDict = {}
	for bigram in bigrams:
		bigramCount = bigrams[bigram] + 1
		unigramCount = unigrams[bigram[0]]
		bigramsProbDict[bigram] = float(bigramCount) / (unigramCount + V)

	# Dictionary that contains Laplace unigram probabilities
	unigramsProbDict = {}
	for unigram in unigrams:
		unigramCount = unigrams[unigram] + 1
		unigramsProbDict[unigram] = float(unigramCount) / (N + V)
	unigramsProbDict['<UNK>'] = 1.0 / (N + V)

	# List of email probabilities to return
	bigramProb = []

	# Loop over all emails in test/validation set
	for email in testData:
		# Initialize email probability
		emailProb = 0.0

		for sentence in email:
			generated_bigrams = zip(sentence, sentence[1:])

			for bigram in generated_bigrams:
				if bigram in bigrams:
					# If the bigrams is in the list of bigrams from training set == if C(bigram) > 0
					emailProb = emailProb + math.log(bigramsProbDict[bigram])

				else:
					# Compute alpha(bigram[0])
					count1, count2 = 0.0, 0.0

					for unigram in unigrams:
						if (bigram[0], unigram) in bigrams:
							count1 = count1 + bigramsProbDict[(bigram[0], unigram)]
							count2 = count2 + unigramsProbDict[unigram]
					alpha = (1.0 - count1) / (1.0 - count2)

					# Prob = alpha(bigram[0]) * P(bigram[1])
					if bigram[1] in unigrams:
						emailProb = emailProb + math.log(alpha * unigramsProbDict[bigram[1]])

					else:
						emailProb = emailProb + math.log(alpha * unigramsProbDict['<UNK>'])

		bigramProb.append(emailProb)

	return bigramProb

def trigramsBackOff(unigrams, bigrams, trigrams, testData):
	'''
	Compute trigram Katz Back-Off probabilities
	unigrams: dictionary {unigram: count} from training set
	bigrams: dictionary {bigram: count} from training set
	trigrams: dictionary {trigram: count} from training set
	testData: tokenized emails from test/validation set
	'''

	print('\ncomputing trigram Katz back-off smoothing probabilities...')

	N = sum(unigrams.values())
	V = len(unigrams)

	# Dictionary that contains Laplace trigram probabilities
	trigramsProbDict = {}
	for trigram in trigrams:
		trigramCount = trigrams[trigram] + 1
		bigramCount = bigrams[(trigram[0], trigram[1])]
		trigramsProbDict[trigram] = float(trigramCount) / (bigramCount + V)

	# Dictionary that contains Laplace bigram probabilities
	bigramsProbDict = {}
	for bigram in bigrams:
		bigramCount = bigrams[bigram] + 1
		unigramCount = unigrams[bigram[0]]
		bigramsProbDict[bigram] = float(bigramCount) / (unigramCount + V)

	# Dictionary that contains Laplace unigram probabilities
	unigramsProbDict = {}
	for unigram in unigrams:
		unigramCount = unigrams[unigram] + 1
		unigramsProbDict[unigram] = float(unigramCount) / (N + V)
	unigramsProbDict['<UNK>'] = 1.0 / (N + V)

	# List of email probabilities to return
	trigramProb = []

	# Loop over all emails in test/validation data
	for email in testData:
		# Initialize email probability
		emailProb = 0.0

		for sentence in email:
			generated_trigrams=zip(sentence, sentence[1:], sentence[2:])

			for trigram in generated_trigrams:
				if trigram in trigrams:
					# If the current trigram is in the list of trigrams from training set == if C(trigram) > 0
					emailProb = emailProb + math.log(trigramsProbDict[trigram])

				elif (trigram[0], trigram[1]) in bigrams:
					# If C(w1, w2) > 0, then Prob = alpha(w1, w2) * Pkatz(w3 | w2)

					# Compute alpha(w1, w2)
					count1, count2 = 0.0, 0.0

					for unigram in unigrams:
						if (trigram[0], trigram[1], unigram) in trigrams:
							count1 = count1 + trigramsProbDict[(trigram[0], trigram[1], unigram)]
							count2 = count2 + bigramsProbDict[(trigram[1], unigram)]
					alpha = (1.0 - count1) / (1.0 - count2)

					if (trigram[1], trigram[2]) in bigrams:
						# If C(w2, w3) > 0
						emailProb = emailProb + math.log(alpha * bigramsProbDict[(trigram[1], trigram[2])])

					else:
						# Compute alpha(w2)
						count1, count2 = 0.0, 0.0

						for unigram in unigrams:
							if (trigram[1], unigram) in bigrams:
								count1 = count1 + bigramsProbDict[(trigram[1], unigram)]
								count2 = count2 + unigramsProbDict[unigram]
						alpha2 = (1.0 - count1) / (1.0 - count2)

						if trigram[2] in unigrams:
							emailProb = emailProb + math.log(alpha * alpha2 * unigramsProbDict[trigram[2]])

						else:
							emailProb = emailProb + math.log(alpha * alpha2 * unigramsProbDict['<UNK>'])

				else:
					# Else, Prob = P(w3)
					if trigram[2] in unigrams:
						emailProb = emailProb + math.log(unigramsProbDict[trigram[2]])

					else:
						emailProb = emailProb + math.log(1.0 / (N + V))

		trigramProb.append(emailProb)

	return trigramProb