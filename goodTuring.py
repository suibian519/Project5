import operator
import random
import bisect
import math

def uniGT(unigrams,testData):

	print('\ncomputing unigram GT smoothing probabilities...')

	N = sum(unigrams.values())

	frequencies = {}
	frequencies[0] = 1

	for count in list(unigrams.values()):
		if count in frequencies:
			frequencies[count] += 1
		else:
			frequencies[count] = 1

	unigramProb = []

	for email in testData:

		probability = 0.0

		for sentence in email:

			for word in sentence:

				if word in unigrams:

					unigramCount = unigrams[word] + 1

					if unigrams[word] in frequencies:
						Na = frequencies[unigrams[word]]
					else:
						Na = frequencies[0]

					if unigrams[word] + 1 in frequencies:
						Na2 = frequencies[unigrams[word] + 1]
					else:
						Na2 = frequencies[0]

				else:

					unigramCount = 1
					Na = frequencies[0]
					Na2 = frequencies[1]

				probability = probability + math.log((float(unigramCount) * Na)/(N * Na2))

		unigramProb.append(probability)

	return unigramProb

def biGT(bigrams, testData):

	print('\ncomputing bigram GT smoothing probabilities...')

	N = sum(bigrams.values())

	frequencies = {}
	frequencies[0] = 1

	for count in list(bigrams.values()):
		if count in frequencies:
			frequencies[count] += 1
		else:
			frequencies[count] = 1

	bigramProb = []

	for email in testData:

		probability = 0.0

		for sentence in email:

			generated_bigrams = zip(sentence, sentence[1:])

			for bigram in generated_bigrams:

				if bigram in bigrams:

					bigramCount = bigrams[bigram] + 1

					if bigrams[bigram] in frequencies:
						Na = frequencies[bigrams[bigram]]
					else:
						Na = frequencies[0]

					if bigrams[bigram] + 1 in frequencies:
						Na2 = frequencies[bigrams[bigram] + 1]
					else:
						Na2 = frequencies[0]

				else:

					bigramCount = 1
					Na = frequencies[0]
					Na2 = frequencies[1]

				probability = probability + math.log((float(bigramCount) * Na)/(N * Na2))

		bigramProb.append(probability)

	return bigramProb

def triGT(trigrams, testData):

	print('\ncomputing trigram GT smoothing probabilities...')

	N = sum(trigrams.values())

	frequencies = {}
	frequencies[0] = 1

	for count in list(trigrams.values()):
		if count in frequencies:
			frequencies[count] += 1
		else:
			frequencies[count] = 1

	trigramProb = []

	for email in testData:

		probability = 0.0

		for sentence in email:

			generated_trigrams = zip(sentence, sentence[1:], sentence[2:])

			for trigram in generated_trigrams:

				if trigram in trigrams:

					trigramCount = trigrams[trigram] + 1

					if trigrams[trigram] in frequencies:
						Na = frequencies[trigrams[trigram]]
					else:
						Na = frequencies[0]

					if trigrams[trigram] + 1 in frequencies:
						Na2 = frequencies[trigrams[trigram] + 1]
					else:
						Na2 = frequencies[0]

				else:

					trigramCount = 1
					Na = frequencies[0]
					Na2 = frequencies[1]

			probability = probability + math.log((float(trigramCount) * Na)/(N * Na2))

		trigramProb.append(probability)

	return trigramProb