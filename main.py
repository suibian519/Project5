import PreProcessing
import NGram
import laplaceSmoothing
import randomGeneration
import perplexity
import validate as v
import goodTuring
import backOff

if __name__ == "__main__":


	'''
	Preprocessing Step:
	 -Read files, spliting into sentences, tokenized
	 -Using array instead of objects to speed up the data I/O
	 -Return following:
		UP_/DOWN_ : UPSPEAK or DOWNSPEAK
		_RAW: Unmodified email text
		_SENTENCES: After spliting email text into sentences
		_TOKENIZED: After tokenizing each sentences
		Train:
			UP_TRAIN_RAW
			DOWN_TRAIN_RAW
			UP_TRAIN_SENTENCES
			DOWN_TRAIN_SENTENCES
			UP_TRAIN_TOKENIZED
			DOWN_TRAIN_TOKENIZED
		Validation:
			UP_VALIDATE_RAW
			DOWN_VALIDATE_RAW
			UP_VALIDATE_SENTENCES
			DOWN_VALIDATE_SENTENCES
			UP_VALIDATE_TOKENIZED
			DOWN_VALIDATE_TOKENIZED
		Test:
			Email number is array index + 1
			TEST_RAW
			TEST_SENTENCES
			TEST_TOKENIZED
	 '''

	UP_TRAIN_RAW,DOWN_TRAIN_RAW,UP_TRAIN_SENTENCES,DOWN_TRAIN_SENTENCES,UP_TRAIN_TOKENIZED,DOWN_TRAIN_TOKENIZED,UP_VALIDATE_RAW,DOWN_VALIDATE_RAW,UP_VALIDATE_SENTENCES,DOWN_VALIDATE_SENTENCES,UP_VALIDATE_TOKENIZED,DOWN_VALIDATE_TOKENIZED,TEST_RAW,TEST_SENTENCES,TEST_TOKENIZED=PreProcessing.process()

	UP_TRAIN_TOTAL_TOKENIZED = UP_TRAIN_TOKENIZED + UP_VALIDATE_TOKENIZED
	DOWN_TRAIN_TOTAL_TOKENIZED = DOWN_TRAIN_TOKENIZED + DOWN_VALIDATE_TOKENIZED

	VALIDATE_LABELS_UP=[1] * len(UP_VALIDATE_RAW)
	VALIDATE_LABELS_DOWN=[0] * len(DOWN_VALIDATE_RAW)


	print('\n\n========== Unsmoothed N-Grams==========\n\n')

	print('-----===== UP_TRAIN =====-----\n\n')
	upTrainUnigram,upTrainBigram,upTrainTrigram = NGram.getNGram(UP_TRAIN_TOKENIZED)
	print('\n\n-----===== DOWN_TRAIN =====-----\n\n')
	downTrainUnigram,downTrainBigram,downTrainTrigram = NGram.getNGram(DOWN_TRAIN_TOKENIZED)


	print('\n\n==========  Random Sentence Generation ==========\n\n')

	print('---=== UPSPEAK Random Sentence ===---\n\n')
	print(randomGeneration.randomSentence(1, upTrainUnigram.items()))
	print(randomGeneration.randomSentence(2, upTrainBigram.items()))
	print(randomGeneration.randomSentence(3, upTrainTrigram.items()))

	print('\n\n---=== DOWNSPEAK Random Sentence ===---')
	print(randomGeneration.randomSentence(1, downTrainUnigram.items()))
	print(randomGeneration.randomSentence(2, downTrainBigram.items()))
	print(randomGeneration.randomSentence(3, downTrainTrigram.items()))


	print('\n\n==========  Laplace Smoothing ==========\n\n')

	#Run Laplace smoothing on training/validation datasets


	UpUniTrVal, UpUniDict = laplaceSmoothing.uniLaSmooth(upTrainUnigram, UP_VALIDATE_TOKENIZED, 1)
	DownUniTrVal, DownUniDict = laplaceSmoothing.uniLaSmooth(downTrainUnigram, DOWN_VALIDATE_TOKENIZED, 1)


	UpBiTrVal, UpBiDict = laplaceSmoothing.biLaSmooth(upTrainUnigram, upTrainBigram, UP_VALIDATE_TOKENIZED, 1)
	DownBiTrVal, DownBiDict = laplaceSmoothing.biLaSmooth(downTrainUnigram, downTrainBigram, DOWN_VALIDATE_TOKENIZED, 1)


	UpTriTrVal, UpTriDict = laplaceSmoothing.triLaSmooth(upTrainUnigram, upTrainBigram, upTrainTrigram, UP_VALIDATE_TOKENIZED, 1)
	DownTriTrVal, DownTriDict = laplaceSmoothing.triLaSmooth(downTrainUnigram, downTrainBigram, downTrainTrigram, DOWN_VALIDATE_TOKENIZED, 1)



	print('\n\n===============  Perplexities ===============\n\n')

	PPUpValUni = perplexity.computeUniPP(UP_VALIDATE_TOKENIZED, UpUniDict)
	PPDownValUni = perplexity.computeUniPP(DOWN_VALIDATE_TOKENIZED, DownUniDict)

	print('Unigram Perplexities:\n PP(UP_Validation) = ' + str(PPUpValUni) + '\n PP(Down_Validation) = ' + str(PPDownValUni) + '\n')


	PPUpValBi = perplexity.computeBiPP(UP_VALIDATE_TOKENIZED, UpBiDict)
	PPDownValBi = perplexity.computeBiPP(DOWN_VALIDATE_TOKENIZED, DownBiDict)

	print('Bigram Perplexities:\n PP(UP_Validation) = ' + str(PPUpValBi) + '\n PP(Down_Validation) = ' + str(PPDownValBi) + '\n')

	
	PPUpValTri = perplexity.computeTriPP(UP_VALIDATE_TOKENIZED, UpTriDict)
	PPDownValTri = perplexity.computeTriPP(DOWN_VALIDATE_TOKENIZED, DownTriDict)

	print('Trigram Perplexities:\n PP(UP_Validation) = ' + str(PPUpValTri) + '\n PP(Down_Validation) = ' + str(PPDownValTri) + '\n')


	print('\n\n===============  Prediction ===============')


	v.validatePrediction(2, 'laplaceSmoothing', upTrainUnigram, downTrainUnigram, upTrainBigram, downTrainBigram, upTrainTrigram, downTrainTrigram, UP_VALIDATE_TOKENIZED, 'TestFile', 1.0, VALIDATE_LABELS_UP)

	v.validatePrediction(2, 'laplaceSmoothing', upTrainUnigram, downTrainUnigram, upTrainBigram, downTrainBigram, upTrainTrigram, downTrainTrigram, DOWN_VALIDATE_TOKENIZED, 'TestFile', 1.0, VALIDATE_LABELS_DOWN)

	# v.validatePrediction(2, 'laplaceSmoothing', upTrainUnigram, downTrainUnigram, upTrainBigram, downTrainBigram, None, None, TEST_TOKENIZED, 'BigramPrediction.csv', 1.0, None)