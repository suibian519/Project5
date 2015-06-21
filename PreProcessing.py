import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer


def readFile(filename):
	with open(filename) as fileIn:
		content = fileIn.read()
	return content

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
        return False

def splitToSentences(text):
	# text = text.lower()
	# sentences=re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text) #Split email into sentences
	sentences = sent_tokenize(text)
	tokenized = []
	
	for s in sentences:
		s = s.rstrip() #Remove new line symobol '\n'
		new_s = nltk.word_tokenize(s)
		for i in range(0, len(new_s)):
			new_s[i] = PorterStemmer().stem(new_s[i])
		new_s = ['<s>'] + ['<s>'] + new_s + ['</s>']
		tokenized.append(new_s) #Tokenize sentences

	return sentences,tokenized

def extractCorpus(content):
	content = content.split('**EOM**')

	UPSPEAK_RAW = []
	DOWNSPEAK_RAW = []
	TEST_RAW = []

	UPSPEAK_SENTENCES = []
	DOWNSPEAK_SENTENCES = []
	TEST_SENTENCES = []

	UPSPEAK_TOKENIZED = []
	DOWNSPEAK_TOKENIZED = []
	TEST_TOKENIZED = []

	IS_TEST = False #Return email number if is test dataset


	for message in content:

		message = message.split('**START**')
		message[0].rstrip() #Remove new line

		if 'UPSPEAK' in message[0]:

			sentences,tokenized = splitToSentences(message[1])
			UPSPEAK_RAW.append(message[1]) 
			UPSPEAK_SENTENCES.append(sentences)
			UPSPEAK_TOKENIZED.append(tokenized)

		elif 'DOWNSPEAK' in message[0]:

			sentences,tokenized = splitToSentences(message[1])
			DOWNSPEAK_RAW.append(message[1]) #Preserve raw texts for future reference
			DOWNSPEAK_SENTENCES.append(sentences)
			DOWNSPEAK_TOKENIZED.append(tokenized)

		elif is_number(message[0]): #Test data doesn't labels,only number

			sentences,tokenized = splitToSentences(message[1])
			TEST_RAW.append(message[1]) #Acutal number in testing dataset is array index+1
			TEST_SENTENCES.append(sentences)
			TEST_TOKENIZED.append(tokenized)
			IS_TEST = True

	if IS_TEST == True:
		return TEST_RAW,TEST_SENTENCES,TEST_TOKENIZED
	else:
		return UPSPEAK_RAW,DOWNSPEAK_RAW,UPSPEAK_SENTENCES,DOWNSPEAK_SENTENCES,UPSPEAK_TOKENIZED,DOWNSPEAK_TOKENIZED		

def process():

	#NLTK Corpora http://www.nltk.org/nltk_data/
	#Comment this line once you have NTLK data installed
	#nltk.download() 

	TRAIN = readFile('training.txt')
	VALIDATION = readFile('validation.txt')
	TEST = readFile('test.txt')

	UP_TRAIN_RAW,DOWN_TRAIN_RAW,UP_TRAIN_SENTENCES,DOWN_TRAIN_SENTENCES,UP_TRAIN_TOKENIZED,DOWN_TRAIN_TOKENIZED = extractCorpus(TRAIN)
	UP_VALIDATE_RAW,DOWN_VALIDATE_RAW,UP_VALIDATE_SENTENCES,DOWN_VALIDATE_SENTENCES,UP_VALIDATE_TOKENIZED,DOWN_VALIDATE_TOKENIZED = extractCorpus(VALIDATION)
	TEST_RAW,TEST_SENTENCES,TEST_TOKENIZED = extractCorpus(TEST)


	return UP_TRAIN_RAW,DOWN_TRAIN_RAW,UP_TRAIN_SENTENCES,DOWN_TRAIN_SENTENCES,UP_TRAIN_TOKENIZED,DOWN_TRAIN_TOKENIZED,UP_VALIDATE_RAW,DOWN_VALIDATE_RAW,UP_VALIDATE_SENTENCES,DOWN_VALIDATE_SENTENCES,UP_VALIDATE_TOKENIZED,DOWN_VALIDATE_TOKENIZED,TEST_RAW,TEST_SENTENCES,TEST_TOKENIZED



