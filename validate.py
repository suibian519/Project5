import math
import laplaceSmoothing
import goodTuring
import backOff




def predict(up,down):

	if len(up)!=len(down):

		print('Invalid input. Length of two lists should be equal')

	prediction=[]


	for i in range(0,len(up)):

		#0 = DOWNSPEAK and 1 = UPSPEAK
		if (math.log((1489 + 287)/(1385 + 1489 + 334 + 287)) + up[i]) > (math.log((1385 + 334)/(1385 + 1489 + 334 + 287)) + down[i]):

			prediction.append(1)

		else:

			prediction.append(0)

	return prediction


def outputCSV(prediction,filename):

	outputContent='Id,Prediction\n'

	for i in range(0,len(prediction)):

		outputContent=outputContent+str(i+1)+','+str(prediction[i])

		if i!=len(prediction):
			outputContent=outputContent+'\n'

	fd=open(filename,'w')
	fd.write(outputContent)
	fd.close()


def validationCheck(prediction,labels,modelName,N,laplaceDelata=None):

	correctCount=0

	if len(prediction)==len(labels):

		for i in range(0,len(prediction)):

			if prediction[i]==labels[i]:

				correctCount+=1


		accuracy=float(correctCount/len(prediction))

		print ('\n\n'+modelName+' '+str(N)+' gram\n'+'Accuracy is '+str(accuracy))

		if laplaceDelata!=None and laplaceDelata!=1:
			print ('Delta for laplace smoothing is '+str(laplaceDelata)+'\n\n')

	return accuracy

def validatePrediction(N,SmoothingModel,UniGramUpTrain,UniGramDownTrain,BiGramUpTrain,BiGramDownTrain,TriGramUpTrain,TriGramDownTrain,NGramTest,outputFileName,laplaceDelta=1,labels=None):


	if SmoothingModel=='laplaceSmoothing':

		if N==1:
			
			UpPredictProb,UpDict=laplaceSmoothing.uniLaSmooth(UniGramUpTrain,NGramTest,laplaceDelta)
			DownPredictProb,DownDict=laplaceSmoothing.uniLaSmooth(UniGramDownTrain,NGramTest,laplaceDelta)
		
		elif N==2:

			UpPredictProb,UpDict=laplaceSmoothing.biLaSmooth(UniGramUpTrain, BiGramUpTrain,NGramTest,laplaceDelta)
			DownPredictProb,DownDict=laplaceSmoothing.biLaSmooth(UniGramDownTrain, BiGramDownTrain,NGramTest,laplaceDelta)
		
		elif N==3:

			UpPredictProb,UpDict=laplaceSmoothing.triLaSmooth(UniGramUpTrain, BiGramUpTrain, TriGramUpTrain,NGramTest,laplaceDelta)
			DownPredictProb,DownDict=laplaceSmoothing.triLaSmooth(UniGramDownTrain, BiGramDownTrain, TriGramDownTrain,NGramTest,laplaceDelta)


	elif SmoothingModel=='goodTuring':


		if N==1:

			UpPredictProb=goodTuring.uniGT(UniGramUpTrain,NGramTest)
			DownPredictProb=goodTuring.uniGT(UniGramDownTrain,NGramTest)

		elif N==2:

			UpPredictProb=goodTuring.biGT(BiGramUpTrain,NGramTest)
			DownPredictProb=goodTuring.biGT(BiGramDownTrain,NGramTest)

		elif N==3:

			UpPredictProb=goodTuring.triGT(TriGramUpTrain,NGramTest)
			DownPredictProb=goodTuring.triGT(TriGramDownTrain,NGramTest)


	elif SmoothingModel=='backOff':

		if N==2:

			UpPredictProb=backOff.bigramsBackOff(UniGramUpTrain, BiGramUpTrain,NGramTest)
			DownPredictProb=backOff.bigramsBackOff(UniGramDownTrain, BiGramDownTrain,NGramTest)

	
		elif N==3:

			UpPredictProb=backOff.trigramsBackOff(UniGramUpTrain, BiGramUpTrain,TriGramUpTrain,NGramTest)
			DownPredictProb=backOff.trigramsBackOff(UniGramDownTrain, BiGramDownTrain,TriGramDownTrain,NGramTest)


	prediction=predict(UpPredictProb,DownPredictProb)
	outputCSV(prediction,outputFileName)

	if labels!=None:
		accuracy=validationCheck(prediction,labels,SmoothingModel,N)
	else:
		accuracy=0.0	


	return accuracy



