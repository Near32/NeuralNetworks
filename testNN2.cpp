#include "NN.h"

int main(int argc, char* argv[] )
{
	float eps = numeric_limits<float>::epsilon();
	Topology topology;
	unsigned int nbrneurons = 40;
	unsigned int nbrlayer = 3;
	unsigned int nbrinput = 20;
	unsigned int nbroutput = 10;
	/*usual parameters*/
	
	/*convolutionnal layers parameters*/
	unsigned int Ix = nbrinput;
	unsigned int Iy = nbrinput;
	unsigned int K0 = 1;
	unsigned int nbrConvLayers = 2;
	std::vector<unsigned int> K,F,S,poolingF,poolingS;
	
	K.push_back(K0);
	
	//let us have an identical nbr of kernels at each layer :
	unsigned int nbrKernels = 10;
	for(int i=1;i<nbrConvLayers;i++)	K.push_back(nbrKernels);
	//let us have an identical stride :
	unsigned int stride = 1;
	for(int i=1;i<nbrConvLayers;i++)	S.push_back(stride);
	
	//let us have  filter rf size growing :
	unsigned int rf = 2;
	for(int i=1;i<nbrConvLayers;i++)	F.push_back(rf++);
	
	poolingF.push_back(3);
	poolingS.push_back(2);
	
	topology.addConvLayers( Ix,Iy,K0,K,F,S,poolingF,poolingS,nbrConvLayers);
	//TODO : compute automatically the new input :
	nbrinput = 640;
	/*---------------------------------*/
	
	topology.push_back(nbrinput,NTSIGMOID);	//input layer
	
	for(int i=nbrlayer;i--;)	topology.push_back(nbrneurons, NTSIGMOID);
	
	topology.push_back(nbroutput, NTSOFTMAX);	//logistic layer
	//topology.push_back(nbroutput, NTC);	//output layer
	
	clock_t time = clock();
	NN<float> nn(topology);
	std::cout << "The RESSOURCES ALLOCATION took : " << (float)(clock()-time)/(float)(CLOCKS_PER_SEC) << " seconds." << std::endl;
	
	
	int nbrVals = 100;
	Mat<float> inputVals((float)eps,nbrinput,nbrVals);
	Mat<float> outputVals((float)eps,nbroutput,nbrVals);
	Mat<float> targetVals((float)eps,nbroutput,nbrVals);
	
	for(int i=nbrinput;i--;)
	{
		inputVals.set( i, i+1,i+1);
	}
	
	for(int i=nbroutput;i--;)
	{
		targetVals.set( i, i+1,i+1);
	}
	
	
	for(int k=0;k<=1000;k++)
	{
		float error = 0.0f;
		time = clock();
		
		for(int i=1;i<=nbrVals;i++)
		{
			
	
			outputVals = nn.feedForward(Cola(inputVals,i));
	
			//std::cout << "The FEEDFORWARD PROPAGATION took : " << (float)(clock()-time)/(float)(CLOCKS_PER_SEC) << " seconds." << std::endl;
	
			//nn.backProp(Cola(targetVals,i));
			nn.backPropCrossEntropy(Cola(targetVals,i));
	
			//transpose(outputVals).afficher();
		
			error += norme2(Cola(targetVals,i) - extract(outputVals,1,1, nbroutput,1) )/nbrVals;
		}
		
		std::cout << " ERROR = " << error << " ; " << (float)(clock()-time)/(CLOCKS_PER_SEC) << " seconds." << std::endl;
	}
		
	return 0;
}