#include "NN.h"

int main(int argc, char* argv[] )
{
	float eps = numeric_limits<float>::epsilon();
	Topology topology;
	unsigned int nbrneurons = 20;
	unsigned int nbrlayer = 5;
	unsigned int nbrinput = 40;
	unsigned int nbroutput = 10;
	topology.push_back(nbrinput,NTSIGMOID);	//input layer
	
	for(int i=nbrlayer;i--;)	topology.push_back(nbrneurons, NTSIGMOID);
	
	topology.push_back(nbroutput, NTSOFTMAX);	//logistic layer
	//topology.push_back(nbroutput, NTC);	//output layer
	
	clock_t time = clock();
	NN<float> nn(topology);
	std::cout << "The RESSOURCES ALLOCATION took : " << (float)(clock()-time)/(float)(CLOCKS_PER_SEC) << " seconds." << std::endl;
	
	
	int nbrVals = 20;
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
