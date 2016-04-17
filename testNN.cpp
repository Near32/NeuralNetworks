#include "NN.h"

int main(int argc, char* argv[] )
{
	Topology topology;
	unsigned int nbrneurons = 5000;
	topology.NperNLayer.push_back(1000);
	topology.NperNLayer.push_back(nbrneurons);
	
	topology.NperNLayer.push_back(nbrneurons);
	topology.NperNLayer.push_back(nbrneurons);
	
	topology.NperNLayer.push_back(nbrneurons);
	topology.NperNLayer.push_back(nbrneurons);
	
	topology.NperNLayer.push_back(nbrneurons);
	topology.NperNLayer.push_back(nbrneurons);
	
	topology.NperNLayer.push_back(nbrneurons);
	topology.NperNLayer.push_back(nbrneurons);
	
	topology.NperNLayer.push_back(500);
	
	clock_t time = clock();
	NN<float> nn(topology);
	std::cout << "The RESSOURCES ALLOCATION took : " << (float)(clock()-time)/(float)(CLOCKS_PER_SEC) << " seconds." << std::endl;
	
	
	int nbrVals = 10;
	Mat<float> inputVals((float)0,1000,nbrVals);
	Mat<float> outputVals((float)0,500,nbrVals);
	Mat<float> targetVals((float)0,500,nbrVals);
	
	for(int i=1000;i--;)
	{
		inputVals.set( i, i+1,i+1);
	}
	
	for(int i=500;i--;)
	{
		targetVals.set( i, i+1,i+1);
	}
	
	
	time = clock();
	
	outputVals = nn.feedForward(Cola(inputVals,1));
	
	std::cout << "The FEEDFORWARD PROPAGATION took : " << (float)(clock()-time)/(float)(CLOCKS_PER_SEC) << " seconds." << std::endl;
	
	nn.backProp(Cola(targetVals,1));
	//nn.getOutputs(Cola(outputVals,1));
	
	transpose(outputVals).afficher();
	
	return 0;
}
