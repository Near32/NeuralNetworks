#include "../QLEARNING/QLEARNING.h"


int main(int argc, char* argv[])
{
	unsigned int nbrepi = 100;
	float gamma_ = 0.1f;
	
	float EOE = 10.0f;	//in seconds...
	unsigned int nbrRobots_ = 3;
	Mat<float> CoR(0.0f,3,1);
	std::vector<float> desiredR_(nbrRobots_);
	for(int i=nbrRobots_;i--;)	desiredR_[i] = 1.0f;
	
	SimulatorRK env_(EOE, nbrRobots_, CoR, desiredR_);

	float lr_ = 0.01f;
	float eps_ = 0.1f;
	int dimActionSpace_ = nbrRobots_+5;
	
	//int dimStateSpace_ = 2*nbrRobots_+5;
	int dimStateSpace_ = 2*nbrRobots_+5+nbrRobots_*2;
	Topology topo;
	unsigned int nbrneurons = 30;
	unsigned int nbrlayer = 5;
	unsigned int nbrinput = dimActionSpace_+dimStateSpace_;
	unsigned int nbroutput = 1;
	//topo.push_back(nbrinput,NTNONE);	//input layer
	topo.push_back(nbrinput,NTSIGMOID);	//input layer
	
	for(int i=nbrlayer;i--;)	topo.push_back(nbrneurons, NTSIGMOID);
	
	topo.push_back(nbroutput, NTNONE);	//linear output
	//topo.push_back(nbroutput, NTSIGMOID);	//linear output
	
	QFANN<float> fa_( lr_, eps_, gamma_, dimActionSpace_, topo);
	 
	QLEARNINGXPReplay instance(nbrepi, gamma_, (Environment<float>*)(&env_), (FA<float>*)&fa_);
	
	instance.run(nbrepi);
	
}
