#include "../../../QLEARNING/QLEARNING.h"


int main(int argc, char* argv[])
{
	std::string filepath("CARTPOLENNACTORCRITIC");
	unsigned int nbrepi = 1000;
	float gamma_ = 0.5f;
	
	float EOE = 5.0f;	//in seconds...
	SimulatorRKCARTPOLE env_(EOE);

	float lr_ = 1e-1f;//unused...
	float eps_ = 0.1f;
	int dimActionSpace_ = 1;
	
	int dimStateSpace_ = 4;
	Topology topoFA;
	unsigned int nbrneuronsFA = 5;
	unsigned int nbrlayerFA = 2;
	unsigned int nbrinputFA = dimActionSpace_+dimStateSpace_;
	unsigned int nbroutputFA = 1;
	//topo.push_back(nbrinput,NTNONE);	//input layer
	topoFA.push_back(nbrinputFA,NTSIGMOID);	//input layer
	
	for(int i=nbrlayerFA;i--;)	topoFA.push_back(nbrneuronsFA, NTSIGMOID);
	
	topoFA.push_back(nbroutputFA, NTNONE);	//linear output
	//topo.push_back(nbroutput, NTSIGMOID);	//linear output
	
	QFANN<float> fa_( lr_, eps_, gamma_, dimActionSpace_, topo);
	 
	 
	 
	 
	 
	Topology topoPA;
	unsigned int nbrneuronsPA = 5;
	unsigned int nbrlayerPA = 2;
	unsigned int nbrinputPA = dimStateSpace_;
	unsigned int nbroutputPA = dimActionSpace_;
	//topo.push_back(nbrinput,NTNONE);	//input layer
	topoPA.push_back(nbrinputPA,NTSIGMOID);	//input layer
	
	for(int i=nbrlayerPA;i--;)	topoPA.push_back(nbrneuronsPA, NTSIGMOID);
	
	topoPA.push_back(nbroutputPA, NTNONE);	//linear output
	//topo.push_back(nbroutput, NTSIGMOID);	//linear output
	
	QPANN<float> pa_( lr_, eps_, gamma_, dimActionSpace_, topo);
	
	
	 
	QLEARNINGXPReplay instance(nbrepi, gamma_, (Environment<float>*)(&env_), (FA<float>*)&fa_);
	
	instance.run(nbrepi);
	
	fa_.save(filepath);
}
