#define sgd_use
#include "../../../QLEARNING/QLEARNING.h"


int main(int argc, char* argv[])
{
	std::string filepath("CARTPOLENNACTORCRITIC");
	std::string filepathRSFA(filepath+".FA.txt");
	std::string filepathRSPA(filepath+".PA.txt");
	
	unsigned int nbrthread = 16;
	unsigned int nbrepi = 10000;
	float gamma_ = 0.99f;
	
	float EOE = 5.0f;	//in seconds...
	SimulatorRKCARTPOLE env_(EOE);
	
	/*
	2x10 
	float lrPA_ = 1e-2f;
	float lrFA_ = 1e-2f;
	*/
	
	/*
	4x10
	*/
	/* 
	float lrPA_ = 1e-2f;
	float lrFA_ = 1e-2f;
	*/
	
	/*
	4x50
	*/ 
	float lrPA_ = 1e-4f;
	float lrFA_ = 1e-4f;
	/**/
	float eps_ = 0.9f;
	int dimActionSpace_ = 1;
	
	int dimStateSpace_ = 4;
	Topology topoFA;
	unsigned int nbrneuronsFA = 100;
	unsigned int nbrlayerFA = 1;
	unsigned int nbrinputFA = dimActionSpace_+dimStateSpace_;
	unsigned int nbroutputFA = 1;
	topoFA.push_back(nbrinputFA,NTNONE);	//input layer
	//topoFA.push_back(nbrinputFA,NTSIGMOID);	//input layer
	
	//for(int i=nbrlayerFA;i--;)	topoFA.push_back(nbrneuronsFA, NTSIGMOID);
	//for(int i=nbrlayerFA;i--;)	topoFA.push_back(nbrneuronsFA, NTTANH);
	for(int i=nbrlayerFA;i--;)	topoFA.push_back(nbrneuronsFA, NTRELU);
	
	//topoFA.push_back(nbroutputFA, NTNONE);	//linear output
	//topoFA.push_back(nbroutputFA, NTSIGMOID);	//sigmoid output
	topoFA.push_back(nbroutputFA, NTTANH);	//tanh output
	
	QFANN<float> fa_( lrFA_, eps_, gamma_, dimActionSpace_, topoFA, filepathRSFA);
	
	 
	 
	 
	Topology topoPA;
	unsigned int nbrneuronsPA = 100;
	unsigned int nbrlayerPA = 1;
	unsigned int nbrinputPA = dimStateSpace_;
	unsigned int nbroutputPA = dimActionSpace_;
	topoPA.push_back(nbrinputPA,NTNONE);	//input layer
	//topoPA.push_back(nbrinputPA,NTSIGMOID);	//input layer
	
	//for(int i=nbrlayerPA;i--;)	topoPA.push_back(nbrneuronsPA, NTSIGMOID);
	//for(int i=nbrlayerPA;i--;)	topoPA.push_back(nbrneuronsPA, NTTANH);
	for(int i=nbrlayerPA;i--;)	topoPA.push_back(nbrneuronsPA, NTRELU);
	
	//it would be difficult to get to higher values with a nonlinearity that would reduice the range of possibility, maybe...
	topoPA.push_back(nbroutputPA, NTNONE);	//linear output
	//topoPA.push_back(nbroutputPA, NTSIGMOID);	//linear output
	//topoPA.push_back(nbroutputPA, NTTANH);	//linear output
	
	QPANN<float> pa_( lrPA_, eps_, gamma_, dimActionSpace_, topoPA, filepathRSPA);
	 
	//QLEARNINGXPReplay instance(nbrepi, gamma_, (Environment<float>*)(&env_), (FA<float>*)&fa_);
	//QLEARNINGXPReplayActorCritic instance(nbrepi, gamma_, (Environment<float>*)(&env_), (FA<float>*)&fa_, (PA<float>*)&pa_);
	float momentumUpdate = 1e-2f;
	int freqUpdate = 100;
	DDPGA3C instance(nbrepi, gamma_, (Environment<float>*)(&env_), (FA<float>*)&fa_, (PA<float>*)&pa_, momentumUpdate, freqUpdate);
	
	
	instance.run(nbrepi,nbrthread);
	
	fa_.save(filepath+"FA");
	pa_.save(filepath+"PA");
}
