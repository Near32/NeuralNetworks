#define sgd_use

#include "../../../QLEARNING/QLEARNING.h"


int main(int argc, char* argv[])
{
	std::string filepath("CARTPOLENNACTORCRITIC");
	std::string filepathRSFA(filepath+".FA.txt");
	std::string filepathRSPA(filepath+".PA.txt");
	
	unsigned int nbrepi = 100;
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
	 
	float lrPA_ = 1e-2f;
	float lrFA_ = 1e-2f;
	/**/
	
	/*
	2x50
	*/
	/*
	float lrPA_ = 1e-2f;
	float lrFA_ = 1e-2f;
	*/
	float eps_ = 0.9f;
	int dimActionSpace_ = 1;
	
	int dimStateSpace_ = 4;
	Topology topoFA;
	std::vector<unsigned int> nbrneuronsFA;
	//nbrneuronsFA.push_back(10);
	//nbrneuronsFA.push_back(10);
	nbrneuronsFA.push_back(200);
	nbrneuronsFA.push_back(100);
	
	unsigned int nbrlayerFA = nbrneuronsFA.size();
	unsigned int nbrinputFA = dimActionSpace_+dimStateSpace_;
	unsigned int nbroutputFA = 1;
	topoFA.push_back(nbrinputFA,NTNONE);	//input layer
	//topoFA.push_back(nbrinputFA,NTSIGMOID);	//input layer
	
	//for(int i=nbrlayerFA;i--;)	topoFA.push_back(nbrneuronsFA, NTSIGMOID);
	for(int i=0;i<nbrlayerFA;i++)	topoFA.push_back(nbrneuronsFA[i], NTTANH);
	//for(int i=0;i<nbrlayerFA;i++)	topoFA.push_back(nbrneuronsFA[i], NTNONE);
	
	//topoFA.push_back(nbroutputFA, NTNONE);	//linear output
	//topoFA.push_back(nbroutputFA, NTSIGMOID);	//linear output
	topoFA.push_back(nbroutputFA, NTTANH);	//linear output
	
	QFANN<float> fa_( lrFA_, eps_, gamma_, dimActionSpace_, topoFA, filepathRSFA);
	
	 
	 
	 
	Topology topoPA;
	unsigned int nbrneuronsPA = 50;
	std::vector<unsigned int> nbrneuronsPA;
	//nbrneuronsPA.push_back(10);
	//nbrneuronsPA.push_back(10);
	nbrneuronsPA.push_back(200);
	nbrneuronsPA.push_back(100);
	int nbrlayerPA = nbrneuronsPA.size();
	
	unsigned int nbrinputPA = dimStateSpace_;
	unsigned int nbroutputPA = dimActionSpace_;
	topoPA.push_back(nbrinputPA,NTNONE);	//input layer
	//topoPA.push_back(nbrinputPA,NTSIGMOID);	//input layer
	
	//for(int i=nbrlayerPA;i--;)	topoPA.push_back(nbrneuronsPA, NTSIGMOID);
	for(int i=0;i<nbrlayerPA;i++)	topoPA.push_back(nbrneuronsPA[i], NTTANH);
	//for(int i=0;i<nbrlayerPA;i++)	topoPA.push_back(nbrneuronsPA[i], NTNONE);
	
	//it would be difficult to get to higher values with a nonlinearity that would reduice the range of possibility, maybe...
	//topoPA.push_back(nbroutputPA, NTNONE);	//linear output
	//topoPA.push_back(nbroutputPA, NTSIGMOID);	//linear output
	topoPA.push_back(nbroutputPA, NTTANH);	//linear output
	
	QPANN<float> pa_( lrPA_, eps_, gamma_, dimActionSpace_, topoPA, filepathRSPA);
	 
	//QLEARNINGXPReplay instance(nbrepi, gamma_, (Environment<float>*)(&env_), (FA<float>*)&fa_);
	//QLEARNINGXPReplayActorCritic instance(nbrepi, gamma_, (Environment<float>*)(&env_), (FA<float>*)&fa_, (PA<float>*)&pa_);
	float momentumUpdate = 0.0001f;
	QLEARNINGXPReplayActorCriticFixedTarget instance(nbrepi, gamma_, (Environment<float>*)(&env_), (FA<float>*)&fa_, (PA<float>*)&pa_, momentumUpdate);
	
	instance.run(nbrepi);
	
	fa_.save(filepath+"FA");
	pa_.save(filepath+"PA");
}
