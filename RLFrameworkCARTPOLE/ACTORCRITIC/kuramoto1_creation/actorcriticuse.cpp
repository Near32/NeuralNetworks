#define sgd_use
//#define Vvalues

#ifdef Vvalues
#define ACTORCRITICVVALUES
#endif

#define ACTORCRITICDDPG

#define ENVnorandomINIT

#define ENVrestrictXPOS
#define ENVpenalizeFORCE

#define kuramoto1

#include "../../../QLEARNING/QLEARNING.h"

//#define USESAVE


int main(int argc, char* argv[])
{
	std::string filepath("CARTPOLENNACTORCRITIC");
	#ifdef USESAVE
	std::string filepathFA("./INGDBSAVE/CARTPOLENNACTORCRITIC.FA");
	std::string filepathPA("./INGDBSAVE/CARTPOLENNACTORCRITIC.PA");
	#else
	std::string filepathFA("./INGDBSAVE/CARTPOLENNACTORCRITIC.FA.NEW");
	std::string filepathPA("./INGDBSAVE/CARTPOLENNACTORCRITIC.PA.NEW");
	#endif
	std::string filepathRSFA(filepath+".FA.txt");
	std::string filepathRSPA(filepath+".PA.txt");
	
	unsigned int nbrthread = 4;
	unsigned int nbrepi = 10000;
	float gamma_ = 0.99f;
	
	#ifndef kuramoto1
	float EOE = 10.0f;	//in seconds...
	SimulatorRKCARTPOLE env_(EOE);
	env_.idxAssociatedThread = -1;
	env_.write = true;
	int dimActionSpace_ = 1;	
	int dimStateSpace_ = 4;
	#else
	float EOE = 10.0f;	//in seconds...
	unsigned int nbrRobots_ = 3;
	Mat<float> CoR(0.0f,3,1);
	std::vector<float> desiredR_(nbrRobots_);
	for(int i=nbrRobots_;i--;)	desiredR_[i] = 1.0f;
	
	SimulatorRK env_(EOE, nbrRobots_, CoR, desiredR_);
	int dimActionSpace_ = nbrRobots_+5;
	//int dimStateSpace_ = 2*nbrRobots_+5;
	int dimStateSpace_ = 2*nbrRobots_+5+nbrRobots_*2;
	#endif
	
	
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
	//we do not want to reach some local minima before finishing learning the Qvalues...
	float lrFA_ = 1e-4f;
	/**/
	float eps_ = 0.01f;
	
	/**/
	#ifndef USESAVE
	Topology topoFA;
	unsigned int nbrneuronsFA = 64*4;
	unsigned int nbrlayerFA = 1;
	#ifndef Vvalues
	unsigned int nbrinputFA = dimActionSpace_+dimStateSpace_;
	#else
	unsigned int nbrinputFA = dimStateSpace_;
	#endif
	unsigned int nbroutputFA = 1;
	topoFA.push_back(nbrinputFA,NTNONE);	//input layer
	//topoFA.push_back(nbrinputFA,NTSIGMOID);	//input layer
	
	//for(int i=nbrlayerFA;i--;)	topoFA.push_back(nbrneuronsFA, NTSIGMOID);
	//for(int i=nbrlayerFA;i--;)	topoFA.push_back(nbrneuronsFA, NTTANH);
	for(int i=nbrlayerFA;i--;)	topoFA.push_back(nbrneuronsFA, NTRELU);
	topoFA.push_back(32, NTRELU);
	
	topoFA.push_back(nbroutputFA, NTNONE);	//linear output
	//topoFA.push_back(nbroutputFA, NTSIGMOID);	//sigmoid output
	//topoFA.push_back(nbroutputFA, NTTANH);	//tanh output
	
	QFANN<float> fa_( lrFA_, eps_, gamma_, dimActionSpace_, topoFA, filepathRSFA);
	#ifdef Vvalues
	fa_.VvaluesOnly = true;
	#endif
	/**/
	#else
	
	QFANN<float> fa_( lrFA_, eps_, gamma_, dimActionSpace_, filepathFA, filepathRSFA); 
	#ifdef Vvalues
	fa_.VvaluesOnly = true;
	#endif
	#endif 
	 
	#ifndef USESAVE
	Topology topoPA;
	unsigned int nbrneuronsPA = 32*4;
	unsigned int nbrlayerPA = 1;
	unsigned int nbrinputPA = dimStateSpace_;
	unsigned int nbroutputPA = dimActionSpace_;
	topoPA.push_back(nbrinputPA,NTNONE);	//input layer
	//topoPA.push_back(nbrinputPA,NTSIGMOID);	//input layer
	
	//for(int i=nbrlayerPA;i--;)	topoPA.push_back(nbrneuronsPA, NTSIGMOID);
	//for(int i=nbrlayerPA;i--;)	topoPA.push_back(nbrneuronsPA, NTTANH);
	for(int i=nbrlayerPA;i--;)	topoPA.push_back(nbrneuronsPA, NTRELU);
	topoPA.push_back(nbrneuronsPA/2, NTRELU);
	
	//it would be difficult to get to higher values with a nonlinearity that would reduice the range of possibility, maybe...
	topoPA.push_back(nbroutputPA, NTNONE);	//linear output
	//topoPA.push_back(nbroutputPA, NTSIGMOID);	//linear output
	//topoPA.push_back(nbroutputPA, NTTANH);	//linear output
	
	QPANN<float> pa_( lrPA_, eps_, gamma_, dimActionSpace_, topoPA, filepathRSPA);
	#else
	QPANN<float> pa_( lrPA_, eps_, gamma_, dimActionSpace_, filepathPA, filepathRSPA);
	#endif
	//QLEARNINGXPReplay instance(nbrepi, gamma_, (Environment<float>*)(&env_), (FA<float>*)&fa_);
	//QLEARNINGXPReplayActorCritic instance(nbrepi, gamma_, (Environment<float>*)(&env_), (FA<float>*)&fa_, (PA<float>*)&pa_);
	//instance.run(nbrepi);
	
	float momentumUpdate = 1e-3f;
	int freqUpdate = 1;
	#ifndef kuramoto1
	DDPGA3C instance(nbrepi, gamma_, (Environment<float>*)(&env_), (FA<float>*)&fa_, (PA<float>*)&pa_, momentumUpdate, freqUpdate);
	#else
	DDPGA3C instance(nbrepi, gamma_, (Environment<float>*)(&env_), (FA<float>*)&fa_, (PA<float>*)&pa_, momentumUpdate, freqUpdate, dimActionSpace_);
	#endif
	instance.run(nbrepi,nbrthread);
	
	
	
	fa_.save(filepath+"FA");
	pa_.save(filepath+"PA");
}