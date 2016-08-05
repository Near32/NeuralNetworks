#define sgd_use
#define Vvalues

#ifdef Vvalues
#define ACTORCRITICVVALUES
#else
#define ACTORCRITICDDPG
#endif

#define ENVnorandomINIT

#include "../../../../QLEARNING/QLEARNING.h"

//#define USESAVE
#ifndef USESAVE
	//#define USESAVEFA
#endif

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
	unsigned int nbrepi = 50000;
	float gamma_ = 0.99f;
	
	float EOE = 10.0f;	//in seconds...
	SimulatorRKCARTPOLE env_(EOE);
	env_.idxAssociatedThread = -1;
	env_.write = true;
	
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
	float lrPA_ = 1e-3f;
	//we do not want to reach some local minima before finishing learning the Qvalues...
	float lrFA_ = 1e-3f;
	/**/
	float eps_ = 0.03f;
	int dimActionSpace_ = 1;
	
	int dimStateSpace_ = 4;
	
	/**/
	#ifndef USESAVE
		#ifndef USESAVEFA
	Topology topoFA;
	unsigned int nbrneuronsFA = 256;
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
	topoFA.push_back(nbrneuronsFA/4, NTRELU);
	
	topoFA.push_back(nbroutputFA, NTNONE);	//linear output
	//topoFA.push_back(nbroutputFA, NTSIGMOID);	//sigmoid output
	//topoFA.push_back(nbroutputFA, NTTANH);	//tanh output
	
	QFANN<float> fa_( lrFA_, eps_, gamma_, dimActionSpace_, topoFA, filepathRSFA);
	#ifdef Vvalues
	fa_.VvaluesOnly = true;
	#endif
		#else
	QFANN<float> fa_( lrFA_, eps_, gamma_, dimActionSpace_, filepathFA, filepathRSFA); 
	#ifdef Vvalues
	fa_.VvaluesOnly = true;
	#endif	
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
	unsigned int nbrneuronsPA = 128;
	unsigned int nbrlayerPA = 1;
	unsigned int nbrinputPA = dimStateSpace_;
	unsigned int nbroutputPA = dimActionSpace_;
	topoPA.push_back(nbrinputPA,NTNONE);	//input layer
	//topoPA.push_back(nbrinputPA,NTSIGMOID);	//input layer
	
	//for(int i=nbrlayerPA;i--;)	topoPA.push_back(nbrneuronsPA, NTSIGMOID);
	//for(int i=nbrlayerPA;i--;)	topoPA.push_back(nbrneuronsPA, NTTANH);
	for(int i=nbrlayerPA;i--;)	topoPA.push_back(nbrneuronsPA, NTRELU);
	topoPA.push_back(nbrneuronsPA/4, NTRELU);
	
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
	//float momentumUpdate = 1e-4f;
	float momentumUpdate = 1e-4f;
	//int freqUpdate = 10;
	int freqUpdate = 1;
	DDPGA3C instance(nbrepi, gamma_, (Environment<float>*)(&env_), (FA<float>*)&fa_, (PA<float>*)&pa_, momentumUpdate, freqUpdate);
	
	
	instance.run(nbrepi,nbrthread);
	
	fa_.save(filepath+"FA");
	pa_.save(filepath+"PA");
}
