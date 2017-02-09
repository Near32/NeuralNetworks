#include "../QLEARNING/QLEARNING.h"


int main(int argc, char* argv[])
{
	std::string filepath("CARTPOLENN");
	unsigned int nbrepi = 200;
	float gamma_ = 0.5f;
	
	float EOE = 10.0f;	//in seconds...
	SimulatorRKCARTPOLE env_(EOE);

	float lr_ = 0.9f;//unused...
	float eps_ = 0.1f;
	int dimActionSpace_ = 1;
	
	int dimStateSpace_ = 4;
	Topology topo;
	unsigned int nbrneurons = 5;
	unsigned int nbrlayer = 10;
	unsigned int nbrinput = dimActionSpace_+dimStateSpace_;
	unsigned int nbroutput = 1;
	//topo.push_back(nbrinput,NTNONE);	//input layer
	topo.push_back(nbrinput,NTSIGMOID);	//input layer
	
	for(int i=nbrlayer;i--;)	topo.push_back(nbrneurons, NTSIGMOID);
	
	//topo.push_back(nbroutput, NTNONE);	//linear output
	topo.push_back(nbroutput, NTSIGMOID);	//linear output
	
	QFANN<float> fa_( lr_, eps_, gamma_, dimActionSpace_, topo);
	 
	QLEARNINGXPReplay instance(nbrepi, gamma_, (Environment<float>*)(&env_), (FA<float>*)&fa_);
	
	instance.run(nbrepi);
	
	instance.save(filepath);
}
