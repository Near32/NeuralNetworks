#include "../../QLEARNING/QLEARNING.h"


int main(int argc, char* argv[])
{
	std::string filepath("CARTPOLENN");
	unsigned int nbrepi = 100;
	float gamma_ = 0.5f;
	
	float EOE = 5.0f;	//in seconds...
	SimulatorRKCARTPOLE env_(EOE);

	float lr_ = 5e-2f;//unused...
	
	if( argc>1)
	{
		lr_ = atof(argv[1]);
	}
	
	float eps_ = 0.1f;
	int dimActionSpace_ = 1;
	
	int dimStateSpace_ = 4;
	
	QFANN<float> fa_( lr_, eps_, gamma_, dimActionSpace_, filepath);
	 
	QLEARNINGXPReplay instance(nbrepi, gamma_, (Environment<float>*)(&env_), (FA<float>*)&fa_);
	
	instance.run(nbrepi);
	
	fa_.save(filepath);
}
