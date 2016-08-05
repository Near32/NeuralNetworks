#define sgd_use
#include "../../../../QLEARNING/QLEARNING.h"

//#define USESAVE

int main(int argc, char* argv[])
{
	std::string filepath("CARTPOLENNACTORCRITIC");
	
	std::string filepathPA("./CARTPOLENNACTORCRITIC.PA.NEW");
	
	std::string filepathRSPA(filepath+".PA.txt");
	
	float gamma_ = 0.99f;
	 
	float lrPA_ = 1e-4f;
	
	float eps_ = 0.9f;
	int dimActionSpace_ = 1;
	
	int dimStateSpace_ = 4;
	
	
	QPANN<float> pa_( lrPA_, eps_, gamma_, dimActionSpace_, filepathPA, filepathRSPA);
	
	
	Mat<float> state(0.0f,dimStateSpace_,1);
	std::vector<float> stateval(dimStateSpace_);
	std::vector<float> staterange(dimStateSpace_);
	staterange[0] = 2*10.0f;
	staterange[1] = 2*2*PI;
	staterange[2] = 200.0f;
	staterange[3] = 2*2*PI;
	
	float nbrpoints = 100.0f;
	std::vector<float> statestep;
	for(int i=dimStateSpace_;i--;)	statestep.insert( statestep.begin(), staterange[i]/nbrpoints);
	
	std::vector<Mat<float> > actions;
	for(int i=nbrpoints;i--;)
	{
		for(int j=nbrpoints;j--;)
		{
			actions.push_back( pa_.estimateAction(state) );
		
			theta += steptheta;
			state.set( theta, 2,1);
		
			std::cout << " ITERATION : " << i*nbrpoints+j << " : STATE :" <<  std::endl;
			transpose(state).afficher();
		}
		
		dtheta += stepdtheta;
		state.set( dtheta, 4,1);
		
		theta = -2*PI;
		
	}
	
	writeInFile(std::string("./VIZUALIZATION_Policy.txt"), actions);
	
	return 0;
}
