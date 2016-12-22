#define sgd_use
#include "../../../QLEARNING/QLEARNING.h"

//#define USESAVE

int main(int argc, char* argv[])
{
	std::string filepath("CARTPOLENNACTORCRITIC");
	
	std::string filepathPA("./INGDBSAVE/CARTPOLENNACTORCRITIC.PA.NEW");
	
	std::string filepathRSPA(filepath+".PA.txt");
	
	float gamma_ = 0.99f;
	 
	float lrPA_ = 1e-4f;
	
	float eps_ = 0.9f;
	int dimActionSpace_ = 1;
	
	int dimStateSpace_ = 4;
	
	
	QPANN<float> pa_( lrPA_, eps_, gamma_, dimActionSpace_, filepathPA, filepathRSPA);
	
	
	Mat<float> state(0.0f,dimStateSpace_,1);
	float theta = -2*PI;
	float dtheta = 0.0f;
	float thetarange = 4*PI;
	float nbrpoints = 10000.0f;
	float steptheta = (thetarange)/nbrpoints;
	
	std::vector<Mat<float> > actions;
	for(int i=nbrpoints;i--;)
	{
		actions.push_back( pa_.estimateAction(state) );
		
		theta += steptheta;
		state.set( theta, 2,1);
		
	}
	
	writeInFile(std::string("./VIZUALIZATION_Policy.txt"), actions);
	
	return 0;
}
