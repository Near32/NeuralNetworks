#define sgd_use
#include "../../../../QLEARNING/QLEARNING.h"


int main(int argc, char* argv[])
{
	std::string filepath("CARTPOLENNACTORCRITIC");
	
	//std::string filepathPA("./CARTPOLENNACTORCRITIC.PA.NEW");
	std::string filepathPA("./CARTPOLENNACTORCRITIC.PA");
	
	std::string filepathRSPA(filepath+".PA.txt");
	
	float gamma_ = 0.99f;
	 
	float lrPA_ = 1e-4f;
	
	float eps_ = 0.9f;
	int dimActionSpace_ = 1;
	int dimStateSpace_ = 4;
	
	
	QPANN<float> pa_( lrPA_, eps_, gamma_, dimActionSpace_, filepathPA, filepathRSPA);
	Mat<float> meanS(4,1);
	meanS(1,1) = 0.333955f;
	meanS(2,1) = 3.14722f;
	meanS(3,1) = 0.105496f;
	meanS(4,1) = 0.00110895f;
	Mat<float> stdS(4,1);
	stdS(1,1) = 21.2932f;
	stdS(2,1) = 1.01208f;
	stdS(3,1) = 5.10674f;
	stdS(4,1) = 3.41514f;
	pa_.setInputNormalization(meanS,stdS);

	
	
	float EOE = 10.0f;	//in seconds...
	SimulatorRKCARTPOLE env_(EOE);
	env_.idxAssociatedThread = -1;
	env_.write = true;
	env_.initialize(false);

	int nbrsteps = 1;
	std::vector<Mat<float> > states;
	std::vector<Mat<float> > actions;
	
	while( !env_.isTerminal() )
	{

		Mat<float> St(env_.getCurrentState() );
		Mat<float> At(pa_.estimateAction(St));
		//Mat<float> Qsat( this->fa->getQvalue(St,At) );

		Mat<float> Rt(env_.executeAction(At));
		//Mat<float> S1t(env_.getCurrentState());
	
		//Mat<float> A1t(pa_.estimateAction(S1t));

		/*
		Mat<float> Qs1a1t(  this->fa->getQvalue(S1t,A1t) );
		Mat<float> error(Rt+gamma*Qs1a1t-Qsat);
		tempLOSS += error % error;
		if( tempTMR.get(1,1) < Rt.get(1,1) )
		{
			tempTMR.set( Rt.get(1,1), 1,1);
		}
		*/
	
		states.push_back( transpose(St) );
		actions.push_back( At);
		
		nbrsteps++;
		std::cout << " VALIDATION : step : " << nbrsteps << " ; state : " << std::endl;
		transpose(St).afficher();
	}
	
		
	writeInFile(std::string("./VALIDATION_Policy.txt"), states);
	writeInFile(std::string("./VALIDATION_Policy_ACTIONS.txt"), actions);
	
	return 0;
}

	
