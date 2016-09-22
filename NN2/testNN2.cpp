//#define USE_ACC

#include "../NN.h"

#define USE_CONV


#ifdef USE_ACC
FProduct<float> fproduct = mtproduct<float>;
#ifdef FPRODUCTDEFINITION
//fproduct = mtproduct<float>;
#endif
#else
FProduct<float> fproduct = product<float>;
#endif

int main(int argc, char* argv[])
{
	float eps = numeric_limits<float>::epsilon();
	Topology topology;
	#ifdef FPRODUCTDEFINITION
	std::cout << "FPRODUCT DEFINED." << std::endl;
	#else
	std::cout << "FPRODUCT NOT DEFINED !!!!!!!!!!!!!!!!" << std::endl;
	#endif
	#ifdef USE_CONV
	// INPUT LAYER :
	unsigned int Hinput = 32;
	unsigned int Winput = 32;
	unsigned int Dinput = 3;
	// NORMAL :
	//topology.push_back(Hinput*Winput*Dinput, NTNONE, LTINPUT);
	//CONVOLUTIONNAL :
	//32x32x3 --> 32x32x3
	convLayerTopoInfo clti_input(CLTINPUT,Hinput,Winput,Dinput,1,1,Dinput);
	//topology.push_back( Hinput*Winput*Dinput, NTNONE, LTINPUT, clti_input);
	topology.push_back( 1, NTNONE, LTINPUT, clti_input);
	 
	//------------------------------
	//		Convolutionnal layers :
	//-------------------------------
	//CONV :
	unsigned int Ffilter1 = 3;
	unsigned int nbrFilter1 = 16;
	unsigned int pad1 = 1;
	unsigned int str1 = 1;
	//32x32x3 --> 32x32x16
	convLayerTopoInfo clti_1(CLTCONV, clti_input.Houtput, clti_input.Woutput, clti_input.Doutput, Ffilter1,Ffilter1,nbrFilter1, str1,pad1);
	//convLayerTopoInfo clti_1(CLTCONV, Hinput, Winput, Dinput, Ffilter1,Ffilter1,nbrFilter1, str1,pad1);
	//topology.push_back( clti_1.Houtput*clti_1.Woutput*clti_1.Doutput, NTNONE, LTCONV, clti_1);
	topology.push_back( 1, NTNONE, LTCONV, clti_1);
	
	//RELU :
	unsigned int Ffilter2 = 1;
	unsigned int nbrFilter2 = 16;
	unsigned int pad2 = 0;
	unsigned int str2 = 1;
	convLayerTopoInfo clti_2(CLTRELU, clti_1.Houtput, clti_1.Woutput, clti_1.Doutput, Ffilter2,Ffilter2,nbrFilter2, str2,pad2);
	//topology.push_back( clti_2.Houtput*clti_2.Woutput*clti_2.Doutput, NTNONE, LTCONV, clti_2);
	topology.push_back( 1, NTNONE, LTCONV, clti_2);
	
	/*
	//CONV :
	unsigned int Ffilter3 = 3;
	unsigned int nbrFilter3 = 8;
	unsigned int pad3 = 1;
	unsigned int str3 = 1;
	//32x32x16 --> 32x32x8
	convLayerTopoInfo clti_3(CLTCONV, clti_2.Houtput, clti_2.Woutput, clti_2.Doutput, Ffilter3,Ffilter3,nbrFilter3, str3,pad3);
	//topology.push_back( clti_3.Houtput*clti_3.Woutput*clti_3.Doutput, NTNONE, LTCONV, clti_3);
	topology.push_back( 1, NTNONE, LTCONV, clti_3);
	
	//RELU :
	unsigned int Ffilter4 = 1;
	unsigned int nbrFilter4 = 8;
	unsigned int pad4 = 0;
	unsigned int str4 = 1;
	//32x32x8
	convLayerTopoInfo clti_4(CLTRELU, clti_3.Houtput, clti_3.Woutput, clti_3.Doutput, Ffilter4,Ffilter4,nbrFilter4, str4,pad4);
	//topology.push_back( clti_4.Houtput*clti_4.Woutput*clti_4.Doutput, NTNONE, LTCONV, clti_4);
	topology.push_back( 1, NTNONE, LTCONV, clti_4);
	/**/
	
	//POOLING :
	unsigned int Ffilter5 = 8;
	//unsigned int nbrFilter5 = nbrFilter4;
	unsigned int nbrFilter5 = nbrFilter2;
	unsigned int pad5 = 0;
	unsigned int str5 = 8;
	//32x32x8 --> 4x4x8
	//convLayerTopoInfo clti_5(CLTPOOL, clti_4.Houtput, clti_4.Woutput, clti_4.Doutput, Ffilter5,Ffilter5,nbrFilter5, str5,pad5);
	convLayerTopoInfo clti_5(CLTPOOL, clti_2.Houtput, clti_2.Woutput, clti_2.Doutput, Ffilter5,Ffilter5,nbrFilter5, str5,pad5);
	topology.push_back( clti_5.Houtput*clti_5.Woutput*clti_5.Doutput, NTNONE, LTCONV, clti_5);
	
	//it is mandatory to put the correct number of neuron expected here...
	//topology.push_back( 1, NTNONE, LTCONV, clti_5);
	
	
	#endif
	
	//------------------------------
	//		Fully connected layers :
	//-------------------------------
	#ifdef USE_CONV
	unsigned int nbrneurons = clti_5.Houtput*clti_5.Woutput*clti_5.Doutput/2; 
	#else
	unsigned int nbrneurons = 512;//clti_5.Houtput*clti_5.Woutput*clti_5.Doutput/2; 
	#endif
	unsigned int nbrlayer = 1;
	#ifdef USE_CONV
	unsigned int nbrinput = nbrneurons*2;
	#else
	unsigned int nbrinput = 32*32*3;//nbrneurons*2;
	#endif
	unsigned int nbroutput = 10;
	
	topology.push_back(nbrinput,NTNONE);	//input layer
	
	for(int i=nbrlayer;i--;)	topology.push_back(nbrneurons, NTRELU);
	
	//topology.push_back(nbroutput, NTSOFTMAX);	//logistic layer
	topology.push_back(nbroutput, NTTANH);	//output layer
	
	clock_t time = clock();
	NN<float> nn(topology);
	std::cout << "The RESSOURCES ALLOCATION took : " << (float)(clock()-time)/(float)(CLOCKS_PER_SEC) << " seconds." << std::endl;
	
	/*
	int nbrVals = 20;
	Mat<float> inputVals((float)eps,nbrinput,nbrVals);
	Mat<float> outputVals((float)eps,nbroutput,nbrVals);
	Mat<float> targetVals((float)eps,nbroutput,nbrVals);
	
	for(int i=nbrinput;i--;)
	{
		inputVals.set( i, i+1,i+1);
	}
	
	for(int i=nbroutput;i--;)
	{
		targetVals.set( i, i+1,i+1);
	}
	
	
	
	for(int k=0;k<=1000;k++)
	{
		float error = 0.0f;
		time = clock();
		
		for(int i=1;i<=nbrVals;i++)
		{
			
	
			outputVals = nn.feedForward(Cola(inputVals,i));
	
			//std::cout << "The FEEDFORWARD PROPAGATION took : " << (float)(clock()-time)/(float)(CLOCKS_PER_SEC) << " seconds." << std::endl;
	
			//nn.backProp(Cola(targetVals,i));
			nn.backPropCrossEntropy(Cola(targetVals,i));
	
			//transpose(outputVals).afficher();
		
			error += norme2(Cola(targetVals,i) - extract(outputVals,1,1, nbroutput,1) )/nbrVals;
		}
		
		std::cout << " ERROR = " << error << " ; " << (float)(clock()-time)/(CLOCKS_PER_SEC) << " seconds." << std::endl;
	}
	*/
	
	#ifdef USE_CONV
	Mat<float> input(1.0f,clti_input.Hinput, clti_input.Winput, clti_input.Dinput);	
	#else
	Mat<float> input(1.0f,nbrinput,1);	
	#endif
	time = clock();
	
	Mat<float> output( nn.feedForward(input) );
	
	std::cout << "The FEEDFORWARD PROPAGATION took : " << (float)(clock()-time)/(float)(CLOCKS_PER_SEC) << " seconds." << std::endl;
	
	return 0;
}
