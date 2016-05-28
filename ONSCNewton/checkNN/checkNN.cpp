#include "../ONSCNewton.h"
#include "../../NN.h"

int main(int argc, char* argv[])
{
	Topology topo;
	unsigned int nbrneurons = 10;
	unsigned int nbrlayer = 2;
	unsigned int nbrinput = 10;
	unsigned int nbroutput = 1;
	topo.push_back(nbrinput,NTNONE);	//input layer
	
	for(int i=nbrlayer;i--;)	topo.push_back(nbrneurons, NTSIGMOID);
	
	topo.push_back(nbroutput, NTNONE);	//linear output
	
	NN<float> instanceNN( topo);
	

	int nbrit = 500;
	float alpha = 1e-1f*nbrlayer;
	//TODO : evaluate the number of it needed.
	
	
	Mat<float> grad( 1, nbrinput);
	Mat<float> Ainit(nbrinput,1,(char)1);
	Mat<float> A(Ainit);
	
	while(nbrit--)
	{
		Mat<float> output( instanceNN.feedForward(A) );
		
		
		grad = instanceNN.getGradientWRTinput();
		
		
		std::cout << "VALUE AT A : " << (float)(output.get(1,1)) << " ; GRAD norme2 = " << norme2(grad) << std::endl;
		
		A += alpha*transpose(grad);
		//CONFIRMED : since it is a maximization : + !!!
	}

	std::cout << "AT THE BEGINNING vs END : " << std::endl;
	operatorL(Ainit,A).afficher();
	
	return 0;
}