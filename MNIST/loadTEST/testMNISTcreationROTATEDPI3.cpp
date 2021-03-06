#define OPENCV_USE
#include "../../NN.h"
#include "../OPU/projectOPU1.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <string>

//#define debuglvl1


// Training image file name
const string training_image_fn = "../mnist/train-images.idx3-ubyte";

// Training label file name
const string training_label_fn = "../mnist/train-labels.idx1-ubyte";

// Weights file name
const string model_fn = "model-neural-network.dat";

// Report file name
const string report_fn = "training-report.dat";

// Number of training samples
const int nTraining = 60000;

// Image size in MNIST database
const int width = 28;
const int height = 28;

std::ifstream image,label;
std::ofstream report;


Mat<float> inputMNIST(char& label_val) 
{
	Mat<float> im(height,width);
	// Reading image
    char number;
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            image.read(&number, sizeof(char));
            if (number == 0) 
            {
				//d[i][j] = 0; 
				im.set( 0, j,i);
			} else 
			{
				//d[i][j] = 1;
				im.set( 1, j,i);
			}
        }
	}

#ifdef debuglvl1	
	cout << "Image:" << endl;
	im.afficher();
#endif

	// Reading label
    label.read(&label_val, sizeof(char));
#ifdef debuglvl1
    cout << "Label: " << (int)(label_val) << endl;
#endif
    
    return im;
}

Mat<float> reshapeV(const Mat<float>& m)
{
	int line = m.getLine();
	int column = m.getColumn();
	Mat<float> r(line*column,1);
	
	for(int i=1;i<=line;i++)
	{
		for(int j=1;j<=column;j++)
		{
			r.set( m.get(i,j), (i-1)*column+j, 1);
		}
	}
	
	return r;
}


int main(int argc, char* argv[])
{
	std::vector<Mat<float> > mse;
	float lr = 1e-1f;
	//Neural Networks settings :
	Topology topo;
	unsigned int nbrneurons = 25;
	unsigned int nbrlayer = 1;
	unsigned int nbrinput = width*height;
	unsigned int nbroutput = 10;
	
	//topo.push_back(nbrinput,NTNONE);	//input layer
	topo.push_back(nbrinput,NTSIGMOID);	//input layer
	
	topo.push_back(nbrneurons, NTSIGMOID);
	topo.push_back(15, NTSIGMOID);
	
	//topo.push_back(nbroutput, NTSOFTMAX);	//linear output
	topo.push_back(nbroutput, NTSIGMOID);	//linear output
	
	NN<float> nn(topo, lr);
	nn.learning = false;
	//------------------------------
	/*
	//DATASET SETTINGS :
	report.open(report_fn.c_str(), ios::out);
    image.open(training_image_fn.c_str(), ios::in | ios::binary); // Binary image file
    label.open(training_label_fn.c_str(), ios::in | ios::binary ); // Binary label file

	// Reading file headers
    char number;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));
	}
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));
	}
	
	//------------------------------
	
	//checking rotation :
	Mat<float> im1(8,8, (char)1);
	Mat<float> im2( rotate(im1,PI/2.0f) );
	
	im1.afficher();
	im2.afficher();
	//--------------------------------
	
	//checking arctan !!!
	float y = -4.0f;
	float x = 4.0f;
	std::cout << arctan(y,x)*180.0f/(PI) << std::endl;
	//--------------------------------------------------
	
	//checking reading :
	char labelval = 0;
	float theta = PI/2;
	im1 = inputMNIST(labelval);
	im2 = rotate(im1,theta);
	im2.afficher();
		
	std::cout << "Rotation of : " << theta*180.0f/PI << std::endl;
	
	report.close();
	label.close();
	image.close();
	*/
	char labelval = 0;
	//---------------------------------------------------
	
	int nbrTimes = 3;
	
	int iteration = 50000;
	int offx = 2;
	int offy = 2;
	int size = 28;
	int countSuccess = 0;
	
	for(int k=1;k<=nbrTimes;k++)
	{
	
	
	//----------------------------------------
	//----------------------------------------
	//----------------------------------------
	//DATASET SETTINGS :
	report.open(report_fn.c_str(), ios::out);
    image.open(training_image_fn.c_str(), ios::in | ios::binary); // Binary image file
    label.open(training_label_fn.c_str(), ios::in | ios::binary ); // Binary label file

	// Reading file headers
    char number;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));
	}
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));
	}
	//----------------------------------------
	//----------------------------------------
	//----------------------------------------
    
	
	int batchSize = 100;
	while( iteration)
	{
		Mat<float> inputOriginal( inputMNIST(labelval) );
		//let us choose the rotation :
		float theta = PI;
		
		//let us apply it :
		Mat<float> rotatedinput(extract( rotate(inputOriginal, theta), offx,offy, offx+(size-1), offy+(size-1) ));
		
		Mat<float> inputR( reshapeV( rotatedinput ) );
		Mat<float> inputO( reshapeV( inputOriginal) );
		Mat<float> target( 0.0f, 10,1);
		target.set( 1.0f, labelval+1, 1);
		
		if(labelval < 10)
		{
			Mat<float> output( nn.feedForward( inputR) );
	
			int idmax = idmin( (-1.0f)*output).get(1,1);
	
			transpose( operatorL(target,output) ).afficher();
	
			std::cout << " LEARNING ITERATION : " << iteration << " ; IDMAX = " << idmax << std::endl;
	
	
			nn.backPropBATCH(target,batchSize);
			//nn.backPropCrossEntropy(target);
			
			
			//counting :
			if(idmax == labelval+1)
			{
				countSuccess++;
			}
			
			
			
			//original input :
			output =  nn.feedForward( inputO);
	
			idmax = idmin( (-1.0f)*output).get(1,1);
	
			transpose( operatorL(target,output) ).afficher();
	
			std::cout << " LEARNING ITERATION : " << iteration << " ; IDMAX = " << idmax << std::endl;
	
	
			//nn.backProp(target);
			nn.backPropBATCH(target, batchSize);
			//nn.backPropCrossEntropy(target);
			
			//counting :
			if(idmax == labelval+1)
			{
				countSuccess++;
			}
			
			//-------------------
			
			if( iteration % 1000 == 0)
			{
				std::cout << " TEST : " << countSuccess << " / " << 1000 << std::endl;
				mse.push_back(Mat<float>((float)countSuccess,1,1));
		
				writeInFile(std::string("./mse.txt"), mse);
		
				countSuccess = 0;
			}
			
			iteration--;
			
			
		}
		
		
		
	}
	
	
	if(k!=nbrTimes)
	{
		report.close();
		label.close();
		image.close();
		iteration = 50000;
	}
	
	}
	std::cout << " VALIDATION TEST : in progress.." << std::endl;
	
	iteration = 1000;
	int success = 0;
	while( iteration)
	{
		Mat<float> rotatedinput( inputMNIST(labelval) );
		//let us choose the rotation :
		//float theta = rand()%360;
		float theta = ((float)(rand()%2))*PI;
		
		//let us apply it :
		rotatedinput = extract( rotate(rotatedinput, theta), offx,offy, offx+(size-1), offy+(size-1) );
		
		Mat<float> input( reshapeV( rotatedinput ) );
		Mat<float> target( 0.0f, 10,1);
		target.set( 1.0f, labelval+1, 1);
		
		if(labelval < 10)
		{
			Mat<float> output( nn.feedForward( input));
			int idmax = idmin( (-1.0f)*output).get(1,1);
		
			transpose(output).afficher();
			std::cout << " ITERATION : " << iteration << " ; IDMAX = " << idmax << std::endl;
		
			if(idmax == labelval+1)
			{
				success++;
			}
			
			iteration--;
		}
		
	}
	
	std::cout << "VALIDATION TEST : " << success << " / 1000." << std::endl;
	
	report.close();
	label.close();
	image.close();
	
	nn.save(std::string("neuralnetworksDIGITROTATEDPI3"));
		
	return 0;
}
