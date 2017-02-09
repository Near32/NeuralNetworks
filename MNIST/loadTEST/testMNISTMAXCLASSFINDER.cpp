#define OPENCV_USE
#include "../../NN.h"
#include "../OPU/projectOPU1.h"
#include "../../RunningStats/RunningStats.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <string>

#define debuglvl1


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

Mat<float> reshapeM(const Mat<float>& m, int line, int column)
{
	Mat<float> r(line,column);
	
	for(int i=1;i<=line;i++)
	{
		for(int j=1;j<=column;j++)
		{
			r.set( m.get( (i-1)*column+j,1), i,j);
		}
	}
	
	return r;
}


int main(int argc, char* argv[])
{
	//Neural Networks settings :
	//std::string filepath = "neuralnetworksTESTLOAD.IN";
	std::string filepath = "neuralnetworksDIGITROTATEDPI2";
	NN<float> nn(filepath);
	nn.learning = false;
	//------------------------------
	
	
	float alpha = 1e-4f;
	float momentum = 0.5f;
	//TODO : evaluate the number of it needed.
	std::vector<Mat<float> > input;
	std::vector<Mat<float> > inputIM;
	for(int i=1;i<=10;i++)	input.push_back( Mat<float>( 0.0f, 28*28,1) );
	int nbrlabel = 10;
	for(int label=0;label<nbrlabel;label++)
	{
		Mat<float> grad(0.0f, 28*28,1);
		int nbrit = 0;		
		Mat<float> target(0.0f,10,1);
		target.set( 1.0f, label+1,1);
		grad *= 0.0f;
		
		
		bool continuer = true;
		while(continuer)
		{
			Mat<float> output( nn.feedForward(input[label]));
			
			transpose(output).afficher();
			
			Mat<float> dNNdinput( transpose(nn.getGradientWRTinput()) * (output-target) );
			grad = (1.0f-momentum)*dNNdinput + momentum*grad;
			
			float norme =norme2(grad);
			std::cout << " ITERATION :" << nbrit << "; NORME GRAD : " << norme << std::endl;
			
			input[label] -= alpha*grad;
			
			if(norme > 2e-1f || nbrit < 10)
			{
				nbrit++;
			}
			else
			{
				continuer = false;
			}
			
		}
		
		inputIM.push_back( reshapeM(input[label], 28,28) );
		
		inputIM[label].afficher();
	}
	
	for(int label=0;label<nbrlabel;label++)
	{
		std::cout << " INPUT LABEL : " << label+1 << std::endl;
		inputIM[label].afficher();
		Mat<float> im(255.0f*inputIM[label]);
		nn.feedForward( input[label] ).afficher();
		afficherMat(std::string("label : ")+std::to_string(label), &im, (Mat<float>*)NULL, (Mat<float>*)NULL, false, 4.0f);
		
		
	}
	
	writeInFile(std::string("labelimages"), inputIM);
	
	//nn.save(std::string("neuralnetworksTESTLOAD.OUT"));
		
	return 0;
}
