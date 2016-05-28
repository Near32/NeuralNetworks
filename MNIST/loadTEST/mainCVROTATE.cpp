#define OPENCV_USE
#include "../../NN.h"
#include "../OPU/projectOPU1.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <string>

#include <thread>
#include <mutex>

extern mutex rMutex;


int main( int argc, char* argv[])
{
    //Neural Networks settings :
	std::string filepath = "neuralnetworksTESTLOAD.IN";
	NN<float> nn(filepath);
	nn.learning = false;
	
	
    cv::Mat debugIm;
    cv::Mat frame, frame1,frame2;
    bool continuer = true;
    
    //cv::VideoCapture cap(1);
    cv::VideoCapture cap;
    cap.open(1);
    
	if(!cap.isOpened())
	{
	    cerr << "Erreur : Impossible de démarrer la capture video sur 1." << endl;
	    cap.open(0);
	    if(!cap.isOpened())
	    {
	    	cerr << "Erreur : Impossible de démarrer la capture video sur 0." << endl;
	        return -1;
	    }
	}
	
    //Gestion ecriture dans un fichier :
    /*
    string filepath("./log.txt");
    FILE* log = fopen(filepath.c_str(), "w+");
    if(log == NULL)
    {
    	cout << "ERROR : cannot open the file." << endl;
    	exit(1);
    }
    else
    	cout << "File opened." << endl;
    */
    //------------------------------------
    
    cv::Scalar color(255);
    
    while(continuer)
    {
		cap >> frame;

		Mat<float> point(2,1);
		point.set( frame.cols/4., 1,1);
		point.set( frame.rows/4., 2,1);

		double theta = 90;
		cv::Mat r = rotate(frame,theta,point);
						
		cv::imshow("Entry",frame);
		//cv::imshow("Output",r);

		/*
		Mat<float> rect(point,0.0f,1,1,2,2);
		rect.set( 400, 1,2);
		rect.set( 400, 2,2);
		
		cv::imshow("Extract", extractPatch(r,rect) );
		*/
		
		/*
		cv::Point pt(40,60);	
		cv::circle(frame,pt,10,color,10);
		pt = cv::Point(60,70);	
		cv::circle(frame,pt,10,color,10);
		*/
		
		Mat<float> finalAssoc(1,1);
		int resfeat = computeAssociationsAndCOG(frame, finalAssoc, &nn);
		
		for(int i=1;i<=finalAssoc.getLine();i++)
		{
			cv::Point pt(finalAssoc.get(i,3),finalAssoc.get(i,4));
			
			cv::circle(frame,pt,30,cv::Scalar(255,255,0));
		}
		
		cv::imshow("OUTPUT",frame);

		if(cv::waitKey(30)>=0)
		{
			continuer = false;
		}
     	       
    }
    	
   	
	return 0;
}

