#define OPENCV_USE
#include "../../NN.h"
#include "../OPU/projectOPU1.h"

#define useAsTensor
#include "../../SparseMat/SparseMat.h"


#include <iostream>
#include <fstream>
#include <cstring>
#include <string>

#include <thread>
#include <mutex>

extern std::mutex mutexRes;


int main1( int argc, char* argv[])
{
    //Neural Networks settings :
	std::string filepath = "neuralnetworksDIGITROTATEDPI";
	NN<float> nn(filepath);
	nn.learning = false;
	
	float treshDist = 500.0f;
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
		int resfeat = computeAssociationsAndCOG(frame, finalAssoc, &nn, treshDist);
		
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





int main( int argc, char* argv[])
{
    //Sensor Settings :	
	float treshDist = 300.0f;
	
	bool erosion = false;
	bool dissect = true;
	bool dilate = true;
	
	sensorCAM sensor(treshDist, erosion, dissect,dilate);
	float ratio = 10.0f;
	float decay = 0.90f;
	float tresholdValuable = 255.0f*0.3f;
	int sizeMapW = 640/ratio;
	int sizeMapH = 480/ratio;
	SparseMat<Mat<float> > pdDigit(sizeMapH,sizeMapW);
	for(int i=1;i<=sizeMapH;i++)
	{
		for(int j=1;j<=sizeMapW;j++)
		{
			pdDigit.set( i,j, Mat<float>(1.0f,10,1) );
		}
	}
	
	
	//-------------------------------------
	//-------------------------------------
		
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
	
    //------------------------------------
    
    cv::Scalar color(255);
    std::thread tSensor( &sensorCAM::runLoop, std::ref(sensor) );
    //cv::namedWindow("OUTPUT RANGE : ");
    cv::namedWindow("OUTPUT");
    cv::namedWindow("CIRCLES");
	
    
    
    if(argc > 1)
	{
		alpha = atof(argv[1]);
	}
	std::cout << " ALPHA == " << alpha << std::endl;
	
	
    while(continuer)
    {
		cap >> frame;
		
		//sensor << frame;
		sensor << frame;
		
		
		Mat<float> finalAssoc(1,1);
		bool trustable = sensor.getAssoc(finalAssoc);
		ostringstream ttrust;
		
		//let us put those data in our density map :
		//decay :
		for(int i=1;i<=sizeMapH;i++)
		{
			for(int j=1;j<=sizeMapW;j++)
			{
				pdDigit.set( i,j, decay * pdDigit(i,j) );
			}
		}
		
		//maximum likelihood digits :
		Mat<float> maxpdDigit(0.0f,sizeMapH,sizeMapW);
		Mat<float> maxDigit(maxpdDigit);
		
		if(trustable)
		{
			for(int i=1;i<=finalAssoc.getLine();i++)
			{
				float probalabel = finalAssoc.get(i,7);
				float label = finalAssoc.get(i,6);
			
				//update :
				Mat<float> pdPosition( pdDigit(1+floor(finalAssoc.get(i,4)/ratio),1+floor(finalAssoc.get(i,3)/ratio) ) );
				pdPosition.set( (probalabel+pdPosition.get(label,1)*2.0f)/2.0f, label,1);
			
				pdDigit.set( 1+floor(finalAssoc.get(i,4)/ratio),1+floor(finalAssoc.get(i,3)/ratio), pdPosition);	
				finalAssoc.afficher();		
			}
			
		
			for(int i=1;i<=sizeMapH;i++)
			{
				for(int j=1;j<=sizeMapW;j++)
				{
					//maxpdDigit.set( 255.0f * sigmoid( max( pdDigit(i,j) )-1.0f ), i,j);
					maxpdDigit.set( 255.0f * tanh( max( pdDigit(i,j) )-1.0f ), i,j);
					maxDigit.set( idmin( (-1.0f)*pdDigit(i,j) ).get(1,1), i,j);
				}
			}
		}
		
		//maxpdDigit.afficher();
		//maxDigit.afficher();
		//afficherMat(std::string("PROBABILITY DENSITY MAP : blobs"), &maxpdDigit, (Mat<float>*)NULL, (Mat<float>*)NULL, true, ratio);
		cv::Mat maxpdDigitCV( Mat2cvp( maxpdDigit, maxpdDigit, maxpdDigit) );
		cv::resize( maxpdDigitCV,maxpdDigitCV, cv::Size(ratio*maxpdDigitCV.cols,ratio*maxpdDigitCV.rows) );
		maxpdDigit = cv2Matp( maxpdDigitCV);
		//so that we can retrieve the precision..?
		
		if(trustable)
		{
		
			for(int i=1;i<=finalAssoc.getLine();i++)
			{
				cv::Point pt(finalAssoc.get(i,3),finalAssoc.get(i,4));
			
				cv::circle(frame,pt,30,cv::Scalar(255,255,0));
				
				
				ostringstream tnumber;
				tnumber << finalAssoc.get(i,6);
				//tnumber << finalAssoc.get(i,6) << ":" << (int)(finalAssoc.get(i,7)*100) << "%";
				      
				int thickness = 3;
				cv::putText(frame, tnumber.str(), pt, cv::FONT_HERSHEY_SIMPLEX, (float)4.0, cv::Scalar(255,0,0), thickness);
				tnumber << ":" << (int)(finalAssoc.get(i,7)*100) << "%";
				cv::putText(maxpdDigitCV, tnumber.str(), pt+cv::Point(-10,20), cv::FONT_HERSHEY_SIMPLEX, (float)0.5f, cv::Scalar(255,255,255), 1);
			}
			
			ttrust << " TRUSTABLE";		      
		}
		else
		{
			int thickness = 3;
			for(int i=1;i<=maxDigit.getLine();i++)
			{
				for(int j=1;j<=maxDigit.getColumn();j++)
				{
					if(maxpdDigit.get(i,j) > tresholdValuable)
					{
						cv::Point pt( j,i);
						
						ostringstream tnumber;
						tnumber << maxDigit.get(i,j) << ":" << (int)(maxpdDigit.get(i,j)*100) << "%";
						cv::putText(maxpdDigitCV, tnumber.str(), pt+cv::Point(-10,20), cv::FONT_HERSHEY_SIMPLEX, (float)0.5f, cv::Scalar(255,255,255), 1);
						
					}
				}
			}
			
			ttrust << " NOT TRUSTABLE";			
		}
		
		
		cv::putText(frame, ttrust.str(), cv::Point(10,40), cv::FONT_HERSHEY_SIMPLEX, (float)2.0, cv::Scalar(0,0,255));
		
		
		cv::imshow("OUTPUT",frame);
		cv::imshow("PROBABILISTIC OUTPUT", maxpdDigitCV);
		
		//from map to matrix enumeration :
		std::vector<int> digitDone;
		std::vector<Mat<float> > identifiedRobots;
		
		for(int i=1;i<=maxDigit.getLine();i++)
		{
			for(int j=1;j<=maxDigit.getColumn();j++)
			{
				if(maxpdDigit.get(i,j) != (float)0)
				{
					int digit = maxpdDigit.get(i,j);
					
					//let us assert that we haven't handle it yet :
					bool done = false;
					for(int k=digitDone.size();k--;)
					{
						if(digitDone[k] == digit)
						{
							done = true;
							break;
						}
					}
					
					if( !done)
					{
						//if it hasn't been handled yet :
						digitDone.push_back(digit);
						
						int nbrOccurrences = 0;
						Mat<float> position(0.0f,2,1);
						
						for(int ii=1;ii<=maxDigit.getLine();ii++)
						{
							for(int jj=1;jj<=maxDigit.getColumn();jj++)
							{
								if( digit == maxDigit.get(ii,jj) )
								{
									nbrOccurrences++;
									
									position.set( position.get(1,1)+ii, 1,1);
									position.set( position.get(2,1)+jj, 2,1);
								} 							
							}
						}
						//let us have the mean of the positions found :
						position *= (1.0f/(float)(nbrOccurrences));
					}
					
				}
			}
		}
		
		if(cv::waitKey(30)>=0)
		{
			continuer = false;
			sensor.setContinuer(false);
		}
     	       
    }
    	
   	if( tSensor.joinable())
   	{
   		tSensor.join();
   	}
   	
	return 0;
}

