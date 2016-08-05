#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <ctype.h>
#include <vector>
#include <map>

#include "./OPU/projectOPU1.h"
#include "./BT/BTHandler.h"
#define BTHANDLER

std::map<int,std::string> digit2addr;

void init()
{
	digit2addr[2] = std::string("00:17:a0:01:81:35");
	digit2addr[3] = std::string("00:17:a0:01:81:a7");
	digit2addr[4] = std::string("00:17:a0:01:7c:64");
}


void sortingLines(Mat<float>& m, int idxColumn)
{
	int nbrdata = m.getLine();
	int nbrdPl = m.getColumn();
	Mat<float> dataOnLine(1,nbrdPl);
	
	for(int i=1;i<=nbrdata;i++)
	{
		float val = m.get(i,idxColumn);
		//let us save that line of data :
		for(int k=1;k<=nbrdPl;k++)
		{
			dataOnLine.set( m.get( i,k), 1, k);
		}		
		
		int j = i;
		//let us move the previous lines over this one or...
		while( j > 1 && m.get(j-1,idxColumn) > val)
		{
			for(int k=1;k<=nbrdPl;k++)
			{
				m.set( m.get( j-1,k), j, k);
			}
			
			j = j-1;
		}
		
		//... let us reassign the saved line :
		for(int k=1;k<=nbrdPl;k++)
		{
			m.set( dataOnLine.get( 1,k), j, k);
		}
	}

}

static void help()
{
    // print a welcome message, and the OpenCV version
    cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo(),\n"
            "Using OpenCV version " << CV_VERSION << endl;
    cout << "\nIt uses camera by default, but you can provide a path to video as an argument.\n";
    cout << "\nHot keys: \n"
            "\tESC - quit the program\n"
            "\tr - auto-initialize tracking\n"
            "\tc - delete all the points\n"
            "\tn - switch the \"night\" mode on/off\n"
            "To add/remove a feature point click it\n" << endl;
}

cv::Point2f point;
bool addRemovePt = false;

static void onMouse( int event, int x, int y, int /*flags*/, void* /*param*/ )
{
    if( event == CV_EVENT_LBUTTONDOWN )
    {
        point = cv::Point2f((float)x, (float)y);
        addRemovePt = true;
    }
}

inline float getElapsedTime(clock_t time)
{
	return (float)( ((float)(clock()-time))/CLOCKS_PER_SEC );
}

int main( int argc, char** argv )
{
    help();
    init();
    
    BTHandler bt;
    
    
    constexpr float cm = 5.369f;     //2.872,3.316  
	constexpr float pi = 3.14159265358979323846264338327950288f;
    
    //Sensor Settings :	
	float treshDist = 300.0f;
	
	bool erosion = false;
	bool dissect = true;
	bool dilate = false;
	
	sensorCAM sensor(treshDist, erosion, dissect,dilate);

    cv::VideoCapture cap;
    cv::TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
    cv::Size subPixWinSize(10,10), winSize(31,31);

    const int MAX_COUNT = 500;
    bool needToInit = false;
    bool nightMode = false;

    if( argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
        cap.open(argc == 2 ? argv[1][0] - '0' : 0);
    else if( argc == 2 )
        cap.open(argv[1]);

    if( !cap.isOpened() )
    {
        cout << "Could not initialize capturing...\n";
        return 0;
    }

    cv::namedWindow( "SENSOR TRACKING", 1 );
    cv::setMouseCallback( "SENSOR TRACKING", onMouse, 0 );

    cv::Mat gray, prevGray, image;
    std::vector<cv::Point2f> points[2];
    std::vector<int> labels;
    std::vector<int> AssociatedLabels;
    
    
    std::thread tSensor( &sensorCAM::runLoop, std::ref(sensor) );
    clock_t time = clock();
    float waitTime = 2.0f;
    
    
    
    //CONTROL LAW :
    bool controlLaw = false;
	int nbrRobots = 0;
	int nbrMaxRobots = 6+1;
	std::vector<float> X1(nbrMaxRobots),Y1(nbrMaxRobots),X2(nbrMaxRobots),Y2(nbrMaxRobots);
	std::vector<float> X(nbrMaxRobots),Y(nbrMaxRobots);
	std::vector<float> r(nbrMaxRobots),ri(nbrMaxRobots);
	std::vector<float> phi(nbrMaxRobots),theta(nbrMaxRobots),THETA(nbrMaxRobots);
	
	std::vector<uchar> status;

	
	//BLUETOOTH :
	int nbrMaxChannels = 10;
	//std::vector<std::string> StringsToSend(nbrMaxChannels);
	//std::map<int,std::string> StringsToSend;
	std::map<std::string,std::string> StringsToSend;
	std::vector<int> channels(nbrMaxRobots);
	for(int i=nbrMaxRobots;i--;)	channels[i] = i+1;
	
	//channels initialization variables : 
	std::vector<bool> channelIsInitialiazed(nbrMaxRobots,false);

	while(1)
    {
        cv::Mat frame;
        cap >> frame;
        if( frame.empty() )
            break;

        frame.copyTo(image);
        cv::cvtColor(image, gray, CV_BGR2GRAY);

        if( nightMode )
            image = cv::Scalar::all(0);

        if( needToInit && getElapsedTime(time) > waitTime )
        {
            // automatic initialization
            /*
            goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, cv::Mat(), 3, 0, 0.04);
            cornerSubPix(gray, points[1], subPixWinSize, cv::Size(-1,-1), termcrit);
            /**/
            
            //bootstrapping with the sensorCAM :
            time = clock();
            sensor << image;
			
			Mat<float> Associations(1,1);
			bool trustable = sensor.getAssoc(Associations);
			
			if(trustable && Associations.getLine() != nbrRobots)
			{
				//let us update the points that we want to track :
				points[0].clear();
				labels.clear();
				for(int i=1;i<=Associations.getLine();i++)
				{
					//points[0].push_back( cv::Point2f( Associations.get(i, 3), Associations.get(i, 4) ) );
					//first point :
					points[0].push_back( cv::Point2f( Associations.get(i, 8), Associations.get(i, 9) ) );
					//second point :
					points[0].push_back( cv::Point2f( Associations.get(i, 10), Associations.get(i, 11) ) );
					
					int label = Associations.get(i,6);
					
					if(label == 0)
					{
						trustable = false;
						Associations = Mat<float>(1,1);
						break;
					}
					
					labels.push_back( label );
					std::cout << " LABELS TO TRACK  i=" << i << " : " << label << std::endl;
				}
				
				nbrRobots = Associations.getLine();
				if( nbrRobots < 2)
				{
					//TODO ???
					controlLaw = false;
				}
				else
				{
					controlLaw = true;
				}
				
			}
			else if(!trustable)
			{
				//we cannot bootstrap yet...
				std::cout << "NOT TRUSTABLE" << std::endl;
				// we trust the tracking.
			}
			
            addRemovePt = false;
        }
        
        if( !points[0].empty() )
        {

            vector<float> err;
            if(prevGray.empty())
                gray.copyTo(prevGray);
            calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                                 3, termcrit, 0, 0.001);
            size_t i, k;
            for( i = k = 0; i < points[1].size(); i++ )
            {
                if( addRemovePt )
                {
                    if( norm(point - points[1][i]) <= 5 )
                    {
                        addRemovePt = false;
                        continue;
                    }
                }

                if( !status[i] )
                {
                	std::cout << "TRACKING : UNSUCCESSFUL : " << i << std::endl;
                	needToInit = true;
                    continue;
                }
                else
                {
                	std::cout << "TRACKING : SUCCESS : " << i << std::endl;
                }

                points[1][k++] = points[1][i];
                cv::circle( image, points[1][i], 3, cv::Scalar(0,255,0), -1, 8);
            }
            points[1].resize(k);
        }

        if( addRemovePt && points[1].size() < (size_t)MAX_COUNT )
        {
            std::vector<cv::Point2f> tmp;
            tmp.push_back(point);
            cornerSubPix( gray, tmp, winSize, cvSize(-1,-1), termcrit);
            points[1].push_back(tmp[0]);
            addRemovePt = false;
        }

        //needToInit = false;
        needToInit = true;
        //TODO : find a better policy to estimate the need of re-initialization...
        

        char c = (char)cv::waitKey(10);
        if( c == 27 )
            break;
        switch( c )
        {
        case 'r':
            needToInit = true;
            break;
        case 'c':
            points[0].clear();
            points[1].clear();
            break;
        case 'n':
            nightMode = !nightMode;
            break;
        }
        
        
        if(controlLaw && nbrRobots > 2)
        {
        	for( int i=0;i < points[1].size(); i++ )
			{
				if( !status[i] )
					continue;

				//points[1][k++] = points[1][i];
				cv::circle( image, points[1][i], 3, cv::Scalar(255,0,0), -1, 8);
				cv::circle( image, points[0][i], 3, cv::Scalar(0,255,0), -1, 8);
				
				for (int t = 0; t < points[1].size(); t++)
				{
					if (status[t])
					{
						//cvLine (image, cvPointFrom32f (points[0][t]), cvPointFrom32f (points[1][t]), CV_RGB (0, 0, 255), 1, CV_AA, 0);
						cv::line (image, points[0][t], points[1][t], CV_RGB (0, 0, 255), 1, CV_AA, 0);
					}
				}
			}


			
			for(int j = 0; j < points[1].size(); j+=2)
			{
				X1[labels[j/2]-1] = (points[1][j].x-400)/cm;
				Y1[labels[j/2]-1] = (-points[1][j].y+300)/cm;
				X2[labels[j/2]-1] = (points[1][j+1].x-400)/cm;
				Y2[labels[j/2]-1] = (-points[1][j+1].y+300)/cm;	//(X1,Y1)ªe-puckÌ¶€C(X2,Y2)ªe-puckÌE€_
				X[labels[j/2]-1] = (X1[labels[j/2]-1]+X2[labels[j/2]-1])/2;
				Y[labels[j/2]-1] = (Y1[labels[j/2]-1]+Y2[labels[j/2]-1])/2;
				theta[labels[j/2]-1] = atan2((Y2[labels[j/2]-1]-Y1[labels[j/2]-1]),(X2[labels[j/2]-1]-X1[labels[j/2]-1]));

				r[labels[j/2]-1] = sqrt(X[labels[j/2]-1]*X[labels[j/2]-1] + Y[labels[j/2]-1]*Y[labels[j/2]-1]);
				ri[labels[j/2]-1] = (int)(r[labels[j/2]-1] + 0.5)+48;

				/*
				if(100<secs && secs<200)
				{
					ri[2] = (int)(r[2] - 20 + 0.5)+48;
					ri[3] = (int)(r[3] - 20 + 0.5)+48;
				}
				*/

				phi[labels[j/2]-1]=atan2(Y[labels[j/2]-1],X[labels[j/2]-1]);
				THETA[labels[j/2]-1] = theta[labels[j/2]-1]-phi[labels[j/2]-1]+pi/2;
			}
			
			//COMPUTE ASSOCIATIONS LABELS :
			AssociatedLabels.clear();
			//AssociatedLabels = std::vector<int>(nbrMaxRobots);
			AssociatedLabels.resize(nbrMaxRobots);
			
			Mat<float> phiLabels(nbrRobots,2);
			for(int i=1;i<=phiLabels.getLine();i++)
			{
				phiLabels.set( phi[ labels[i-1]-1 ], i, 1);
				phiLabels.set( labels[i-1], i, 2);
			}
			//let us sort them by growing order :
			sortingLines( phiLabels, 1);
			//TODO : remove the following line once the sorting algorithm will be implemented...
			for(int i=nbrMaxRobots;i--;)	AssociatedLabels[i] = 2;
			//let us construst the associations :
			for(int i=1;i<=nbrRobots;i++)
			{
				int label = phiLabels.get(i,2);
				int labelnext = phiLabels.get( i+1, 2);
				if( i==nbrRobots)
				{
					labelnext = phiLabels.get(1,2);
				}
				
				if(labelnext <= 0)	labelnext = 2;
				
				AssociatedLabels[ label-1 ] = labelnext;
			}
			
			
			//DATA conversion to STRINGS :
			//#pragma omp parallel for
			for(int i=0;i<nbrRobots;i++)
			{
				std::string data1( std::to_string(ri[labels[i]-1]) + ',');
				std::string data2( std::to_string(phi[labels[i]-1])+',');
				std::string data3( std::to_string(THETA[labels[i]-1])+',');
				std::string data4( std::to_string(phi[ AssociatedLabels[ labels[i]-1 ]-1 ]) );
				
				//StringsToSend[ channels[ labels[i]-1 ] ] = data1 + data2 + data3 + data4;
				StringsToSend[ digit2addr[ labels[i] ] ] = data1 + data2 + data3 + data4;
				
				//std::cout << " MESSAGE : " << StringsToSend[ channels[ labels[i]-1 ] ] << std::endl;
				std::cout << " MESSAGE : " << StringsToSend[ digit2addr[ labels[i] ] ] << std::endl;
				
			}

/*
			//CHANNELS INITIALIZATION :
			for(int i=0;i<nbrRobots;i++)
			{
				if( ! channelIsInitialiazed[ labels[i]-1 ] )
				{
					//let us initialized it :
					if(ERS_Open(channels[ labels[i]-1 ], 5000, 5000 ) == 0)
					{
						//alright, some verifications to do, aren't there ?
						//TODO
						ERS_Config(channels[ labels[i]-1 ],ERS_115200);
						std::string data(std::to_string(nbrRobots));
						
						ERS_Puts(channels[ labels[i]-1 ],data);
						ERS_ClearSend( channels[ labels[i]-1 ] );
		
						channelIsInitialiazed[ labels[i]-1 ] = true;
					}
					else
					{
						std::cout << " ERROR : channels initialization : " << channels[ labels[i] ] << " ; robot id : " << labels[i] << std::endl;
						throw;
					}
				}
			}
			
			//SENDING DATAS :
			//#pragma omp parallel for
			for(int i=0;i<nbrRobots;i++)
			{
				ERS_Puts( channels[ labels[i]-1 ], StringsToSend[ channels[ labels[i]-1 ] ] );
				ERS_ClearSend( channels[ labels[i]-1 ] );
			}
*/
#ifdef BTHANDLER
			for(int i=0;i<nbrRobots;i++)
			{
				if( ! channelIsInitialiazed[ labels[i]-1 ] )
				{
					std::string strnbrRobot(std::to_string(nbrRobots));
					//let us initialized it :
					if(bt.Open(digit2addr[ labels[i] ], &strnbrRobot) == 0)
					{
						//alright, some verifications to do, aren't there ?
						//TODO
						
						//ERS_Config(channels[ labels[i]-1 ],ERS_115200);
						
		
						channelIsInitialiazed[ labels[i]-1 ] = true;
					}
					else
					{
						std::cout << " ERROR : channels initialization : " << digit2addr[ labels[i] ] << " ; robot id : " << labels[i] << std::endl;
						throw;
					}
				}
			}
			
			//SENDING DATAS :
			//#pragma omp parallel for
			for(int i=0;i<nbrRobots;i++)
			{
				
				bt.Send( digit2addr[ labels[i] ], StringsToSend[ digit2addr[ labels[i] ] ] );
				
				//ERS_Puts( channels[ labels[i]-1 ], StringsToSend[ channels[ labels[i]-1 ] ] );
				//ERS_ClearSend( channels[ labels[i]-1 ] );
			}
#endif		
			
			
		}

		imshow("SENSOR TRACKING", image);
        std::swap(points[1], points[0]);
        cv::swap(prevGray, gray);
    }
    
    sensor.setContinuer(false);
    
    if( tSensor.joinable())
   	{
   		tSensor.join();
   	}

    return 0;
}
