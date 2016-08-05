#ifndef PROJECTOPU1_H
#define PROJECTOPU1_H

#define OPENCV_USE
#include "../MAT/Mat.h"
#include <mutex>
#include <thread>
#include "../NN.h"

//#define debug
//#define VERBOSE1


float computeDistance(const cv::Vec3f& c1, const cv::Vec3f& c2);
//int computeAssociationsAndCOG(const cv::Mat& im, Mat<float>& finalAssoc, NN<float>* instanceNN = NULL, float tresholddist = 400.0f);
int computeAssociationsAndCOG(cv::Mat& im, Mat<float>& finalAssoc, NN<float>* instanceNN = NULL, float tresholddist =400.0f, bool erosion = false, bool dissect = false);

int computeAssociationsAndCOGDILATION(cv::Mat& im, Mat<float>& finalAssoc, NN<float>* instanceNN = NULL, float tresholddist =400.0f, bool erosion = false, bool dissect = false, bool dilate = false);
Mat<float> rotate(const Mat<float>& im, float theta);
Mat<float> rotation(float theta);
Mat<float> extractDigitPatch(const Mat<float>& im, const Mat<float>& center, float theta);
cv::Mat rotate(cv::Mat src, double angle, const Mat<float>& point);
cv::Mat extractPatch( cv::Mat src, const Mat<float>& rect);
cv::Mat translate(cv::Mat src, const Mat<float>& point);
cv::Mat scale(cv::Mat src, float factor = 2.0f);
Mat<float> tresholding( const Mat<float>& m, float tresh);

float computeDistance(const cv::Vec3f& c1, const cv::Vec3f& c2)
{
	float r = 0.0f;
	for(int i=0;i<2;i++)
	{
		r+= pow(c1[i]-c2[i], 2);
	}
	
	return sqrt(r);
}

//Error <==> the returned value is odd.
// all okay if the returned value is even <===> the returned value is equal to 2*finalassoc.getLine();
//computes the association : idx1 idx2 and the coordinates x y of the center of gravity...
//int computeAssociationsAndCOG(const cv::Mat& im, Mat<float>& finalAssoc, NN<float>* instanceNN, float tresholddist)
int computeAssociationsAndCOG(cv::Mat& im, Mat<float>& finalAssoc, NN<float>* instanceNN, float tresholddist, bool erosion, bool dissect)
{
	//cv::Mat src = im;
	cv::Mat src;
	im.copyTo(src);
	cv::Mat imOriginal;
	im.copyTo(imOriginal);
//#ifdef debug	
	clock_t time = clock();
//#endif	

	float alpha = 2.0f;
    float beta = -130.0f;
	src.convertTo(src, -1, alpha, beta);
	
	cv::Scalar scalar_low(0,0,0,0);
	cv::Scalar scalar_up(100,100,100,0);
	//cv::Scalar scalar_low(100,100,100,0);
	//cv::Scalar scalar_up(255,255,255,0);
	cv::Mat out(cv::Size(src.rows,src.cols), CV_8UC3);

	cv::inRange(src, scalar_low, scalar_up, src);
				

	/// Reduce the noise so we avoid false circle detection
	//cv::GaussianBlur( src, src, cv::Size(10, 10), 2, 2 );
	cv::GaussianBlur( src, src, cv::Size(9, 9), 2, 2 );
	
	
	try
	{
		cv::cvtColor(src,src, CV_GRAY2BGR);
		cv::cvtColor(src,src, CV_BGR2GRAY);
	}
	catch(cv::Exception e)
	{
		//no problem, it should means that the src image is already GRAY...
	}
	
	vector<cv::Vec3f> circles;

	/// Apply the Hough Transform to find the circles
	//cv::HoughCircles( src, circles, CV_HOUGH_GRADIENT, 1.2, 100);//, 200, 40, 0, 0 );
	//cv::HoughCircles( src, circles, CV_HOUGH_GRADIENT, 1, src.rows/32, 95, 25, 0, 30);
	cv::HoughCircles( src, circles, CV_HOUGH_GRADIENT, 1, src.rows/32, 200, 25, 0, 30);
	//cv::HoughCircles( src, circles, CV_HOUGH_GRADIENT, 1, src.rows/8, 200, 50, 30, 0);
	
	//std::cout << " EXECUTION TIME :: PART FILTER : " << ((float)clock()-time)/CLOCKS_PER_SEC << " seconds." << std::endl;
	
	// Loop over all detected circles and outline them on the original image
 	// by the way, we delete the circles whose radius is not within the range that we are looking for.
 	int radiusMAX = 20;
 	int radiusMIN = 3;
 	if(circles.size())
 	{
	 	for(size_t current_circle = 0; current_circle < circles.size(); ++current_circle) 
	 	{
	 		cv::Point center(std::round(circles[current_circle][0]), std::round(circles[current_circle][1]));
	 		int radius = std::round(circles[current_circle][2]);

#ifdef debug 		
	 		std::cout << "RADIUS : " << radius << std::endl;
	 		//std::cout << " Circle : " << current_circle << " : " << circles[current_circle][0] << " ; " << circles[current_circle][1] << std::endl;
#endif
	 		//if(radius > radiusMAX || radius < radiusMIN)
	 		if( radius < radiusMIN)
	 		{
	 			circles.erase(circles.begin()+current_circle);
	 			--current_circle;
	 		}
	 		else
	 		{
	 			cv::circle(im, center, radius-1, cv::Scalar(100, 100, 100), 2);
	 		}
	 	}
	 }
	
	//cv::namedWindow("IMAGE CIRCLES");
	//cv::imshow("IMAGE CIRCLES", src);
	//afficher("IMAGES CIRCLES", &src, NULL,NULL,true,1.0f);
	
	
	int nbrFeatures = circles.size();
#ifdef VERBOSE1	
	std::cout << " NBR CIRCLES : " << nbrFeatures << std::endl;
#endif

	if( nbrFeatures%2 != 0)
	{
		//then there is a problem : because there ought to be 2 features per robot identifier...
		//return nbrFeatures;
		//TODO : handle that case in the rest of the code 
	}
	if(nbrFeatures == 0)
	{
		//there are currently no robots in the experiment field.
		return 0;
	}
	
	
	//let us compute a distance matrix for each features, in order to know how to associate them correctly :
	Mat<float> dist(nbrFeatures,nbrFeatures);
	for(int i=1;i<=nbrFeatures;i++)
	{
		//let us compute the distances for half of the matrix, since it is symmetric :
		for(int j=1;j<i;j++)
		{
			float distanceij = computeDistance(circles[i-1],circles[j-1]);
			
			if(distanceij > tresholddist)
			{
				distanceij = 2*tresholddist;
			}
			
			dist.set( distanceij, i,j);
			dist.set( distanceij, j,i);
		}
	}
	
	//let us find the minimal value for each column : it is the distance of the closest feature.
	Mat<float> association(0.0f,nbrFeatures,1);
	for(int i=1;i<=nbrFeatures;i++)
	{
		dist.set( 10000, i,i);
		int idxMin = idmin( extract(dist, i,1,i,nbrFeatures) ).get(2,1);
		
		if( dist.get(idxMin,i) <= tresholddist)
		{
			association.set( idxMin, i,1);
		}
		else
		{		
			std::cout << " ERROR : DISTANCE TOO LARGE FOUND BETWEEN TWO DOTS THAT HAVE BEEN ASSOCIATED..." << std::endl;
			dist.afficher();	
			return -2;
		}
	}
	
#ifdef debug
	std::cout << " ASSOC before reaching uniqueness : " << std::endl;
	association.afficher();
	dist.afficher();
#endif	
	//associations are not ready yet, we have to prevent ourselves from the case 
	// where one point is associated to two or more :
	for(int i=1;i<=association.getLine();i++)
	{
		std::vector<int> idxAssoci;
		for(int j=1;j<=association.getLine();j++)
		{
			if(association.get(j,1) == i)
			{
				idxAssoci.push_back(j);
			}
		}
		
		if(idxAssoci.size() > 0)
		{
			//let us found out the closest association :
			int idxmin = 0;
			float distmin = 10000;
			for(int j=0;j<idxAssoci.size();j++)
			{
				if( dist.get(i, idxAssoci[j]) < distmin)
				{
					distmin = dist.get(i, idxAssoci[j]);
					idxmin = idxAssoci[j];
				}
			}
		
			//now that we know the minimal index, we just have to associate our troubling point to a non-existing inde, 0 :
			for(int j=0;j<=idxAssoci.size();j++)
			{
				if( idxAssoci[j] != idxmin)
				{
					association.set( 0.0f, idxAssoci[j], 1);
				}
			}
		}
	}
	
#ifdef debug	
	std::cout << " ASSOC after regularization on uniqueness : " << std::endl;
	association.afficher();
	dist.afficher();
#endif
	
	finalAssoc = Mat<float>(nbrFeatures/2,2);
	int i = 1;
	std::vector<int> alreadyAssociated;
	int idxfeatassoc = 0;
	
	while( idxfeatassoc<nbrFeatures/2)
	{
		int j=1;
		bool discard = false;
		
		while( association.get(j,1) != i)
		{
			j++;
			if(j>nbrFeatures)
			{
#ifdef VERBOSE1
				std::cout << " NO ASSOCIATION found : we discard that feature : idxline = " << i  << std::endl;
#endif				
				//it is the kind of features that have been associated with 0 previously.
				discard = true;
				break;
			}
		}
		
		if( !discard)
		{
			bool alreadyAssociatedorNot = false;
			for(int k=0;k<alreadyAssociated.size();k++)
			{
				if(alreadyAssociated[k] == j)
				{
					alreadyAssociatedorNot = true;
					break;
				}
			}
		
			if( !alreadyAssociatedorNot)
			{
				idxfeatassoc++;
				finalAssoc.set( i, idxfeatassoc,1);
				finalAssoc.set( j, idxfeatassoc,2);
				alreadyAssociated.push_back(i);
				alreadyAssociated.push_back(j);

			}
		}
		
		

		if(j>nbrFeatures)
		{
			//which means it has been discarded :
			//TODO : is there something to do?
		}
		
		
		
		//next item :
		i++;
		if( i>nbrFeatures)
		{
			//and it means that idxfeatassoc has not reached nbrFeatures/2...
			//TODO : figure out the best thing to do...
			finalAssoc = extract( finalAssoc, 1,1, idxfeatassoc, 2);
			nbrFeatures = idxfeatassoc*2;
			break;	
		}
	}
	
	
	//let us compute the rotation angles and the center of gravity :
	Mat<float> pointsPositions(nbrFeatures/2,4);
	Mat<float> thetas(nbrFeatures/2,1);
	Mat<float> CoG(nbrFeatures/2,2);
	for(int i=1;i<=nbrFeatures/2;i++)
	{
		int idxf1 =finalAssoc.get(i,1);
		int idxf2 =finalAssoc.get(i,2);
		
		pointsPositions.set( circles[idxf1-1][0], i, 1);
		pointsPositions.set( circles[idxf1-1][1], i, 2);
		pointsPositions.set( circles[idxf2-1][0], i, 3);
		pointsPositions.set( circles[idxf2-1][1], i, 4);
		
		float x = (circles[idxf1-1][0]+circles[idxf2-1][0]) / 2.0f;
		float y = (circles[idxf1-1][1]+circles[idxf2-1][1]) / 2.0f;
		
		CoG.set( x, i,1);
		CoG.set( y, i,2);
		
		float dx = (circles[idxf1-1][0]-circles[idxf2-1][0]);
		float dy = (circles[idxf1-1][1]-circles[idxf2-1][1]);
		Mat<float> delta(2,1);
		delta.set( dx,1,1);
		delta.set( dy,2,1);
		Mat<float> ortho( rotation(PI/2) * delta);
		
		thetas.set( arctan( ortho.get(2,1), ortho.get(1,1) ), i,1);
		
	}
	

	finalAssoc = operatorL(finalAssoc, CoG);
	finalAssoc = operatorL(finalAssoc, thetas);
	
		
	//--------------------------------------
	//let us identify the numbers :
	Mat<float> number(finalAssoc.getLine(),1);
	Mat<float> confidence(finalAssoc.getLine(),1);
	int size = 28;
	float factor = 2.5f;
	//int coeff = 16;
	int coeff = 3;
	int offsetx = 0;
	int offsety = 0;
	int upsize = size*coeff/factor;
	Mat<float> kernel(1.0f/((float)(coeff*coeff)), coeff/factor,coeff/factor);
	
	cv::Mat rim,sim;
	imOriginal.copyTo(rim);
	imOriginal.copyTo(sim);
	
	for(int i=1;i<=number.getLine();i++)
	{
		Mat<float> point( transpose(extract( finalAssoc, i,3,i,4) ) );
		Mat<float> origin(2,1);
		origin.set( imOriginal.cols/2+offsetx, 1,1);
		origin.set( imOriginal.rows/2+offsety, 2,1);
		
		/*
		clock_t timetrans = clock();
		cv::Mat tim = translate(imOriginal, (-1.0f)*(point-origin));
		std::cout << "EXECUTION TIME : TRANSFORMATION TIME : TRANS :: " << ((float)clock()-timetrans)/CLOCKS_PER_SEC << " seconds." << std::endl;
		cv::Mat rim = rotate( tim, 90+finalAssoc.get(i,5)*180.0f/PI, origin);
		std::cout << "EXECUTION TIME : TRANSFORMATION TIME : TRANS+ROT" << ((float)clock()-timetrans)/CLOCKS_PER_SEC << " seconds." << std::endl;
		cv::Mat sim;
		cv::resize(rim, sim, cv::Size(factor*rim.cols,factor*rim.rows) );
		
		std::cout << "EXECUTION TIME : PART RECOGNITION ONLY TRANSFORMATION TIME : " << ((float)clock()-timetrans)/CLOCKS_PER_SEC << " seconds." << std::endl;
		*/
		
		//clock_t timetrans = clock();
		//cv::Mat tim = translate(imOriginal, (-1.0f)*(point-origin));
		//std::cout << "EXECUTION TIME : TRANSFORMATION TIME : TRANS :: " << ((float)clock()-timetrans)/CLOCKS_PER_SEC << " seconds." << std::endl;
		//cv::Mat rim = rotate( tim, 90+finalAssoc.get(i,5)*180.0f/PI, origin);
		rim = rotate( imOriginal, 90+finalAssoc.get(i,5)*180.0f/PI, point);
		//cv::imshow("OUTPUT TEMP",rim);
		//std::cout << "EXECUTION TIME : TRANSFORMATION TIME : TRANS+ROT" << ((float)clock()-timetrans)/CLOCKS_PER_SEC << " seconds." << std::endl;
		cv::resize(rim, sim, cv::Size(factor*rim.cols,factor*rim.rows) );
		
		//std::cout << "EXECUTION TIME : PART RECOGNITION ONLY TRANSFORMATION TIME : " << ((float)clock()-timetrans)/CLOCKS_PER_SEC << " seconds." << std::endl;
		
		
		
		Mat<float> rect( 2,1);
		rect.set( upsize, 1,1);
		rect.set( upsize, 2,1);		
		
		//In order to look more at the center of the patch, we make a little translation of one pixel in both x and y axises :
		//rect = operatorL(factor*(origin+Mat<float>(2.0f,2,1))-0.5f*rect,rect);
		rect = operatorL(factor*(point+Mat<float>(2.0f,2,1))-0.5f*rect,rect);
		cv::Mat patch;
		
		try
		{
			//patch = extractPatch(rim, rect);
			patch = extractPatch(sim, rect);
			
			//cv::resize(patch,im,cv::Size(4*patch.rows,4*patch.cols) );
			//cv::imshow("SCALED",sim);
			//cv::imshow("PATCH",patch);
			//afficher("PATCH", &patch,NULL,NULL,true,4.0f);
		
		}
		catch(cv::Exception e)
		{
			return 0;
		}
		
		
		//TODO : let us subsample that patch and feed it to the neural nets:
		//EROSION OF THE PATCH :
		if(erosion)
		{
			int erosion_size = 1;
			cv::Mat element = getStructuringElement(cv::MORPH_CROSS,
		          cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		          cv::Point(erosion_size, erosion_size) );
		   	cv::erode(patch,patch,element);
		}
		
		//SUBSAMPLING :
		Mat<float> patchm( cv2Matp<float>(patch) );
		Mat<float> patchmsubconv( subsampled( &(patchm), &kernel) );
		Mat<float> inputIM( tresholding( patchmsubconv, 110.0f) );
		
		inputIM = extract( inputIM, 1,1, size,size);
#ifdef VERBOSE1		
		inputIM.afficher();
#endif
		if(dissect)
		{
			//let us put a column of zeros between the dots and the digit :
			
			//1) let us identify the minimals the maximums that are not equal to the number of line :
			Mat<float> nbrZeros(1,inputIM.getColumn());
			for(int i=1;i<=inputIM.getColumn();i++)
			{
				int nbrzeros = 0;
				for(int j=1;j<=inputIM.getLine();j++)
				{
					if(inputIM.get(j,i) == (float)0)
					{
						nbrzeros++;
					}
				}
				if(nbrzeros == inputIM.getLine())
				{
					nbrzeros = 0;
				}
				nbrZeros.set( nbrzeros, 1,i);
			}
			
			//2) Do the zeroing on numberoftimes columns...
			int numberoftimes = 2;
			std::vector<int> idCol;
			while(numberoftimes--)
			{
				
				//from the beginning of the patch :
				Mat<float> bnbrZeros( extract(nbrZeros, 1,1, 1,nbrZeros.getColumn()/2) );
				int idcol = idmin( (-1.0f)*bnbrZeros).get(2,1);
				idCol.push_back( idcol);
				//make sure that it does not make a cut in the middle of the digit...
				if(idcol < size/2-2 || idcol > size/2+2)
				{
					for(int i=1;i<=inputIM.getLine();i++)
					{
						inputIM.set( 0.0f, i,idcol);
					}
				}
				nbrZeros.set( 0.0f, 1, idcol);
				
				//from the end of the patch :
				Mat<float> enbrZeros( extract(nbrZeros, 1,nbrZeros.getColumn()/2+1, 1,nbrZeros.getColumn() ) );
				idcol = idmin( (-1.0f)*enbrZeros).get(2,1)+nbrZeros.getColumn()/2;
				idCol.push_back( idcol);
				//make sure that it does not make a cut in the middle of the digit...
				if(idcol < size/2-2 || idcol > size/2+2)
				{
					for(int i=1;i<=inputIM.getLine();i++)
					{
						inputIM.set( 0.0f, i,idcol);
					}
				}
				nbrZeros.set( 0.0f, 1, idcol+nbrZeros.getColumn()/2);
				
			}
			
#ifdef VERBOSE1			
			std::cout << " >> AFTER DISSECTION : idx = ";
			for(int i=0;i<idCol.size();i++)
			{
				std::cout  << idCol[i] << " " ;
			}
			std::cout << std::endl;
			inputIM.afficher();
#endif	
			
		}
		
		int w = inputIM.getLine();
		int h = inputIM.getColumn();
		Mat<float> input(w*h,1);
		
		//std::cout << " W x H = " << w << " x " << h << std::endl;
		
		//unrolling the inputIM into a vector :
		for(int i=1;i<=w;i++)
		{
			for(int j=1;j<=h;j++)
			{
				input.set( inputIM.get(i,j), (i-1)*h+j, 1);
			}
		}
		
		//let us evaluate it :
		Mat<float> output( instanceNN->feedForward(input) );
#ifdef debug		
		transpose(output).afficher();
#endif		
		int label = idmin( (-1.0f)*output ).get(1,1);
		std::cout << " LABELED :: " << label-1 << " CONFIDENCE : " << output.get(label,1)*100 << " %." << std::endl;
#ifdef debug		
		number.afficher();
#endif		
		number.set( label-1, i,1);
		confidence.set( output.get(label,1), i,1);
	}
	
	finalAssoc = operatorL(finalAssoc, number);
	finalAssoc = operatorL(finalAssoc, confidence);
	
	finalAssoc = operatorL(finalAssoc, pointsPositions);
	
	
	
//#ifdef debug
	std::cout << "EXECUTION TIME :: " << ((float)clock()-time)/CLOCKS_PER_SEC << " seconds." << std::endl;
//#endif

	return nbrFeatures;
	
}

//rotation counterclockwise :
Mat<float> rotation(float theta)
{
	Mat<float> r(2,2);
	r.set( cos(theta), 1,1);
	r.set( cos(theta),2,2);
	r.set( sin(theta), 1,2);
	r.set( -sin(theta),2,1);
	
	return r;
	
}


//add 4 lines and 4 columns...
Mat<float> rotate(const Mat<float>& im, float theta)
{
	Mat<float> rot( rotation(theta) );
	
	int h = im.getLine();
	int w = im.getColumn();
	float ox = ((float)w)/2.0f;
	float oy = ((float)h)/2.0f;
	
	Mat<float> rim(0.0f, h+4,w+4);
	
	for(int i=1;i<=h;i++)
	{
		for(int j=1;j<=w;j++)
		{
			float x = j-ox;
			float y = i-oy;
			Mat<float> coord(2,1);
			coord.set( x, 1,1);
			coord.set( y, 2,1);
			
			//new coordinate :
			coord = rot*coord;
			
			rim.set( im.get(i,j), oy+3+floor(coord.get(2,1)), ox+2+floor(coord.get(1,1)) );
		}
	}
	
	return rim;
}


cv::Mat rotate(cv::Mat src, double angle, const Mat<float>& point)
{
    cv::Mat dst;
    cv::Point2f pt( point.get(1,1), point.get(2,1));    
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
    cv::warpAffine(src, dst, r, cv::Size(src.cols, src.rows));
    return dst;
}

cv::Mat translate(cv::Mat src, const Mat<float>& point)
{
    cv::Mat dst;
    cv::Point2f pt( point.get(1,1), point.get(2,1));    
	cv::Mat r = (cv::Mat_<double>(2,3) << 1, 0, point.get(1,1), 0, 1, point.get(2,1));
    
    
    cv::warpAffine(src, dst, r, cv::Size(src.cols, src.rows));
    return dst;
}

cv::Mat scale(cv::Mat src, float factor)
{
    cv::Mat dst;
	cv::Mat r = (cv::Mat_<double>(2,3) << 1, 0, 0, 0, 1, 0);
    
    
    cv::warpAffine(src, dst, r, cv::Size(src.cols*factor, src.rows*factor));
    return dst;
}

cv::Mat extractPatch( cv::Mat src, const Mat<float>& rect)
{
	//cv::Rect gate = cv::roi ( src, rect.get(1,1), rect.get(2,1), rect.get(1,2), rect.get(2,1) );
    
    //return cv::Mat ( src, gate ); 
    return cv::Mat ( src, cv::Rect(rect.get(1,1), rect.get(2,1), rect.get(1,2), rect.get(2,2))  ); 
}

Mat<float> tresholding( const Mat<float>& m, float tresh)
{
	Mat<float> r(0.0f, m.getLine(), m.getColumn());
	
	for(int i=1;i<=r.getLine();i++)
	{
		for(int j=1;j<=r.getColumn();j++)
		{
			if( m.get(i,j) >= tresh )
			{
				r.set( 1.0f, i,j);
			}
		}
	}
	
	return r;
}
/*
Mat<float> extractDigitPatch(const Mat<float>& im, const Mat<float>& center, float theta)
{
	Mat<float> patch(28,28);
	Mat<float> coordx(41,41);
	Mat<float> coordy(41,41);
	float ox = 20;
	float oy = 20;
	
	for(int i=1;i<=coordx.getLine();i++)
	{
		for(int j=1;j<=coordx.getColumn();j++)
		{
			coordx.set( i-ox, i,j);
			coordy.set( j-oy, i,j);
		}
	}
	
	
	std::cout << " COORD : " << std::endl;
	coordx.afficher();
	coordy.afficher();
	
	
	Mat<float> rotcoordx = rotate(coordx,theta);
	Mat<float> rotcoordy = rotate(coordy,theta);//+4 line columns...
	
	std::cout << "ROTCOORD : " << std::endl;
	rotcoordx.afficher();
	rotcoordy.afficher();
	
	Mat<float> seekx(0.0f,28,28);
	Mat<float> seeky(0.0f,28,28);
	float cx = 14;
	float cy = 14;
	
	
	for(int i=1;i<=rotcoordx.getLine();i++)
	{
		for(int j=1;j<=rotcoordx.getColumn();j++)
		{
			//seekx.set( i-cx, ox+rotcoordx.get(i,j), oy+rotcoordy.get(i,j));
			//seeky.set( j-cy-13, ox+rotcoordx.get(i,j), oy+rotcoordy.get(i,j));
			seekx.set( ox+rotcoordx.get(i,j), i,j);
			seeky.set( ox+rotcoordy.get(i,j), i,j);
		}
	}
	
	
	int offl = (rotcoordx.getLine()-patch.getLine()+1)/2.0f;
	int offc = (rotcoordx.getColumn()-patch.getColumn()+1)/2.0f;
	seekx = extract( rotcoordx, offl, offc, offl+seekx.getLine()-1, offc+seekx.getColumn()-1); 
	seeky = extract( rotcoordy, offl, offc, offl+seekx.getLine()-1, offc+seekx.getColumn()-1);
	
	std::cout << " SEEK : " << std::endl;
	seekx.afficher();
	seeky.afficher();
	
	cx = center.get(1,1);
	cy = center.get(2,1);
	for(int i=1;i<=seekx.getLine();i++)
	{
		for(int j=1;j<=seekx.getColumn();j++)
		{
			//patch.set( im.get( cx+seekx.get(i,j), cy+seeky.get(i,j) ), i,j);
			patch.set( im.get( cx+rotcoordx.get(i+2,j+2), cy+rotcoordy.get(i+2,j+2) ), i,j);
		}
	}
	
	return patch;
}
*/

std::mutex mutexRes;

class sensorCAM
{
	public :
	
	sensorCAM(bool& nti,float tresh_ = 500.0f, bool erosion_ = false, bool dissect_ = false, bool dilate_ = false) : needToInit(nti), continuer(true), trustable(false), assoc(Mat<float>(1,1)), tresholddist(tresh_), erosion(erosion), dissect(dissect_), dilate(dilate_)
	{
		//std::string filepath = "neuralnetworksDIGITROTATEDPI";
		
		//BEST VERSION SO FAR :
		//std::string filepath = "neuralnetworksDIGITROTATEDPI2";
		
		//std::string filepath = "neuralnetworksDIGITROTATEDPIDIGIT";
		std::string filepath = "neuralnetworksDIGITROTATEDPIDIGITSTRAINER1";
		//std::string filepath = "neuralnetworksDIGITROTATEDPI3";
		
		nn = new NN<float>(filepath);
		nn->learning = false;
		
		nbrdatas = 0;
	}
	
	~sensorCAM()
	{
		delete nn;
	}
	
	void runLoop()
	{
		mutexRes.lock();
		while(continuer)
		{
			mutexRes.unlock();
			
			mutexRes.lock();
			if(this->ims.size() > 0)
			{
				mutexRes.unlock();
				
				mutexRes.lock();
				this->currentIm = this->ims[this->ims.size()-1];
				this->ims.clear();
				mutexRes.unlock();
				
				//let us deal with the current ims :
				std::cout << " SENSOR CAM  :: " ;
				std::cout << " in progress ... " << std::endl;
				int res = this->wrapper(currentIm);
				
				this->trustableIm = this->currentIm;
				
				if( res < 0)
				{
					mutexRes.lock();
					this->trustable = false;
					mutexRes.unlock();
				}
				else
				{
					mutexRes.lock();
					this->trustable = true;
					
					if(nbrdatas != this->assoc.getLine() )
					{
						this->needToInit = true;
						this->nbrdatas = this->assoc.getLine();
					}
					
					//this->trustableIm = this->currentIm;
					mutexRes.unlock();
				}
				
				mutexRes.lock();
			}
			mutexRes.unlock();
			
			
			mutexRes.lock();
		}
		mutexRes.unlock();
	}
	
	void operator<<(const cv::Mat& im)
	{
		mutexRes.lock();
		this->ims.push_back(im);
		#ifdef debug
		std::cout << " SENSOR CAM  :: " ;
		std::cout << " NEW FRAME. " << std::endl;
		#endif
		while(ims.size() >2)
		{
			ims.erase(ims.begin());
		}
		mutexRes.unlock();
	}
	
	bool getAssoc(Mat<float>& r)	const
	{
		mutexRes.lock();
		//if(mutexRes.try_lock())
		//{
			if(this->trustable)
			{
				
				r = this->assoc;
				mutexRes.unlock();
				
				if(this->trustableIm.rows > 0)
				{
					//cv::imshow("CIRCLES",this->trustableIm);
				}
				
				return true;
			}
			mutexRes.unlock();
		//}
		
		if(this->trustableIm.rows > 0)
		{
			//cv::imshow("CIRCLES",this->trustableIm);
		}
		
		return false;
	}
	
	void setContinuer(bool cont)
	{
		mutexRes.lock();
		this->continuer = cont;
		mutexRes.unlock();
	}
	
	int wrapper(cv::Mat& im)
	{
		int res = (dilate ? 
		computeAssociationsAndCOGDILATION( im, this->assoc, this->nn, this->tresholddist, this->erosion, this->dissect, this->dilate)
		:
		computeAssociationsAndCOG( im, this->assoc, this->nn, this->tresholddist, this->erosion, this->dissect) 
					
				);
		return res;
	}
	
	private :
	
	bool continuer;
	std::vector<cv::Mat> ims;
	cv::Mat currentIm;
	cv::Mat trustableIm;
	Mat<float> assoc;
	
	NN<float>* nn;
	bool& needToInit;
	int nbrdatas;
	bool trustable;
	float tresholddist;
	bool erosion;
	bool dissect;
	bool dilate;
};


//Error <==> the returned value is odd.
// all okay if the returned value is even <===> the returned value is equal to 2*finalassoc.getLine();
//computes the association : idx1 idx2 and the coordinates x y of the center of gravity...
//int computeAssociationsAndCOG(const cv::Mat& im, Mat<float>& finalAssoc, NN<float>* instanceNN, float tresholddist)
int computeAssociationsAndCOGDILATION(cv::Mat& im, Mat<float>& finalAssoc, NN<float>* instanceNN, float tresholddist, bool erosion, bool dissect, bool dilate)
{
	cv::Mat src = im;
	cv::Mat imOriginal;
	im.copyTo(imOriginal);
//#ifdef debug	
	clock_t time = clock();
//#endif	

	float alpha = 2.0f;
    float beta = -110.0f;
	src.convertTo(src, -1, alpha, beta);
	
	cv::Scalar scalar_low(0,0,0,0);
	cv::Scalar scalar_up(100,100,100,0);
	//cv::Scalar scalar_low(100,100,100,0);
	//cv::Scalar scalar_up(255,255,255,0);
	cv::Mat out(cv::Size(src.rows,src.cols), CV_8UC3);

	cv::inRange(src, scalar_low, scalar_up, src);
				

	/// Reduce the noise so we avoid false circle detection
	cv::GaussianBlur( src, src, cv::Size(9, 9), 2, 2 );
	//cv::GaussianBlur( src, src, cv::Size(9, 9), 2, 2 );
	
	try
	{
		cv::cvtColor(src,src, CV_GRAY2BGR);
		cv::cvtColor(src,src, CV_BGR2GRAY);
	}
	catch(cv::Exception e)
	{
		//no problem, it should means that the src image is already GRAY...
	}
	
	vector<cv::Vec3f> circles;

	/// Apply the Hough Transform to find the circles
	//cv::HoughCircles( src, circles, CV_HOUGH_GRADIENT, 1.2, 100);//, 200, 40, 0, 0 );
	cv::HoughCircles( src, circles, CV_HOUGH_GRADIENT, 1, src.rows/32, 90, 23, 0, 30);
	//cv::HoughCircles( src, circles, CV_HOUGH_GRADIENT, 1, src.rows/8, 200, 50, 30, 0);
	
	// Loop over all detected circles and outline them on the original image
 	// by the way, we delete the circles whose radius is not within the range that we are looking for.
 	int radiusMAX = 10;
 	int radiusMIN = 4;
 	if(circles.size())
 	{
	 	for(size_t current_circle = 0; current_circle < circles.size(); ++current_circle) 
	 	{
	 		cv::Point center(std::round(circles[current_circle][0]), std::round(circles[current_circle][1]));
	 		int radius = std::round(circles[current_circle][2]);
	 		
	 		//std::cout << "RADIUS : " << radius << std::endl;
	 		//std::cout << " Circle : " << current_circle << " : " << circles[current_circle][0] << " ; " << circles[current_circle][1] << std::endl;
	 		//if(radius > radiusMAX || radius < radiusMIN)
	 		if( radius < radiusMIN)
	 		{
	 			circles.erase(circles.begin()+current_circle);
	 			--current_circle;
	 		}
	 		else
	 		{
	 			cv::circle(im, center, radius-1, cv::Scalar(100, 100, 100), 2);
	 		}
	 	}
	 }
	
	//cv::imshow("IMAGE CIRCLES", im);
	//afficher("IMAGES CIRCLES", &src, NULL,NULL,true,1.0f);
	
	
	int nbrFeatures = circles.size();
	std::cout << " NBR CIRCLES : " << nbrFeatures << std::endl;
	
	if( nbrFeatures%2 != 0)
	{
		//then there is a problem : because there ought to be 2 features per robot identifier...
		//return nbrFeatures;
		//TODO : handle that case in the rest of the code 
	}
	if(nbrFeatures == 0)
	{
		//there are currently no robots in the experiment field.
		return 0;
	}
	
	
	//let us compute a distance matrix for each features, in order to know how to associate them correctly :
	Mat<float> dist(nbrFeatures,nbrFeatures);
	for(int i=1;i<=nbrFeatures;i++)
	{
		//let us compute the distances for half of the matrix, since it is symmetric :
		for(int j=1;j<i;j++)
		{
			float distanceij = computeDistance(circles[i-1],circles[j-1]);
			
			if(distanceij > tresholddist)
			{
				distanceij = 2*tresholddist;
			}
			
			dist.set( distanceij, i,j);
			dist.set( distanceij, j,i);
		}
	}
	
	//let us find the minimal value for each column : it is the distance of the closest feature.
	Mat<float> association(0.0f,nbrFeatures,1);
	for(int i=1;i<=nbrFeatures;i++)
	{
		dist.set( 10000, i,i);
		int idxMin = idmin( extract(dist, i,1,i,nbrFeatures) ).get(2,1);
		
		if( dist.get(idxMin,i) <= tresholddist)
		{
			association.set( idxMin, i,1);
		}
		else
		{
			std::cout << " ERROR : DISTANCE TOO LARGE FOUND BETWEEN TWO DOTS THAT HAVE BEEN ASSOCIATED..." << std::endl;
			dist.afficher();
			return -2;
		}
	}
	
	//std::cout << " ASSOC before reaching uniqueness : " << std::endl;
	//association.afficher();
	//dist.afficher();
	//associations are not ready yet, we have to prevent ourselves from the case 
	// where one point is associated to two or more :
	for(int i=1;i<=association.getLine();i++)
	{
		std::vector<int> idxAssoci;
		for(int j=1;j<=association.getLine();j++)
		{
			if(association.get(j,1) == i)
			{
				idxAssoci.push_back(j);
			}
		}
		
		if(idxAssoci.size() > 0)
		{
			//let us found out the closest association :
			int idxmin = 0;
			float distmin = 10000;
			for(int j=0;j<idxAssoci.size();j++)
			{
				if( dist.get(i, idxAssoci[j]) < distmin)
				{
					distmin = dist.get(i, idxAssoci[j]);
					idxmin = idxAssoci[j];
				}
			}
		
			//now that we know the minimal index, we just have to associate our troubling point to a non-existing inde, 0 :
			for(int j=0;j<=idxAssoci.size();j++)
			{
				if( idxAssoci[j] != idxmin)
				{
					association.set( 0.0f, idxAssoci[j], 1);
				}
			}
		}
	}
	
	//std::cout << " ASSOC after regularization on uniqueness : " << std::endl;
	//association.afficher();
	//dist.afficher();
	
	
	finalAssoc = Mat<float>(nbrFeatures/2,2);
	int i = 1;
	std::vector<int> alreadyAssociated;
	int idxfeatassoc = 0;
	
	while( idxfeatassoc<nbrFeatures/2)
	{
		int j=1;
		bool discard = false;
		
		while( association.get(j,1) != i)
		{
			j++;
			if(j>nbrFeatures)
			{
				std::cout << " NO ASSOCIATION found : we discard that feature : idxline = " << i  << std::endl;
				//it is the kind of features that have been associated with 0 previously.
				discard = true;
				break;
			}
		}
		
		if( !discard)
		{
			bool alreadyAssociatedorNot = false;
			for(int k=0;k<alreadyAssociated.size();k++)
			{
				if(alreadyAssociated[k] == j)
				{
					alreadyAssociatedorNot = true;
					break;
				}
			}
		
			if( !alreadyAssociatedorNot)
			{
				idxfeatassoc++;
				finalAssoc.set( i, idxfeatassoc,1);
				finalAssoc.set( j, idxfeatassoc,2);
				alreadyAssociated.push_back(i);
				alreadyAssociated.push_back(j);

			}
		}
		
		

		if(j>nbrFeatures)
		{
			//which means it has been discarded :
			//TODO : is there something to do?
		}
		
		
		
		//next item :
		i++;
		if( i>nbrFeatures)
		{
			//and it means that idxfeatassoc has not reached nbrFeatures/2...
			//TODO : figure out the best thing to do...
			finalAssoc = extract( finalAssoc, 1,1, idxfeatassoc, 2);
			nbrFeatures = idxfeatassoc*2;
			break;	
		}
	}
	
	
	//let us compute the rotation angles and the center of gravity :
	Mat<float> pointsPositions(nbrFeatures/2,4);
	Mat<float> thetas(nbrFeatures/2,1);
	Mat<float> CoG(nbrFeatures/2,2);
	for(int i=1;i<=nbrFeatures/2;i++)
	{
		int idxf1 =finalAssoc.get(i,1);
		int idxf2 =finalAssoc.get(i,2);
		
		
		pointsPositions.set( circles[idxf1-1][0], i, 1);
		pointsPositions.set( circles[idxf1-1][1], i, 2);
		pointsPositions.set( circles[idxf2-1][0], i, 3);
		pointsPositions.set( circles[idxf2-1][1], i, 4);
		
		
		float x = (circles[idxf1-1][0]+circles[idxf2-1][0]) / 2.0f;
		float y = (circles[idxf1-1][1]+circles[idxf2-1][1]) / 2.0f;
		
		CoG.set( x, i,1);
		CoG.set( y, i,2);
		
		float dx = (circles[idxf1-1][0]-circles[idxf2-1][0]);
		float dy = (circles[idxf1-1][1]-circles[idxf2-1][1]);
		Mat<float> delta(2,1);
		delta.set( dx,1,1);
		delta.set( dy,2,1);
		Mat<float> ortho( rotation(PI/2) * delta);
		
		thetas.set( arctan( ortho.get(2,1), ortho.get(1,1) ), i,1);
		
	}
	

	finalAssoc = operatorL(finalAssoc, CoG);
	finalAssoc = operatorL(finalAssoc, thetas);
	
	//--------------------------------------
	//let us identify the numbers :
	Mat<float> number(finalAssoc.getLine(),1);
	Mat<float> confidence(finalAssoc.getLine(),1);
	int size = 28;
	float factor = 2.0f;
	//int coeff = 16;
	int coeff = 2;
	int upsize = size*coeff/factor;
	Mat<float> kernel(1.0f/((float)(coeff*coeff)), coeff/factor,coeff/factor);
	
	for(int i=1;i<=number.getLine();i++)
	{
		Mat<float> point( transpose(extract( finalAssoc, i,3,i,4) ) );
		Mat<float> origin(2,1);
		origin.set( imOriginal.cols/2, 1,1);
		origin.set( imOriginal.rows/2, 2,1);
		
		cv::Mat tim = translate(imOriginal, (-1.0f)*(point-origin));
		cv::Mat rim = rotate( tim, 90+finalAssoc.get(i,5)*180.0f/PI, origin);
		cv::Mat sim;
		cv::resize(rim, sim, cv::Size(factor*rim.cols,factor*rim.rows) );
		
		Mat<float> rect( 2,1);
		rect.set( upsize, 1,1);
		rect.set( upsize, 2,1);		
		
		//In order to look more at the center of the patch, we make a little translation of one pixel in both x and y axises :
		rect = operatorL(factor*(origin+Mat<float>(1.0f,2,1))-0.5f*rect,rect);
		cv::Mat patch;
		
		try
		{
			//patch = extractPatch(rim, rect);
			patch = extractPatch(sim, rect);
			
			//cv::resize(patch,im,cv::Size(4*patch.rows,4*patch.cols) );
			//cv::imshow("SCALED",sim);
			//cv::imshow("PATCH",patch);
			//afficher("PATCH", &patch,NULL,NULL,true,4.0f);
		
		}
		catch(cv::Exception e)
		{
			return 0;
		}
		
		
		//TODO : let us subsample that patch and feed it to the neural nets:
		//EROSION OF THE PATCH :
		if(erosion)
		{
			int erosion_size = 1;
			cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE,
		          cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		          cv::Point(erosion_size, erosion_size) );
		   	cv::erode(patch,patch,element);
		}
		
		//DILATATION OF THE PATCH :
		if(dilate)
		{
			int dilate_size = 1;
			cv::Mat element = getStructuringElement(cv::MORPH_CROSS,
		          cv::Size(2 * dilate_size + 1, 2 * dilate_size + 1),
		          cv::Point(dilate_size, dilate_size) );
		   	cv::dilate(patch,patch,element);
		}
		
		//SUBSAMPLING :
		Mat<float> patchm( cv2Matp<float>(patch) );
		Mat<float> patchmsubconv( subsampled( &(patchm), &kernel) );
		Mat<float> inputIM( tresholding( patchmsubconv, 110.0f) );
		
		inputIM = extract( inputIM, 1,1, size,size);
		//inputIM.afficher();
		
		if(dissect)
		{
			//let us put a column of zeros between the dots and the digit :
			
			//1) let us identify the minimals the maximums that are not equal to the number of line :
			Mat<float> nbrZeros(1,inputIM.getColumn());
			for(int i=1;i<=inputIM.getColumn();i++)
			{
				int nbrzeros = 0;
				for(int j=1;j<=inputIM.getLine();j++)
				{
					if(inputIM.get(j,i) == (float)0)
					{
						nbrzeros++;
					}
				}
				if(nbrzeros == inputIM.getLine())
				{
					nbrzeros = 0;
				}
				nbrZeros.set( nbrzeros, 1,i);
			}
			
			//2) Do the zeroing on numberoftimes columns...
			int numberoftimes = 2;
			std::vector<int> idCol;
			while(numberoftimes--)
			{
				
				//from the beginning of the patch :
				Mat<float> bnbrZeros( extract(nbrZeros, 1,1, 1,nbrZeros.getColumn()/2) );
				int idcol = idmin( (-1.0f)*bnbrZeros).get(2,1);
				idCol.push_back( idcol);
				//make sure that it does not make a cut in the middle of the digit...
				if(idcol < size/2-2 || idcol > size/2+2)
				{
					for(int i=1;i<=inputIM.getLine();i++)
					{
						inputIM.set( 0.0f, i,idcol);
					}
				}
				nbrZeros.set( 0.0f, 1, idcol);
				
				//from the end of the patch :
				Mat<float> enbrZeros( extract(nbrZeros, 1,nbrZeros.getColumn()/2+1, 1,nbrZeros.getColumn() ) );
				idcol = idmin( (-1.0f)*enbrZeros).get(2,1)+nbrZeros.getColumn()/2;
				idCol.push_back( idcol);
				//make sure that it does not make a cut in the middle of the digit...
				if(idcol < size/2-2 || idcol > size/2+2)
				{
					for(int i=1;i<=inputIM.getLine();i++)
					{
						inputIM.set( 0.0f, i,idcol);
					}
				}
				nbrZeros.set( 0.0f, 1, idcol+nbrZeros.getColumn()/2);
				
			}
			
			
			//std::cout << " >> AFTER DISSECTION : idx = ";
			for(int i=0;i<idCol.size();i++)
			{
				std::cout  << idCol[i] << " " ;
			}
			std::cout << std::endl;
			
			inputIM.afficher();
		}
		
		int w = inputIM.getLine();
		int h = inputIM.getColumn();
		Mat<float> input(w*h,1);
		
		//std::cout << " W x H = " << w << " x " << h << std::endl;
		
		//unrolling the inputIM into a vector :
		for(int i=1;i<=w;i++)
		{
			for(int j=1;j<=h;j++)
			{
				input.set( inputIM.get(i,j), (i-1)*h+j, 1);
			}
		}
		
		//let us evaluate it :
		Mat<float> output( instanceNN->feedForward(input) );
#ifdef debug		
		transpose(output).afficher();
#endif		
		int label = idmin( (-1.0f)*output ).get(1,1);
		std::cout << " LABELED :: " << label-1 << " CONFIDENCE : " << output.get(label,1)*100 << " %." << std::endl;
		number.afficher();
		number.set( label-1, i,1);
		confidence.set( output.get(label,1), i,1);
	}
	
	finalAssoc = operatorL(finalAssoc, number);
	finalAssoc = operatorL(finalAssoc, confidence);
	
	//nbrfeatures x 7+4
	finalAssoc = operatorL(finalAssoc, pointsPositions);
	
	
	
//#ifdef debug
	std::cout << "EXECUTION TIME : " << ((float)clock()-time)/CLOCKS_PER_SEC << " seconds." << std::endl;
//#endif

	return nbrFeatures;
	
}

/*
class FlowPrediction
{
	public :
	
	FlowPrediction(

};
*/


#endif
