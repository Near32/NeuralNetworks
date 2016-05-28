#ifndef PROJECTOPU1_H
#define PROJECTOPU1_H

#define OPENCV_USE
#include "../../MAT/Mat.h"


float computeDistance(const cv::Vec3f& c1, const cv::Vec3f& c2);
int computeAssociationsAndCOG(const cv::Mat& im, Mat<float>& finalAssoc, NN<float>* instanceNN = NULL);
Mat<float> rotate(const Mat<float>& im, float theta);
Mat<float> rotation(float theta);
Mat<float> extractDigitPatch(const Mat<float>& im, const Mat<float>& center, float theta);
cv::Mat rotate(cv::Mat src, double angle, const Mat<float>& point);
cv::Mat extractPatch( cv::Mat src, const Mat<float>& rect);
cv::Mat translate(cv::Mat src, const Mat<float>& point);
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
int computeAssociationsAndCOG(const cv::Mat& im, Mat<float>& finalAssoc, NN<float>* instanceNN)
{
	cv::Mat src;
	//src is a picture on which we have reduiced the area in which to look for :
	/// Reduce the noise so we avoid false circle detection
	cv::GaussianBlur( im, src, cv::Size(9, 9), 2, 2 );
	//cv::GaussianBlur( src, src, cv::Size(9, 9), 2, 2 );
	cv::cvtColor(src,src, CV_BGR2GRAY);
	
	vector<cv::Vec3f> circles;

	/// Apply the Hough Transform to find the circles
	//cv::HoughCircles( src, circles, CV_HOUGH_GRADIENT, 1.2, 100);//, 200, 40, 0, 0 );
	cv::HoughCircles( src, circles, CV_HOUGH_GRADIENT, 1, src.rows/8, 100, 30, 0, 0);
	// Loop over all detected circles and outline them on the original image
 	if(circles.size())
 	{
	 	for(size_t current_circle = 0; current_circle < circles.size(); ++current_circle) 
	 	{
	 		cv::Point center(std::round(circles[current_circle][0]), std::round(circles[current_circle][1]));
	 		int radius = std::round(circles[current_circle][2]);
	 
	 		cv::circle(src, center, radius, cv::Scalar(0, 255, 0), 5);
	 	}
	 }
	
	cv::imshow("IMAGE CIRCLES", src);
	
	int nbrFeatures = circles.size();
	std::cout << " NBR CIRCLES : " << nbrFeatures << std::endl;
	if( nbrFeatures%2 != 0)
	{
		//then there is a problem : let us define what to do...
		return nbrFeatures;
	}
	if(nbrFeatures == 0)
	{
		return 0;
	}
	
	
	Mat<float> dist(nbrFeatures,nbrFeatures);
	for(int i=1;i<=nbrFeatures;i++)
	{
		//let us compute the distances for half of the matrix, since it is symmetric :
		for(int j=1;j<=i;j++)
		{
			float distanceij = computeDistance(circles[i],circles[j]);
			dist.set( distanceij, i,j);
			dist.set( distanceij, j,i);
		}
	}
	
	//let us find the minimal value for each column :
	Mat<float> association(0.0f,nbrFeatures,1);
	float tresholddist = 300.0f;
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
			std::cout << "DISTANCE TOO LARGE FOUND BETWEEN TWO DOTS THAT HAVE BEEN ASSOCIATED..." << std::endl;
			dist.afficher();
			return -2;
		}
	}
	
	//associations are ready.
	finalAssoc = Mat<float>(nbrFeatures/2,2);
	int i = 1;
	while( i<=nbrFeatures/2)
	{
		int j=1;
		while( association.get(j,1) != i)
		{
			j++;
			if(j>nbrFeatures)
			{
				std::cout << " NO ASSOCIATION... : association = " << std::endl;
				association.afficher();
				dist.afficher();
				return -1;
			}
		}
		
		finalAssoc.set( i, i,1);
		finalAssoc.set( j, i,2);
		
		i++;
	}
	
	
	//let us compute the rotation angles and the center of gravity :
	Mat<float> thetas(nbrFeatures/2,1);
	Mat<float> CoG(nbrFeatures/2,2);
	for(int i=1;i<=nbrFeatures/2;i++)
	{
		int idxf1 =finalAssoc.get(i,1);
		int idxf2 =finalAssoc.get(i,2);
		
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
	int size = 28;
	int coeff = 2;
	int upsize = size*coeff;
	Mat<float> kernel(1.0f/((float)(coeff*coeff)), coeff,coeff);
	
	for(int i=1;i<=number.getLine();i++)
	{
		Mat<float> point( transpose(extract( finalAssoc, i,3,i,4) ) );
		Mat<float> origin(2,1);
		origin.set( im.cols/2, 1,1);
		origin.set( im.rows/2, 2,1);
		
		cv::Mat tim = translate(im, (-1.0f)*(point-origin));
		cv::Mat rim = rotate( tim, 90+finalAssoc.get(i,5)*180.0f/PI, origin);
		
		Mat<float> rect( 2,1);
		rect.set( upsize, 1,1);
		rect.set( upsize, 2,1);		
		rect = operatorL(origin-0.5f*rect,rect);
		cv::Mat patch;
		
		try
		{
		patch = extractPatch(rim, rect);

		cv::imshow("PATCH",patch);
		}
		catch(cv::Exception e)
		{
			return 0;
		}
		
		
		//TODO : let us subsample that patch and feed it to the neural nets:
		Mat<float> patchm( cv2Matp<float>(patch) );

		Mat<float> inputIM( tresholding( subsampled( &(patchm), &kernel), 110.0f) );
		
		inputIM = extract( inputIM, 1,1, size,size);
		inputIM.afficher();
		
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
		transpose(output).afficher();
		int label = idmin( (-1.0f)*output ).get(1,1);
		std::cout << " LABELED :: " << label << std::endl;
	}
	
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
			if( m.get(i,j) <= tresh )
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

#endif
