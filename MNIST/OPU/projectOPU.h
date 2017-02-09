#ifndef PROJECTOPU_H
#define PROJECTOPU_H

#define OPENCV_USE
#include "MAT/Mat.h"

float computeDistance(const cv::Vec3f& c1, const cv::Vec3f& c2);
int computeAssociationsAndCOG(const cv::Mat& im, Mat<float>& finalAssoc);
Mat<float> rotate(const Mat<float>& im, float theta);
Mat<float> rotation(float theta);

#endif
