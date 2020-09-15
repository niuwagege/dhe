#ifndef dhe_hpp
#define dhe_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void build_is_hist(Mat img, Mat& hist_i, Mat &hist_s);
void dhe(cv::Mat img, cv::Mat & result,float alpha=0.5);



#endif /* dhe_hpp */
