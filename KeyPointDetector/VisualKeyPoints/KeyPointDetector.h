#pragma once
#include "opencv2/imgproc.hpp"

struct KPInfo
{
	int x = 0;
	int y = 0;
	int diff = 0;
	int diffAbs = 0;
};

class KeyPointDetector
{
public:
	KeyPointDetector() {};

	virtual void beforeStart() = 0;
	virtual void detectPoints(cv::Mat& image, KPInfo** keyPoints, double* param) = 0;
	virtual void afterFinish() = 0;

	~KeyPointDetector() {};
};

