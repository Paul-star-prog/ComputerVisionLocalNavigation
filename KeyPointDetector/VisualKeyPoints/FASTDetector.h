#pragma once
#include "KeyPointDetector.h"
class FASTDetector :
    public KeyPointDetector
{
protected:
  int indexesX[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
  int indexesY[16] = {3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3};

public:
  FASTDetector() {};
  void beforeStart() override {};
  void detectPoints(cv::Mat& image, KPInfo** keyPoints, double param = 0) override;
  void afterFinish() override {};

  ~FASTDetector() {};
};

