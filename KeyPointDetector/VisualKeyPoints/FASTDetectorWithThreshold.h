#pragma once
#include "KeyPointDetector.h"
#include <math.h>



class FASTDetectorWithThreshold :
    public KeyPointDetector {
protected:
    int indexesX[16] = { 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1 };
    int indexesY[16] = { -3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3 };
    int preCheckIndexes[4] = { 0, 8, 4, 12 };
    double MIN_DISTANCE_BETWEEN_POINTS = 5;
    uchar imageMin = 0;
    uchar imageMax = 0;
    double pixK = 0;
public:
  FASTDetectorWithThreshold() {};
  void beforeStart() override {};
  void detectPoints(cv::Mat& image, KPInfo** keyPoints, double* param) override;
  void addKeyPoint(KPInfo *KPHolder, KPInfo kp, int kpPerRegion);
  void findMinMax(cv::Mat& image);
  uchar pixValueReshaped(uchar pix);
  void prepareImage(cv::Mat& image);
  bool kpInRange(KPInfo* KPHolder, KPInfo kp, int kpPerRegion);
  void afterFinish() override {};
  ~FASTDetectorWithThreshold() {};
};

