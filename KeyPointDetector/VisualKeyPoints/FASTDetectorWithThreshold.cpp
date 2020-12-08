#include "FASTDetectorWithThreshold.h"

void FASTDetectorWithThreshold::detectPoints(cv::Mat& image, KPInfo** keyPoints, double* param) {

	double thresholdK = param[0];
	int kpPerRegion = param[1];
	int xRegions = int(param[2]), yRegions = int(param[3]);
	int maxDelta = int(param[4]);

	int deltaX = (image.rows-6) / xRegions;
	int deltaY = (image.cols-6) / yRegions;
	int *xStartPositions = new int[xRegions];
	int *yStartPositions = new int[yRegions];
	for (int xIndex = 0; xIndex < xRegions; xIndex++)
		xStartPositions[xIndex] = 3 + xIndex * deltaX;
	
	for (int yIndex = 0; yIndex < yRegions; yIndex++)
		yStartPositions[yIndex] = 3 + yIndex * deltaY;

	int diffCounter = 0;
	int absDiffCounter = 0;
	int regionId = 0;
	int pointsToBecomeKP = 12;

	uchar centralPix;
	uchar threshold;
	uchar pixValue;
	KPInfo newKP;

	for (int xIndex = 0; xIndex < xRegions; xIndex++) {
		for (int yIndex = 0; yIndex < yRegions; yIndex++) {

			regionId = xIndex * xRegions + yIndex;

			for (int i = xStartPositions[xIndex]; i < xStartPositions[xIndex] + deltaX; i++) {
				uchar* rowPtr = image.ptr<uchar>(i);

				for (int j = yStartPositions[yIndex]; j < yStartPositions[yIndex] + deltaY; j++) {

					diffCounter = 0;
					absDiffCounter = 0;

					centralPix = rowPtr[j];
					threshold = centralPix * thresholdK;

					for (int preIndex = 0; preIndex < 4; preIndex++) {
						pixValue = (rowPtr + indexesX[this->preCheckIndexes[preIndex]])[j + indexesY[this->preCheckIndexes[preIndex]]];
						if (centralPix > pixValue + threshold)
							diffCounter++;
						if (centralPix < pixValue - threshold)
							diffCounter--;
					}

					if (diffCounter <= 2)
						continue;

					diffCounter = 0;

					for (int pixelIndex = 0; pixelIndex < 16; pixelIndex++) {
						pixValue = (rowPtr+indexesX[pixelIndex])[j+indexesY[pixelIndex]]; 
						if (centralPix < pixValue - threshold) {
							diffCounter--;
							absDiffCounter += (centralPix - pixValue);
						}
						if (centralPix > pixValue + threshold) {
							diffCounter++;
							absDiffCounter += (centralPix - pixValue);
						}	
					}
					if (abs(diffCounter) >= pointsToBecomeKP) {

						if (maxDelta != 0 && abs(absDiffCounter) > maxDelta)
							continue;

						newKP.x = i; newKP.y = j;
						newKP.diff = diffCounter;
						newKP.diffAbs = absDiffCounter;
						this->addKeyPoint(keyPoints[regionId], newKP, kpPerRegion);
					}
				}
			}
		}
	}
}

double calcDistance(KPInfo p1, KPInfo p2) {
	return pow((p1.x-p2.x) * (p1.x-p2.x) + (p1.y-p2.y) * (p1.y-p2.y), 0.5);
}

void FASTDetectorWithThreshold::addKeyPoint(KPInfo * kpHolder, KPInfo kp, int kpPerRegion) {
	for (int kpIndex = 0; kpIndex < kpPerRegion; kpIndex++) {
		if (
			abs(kp.diff) >= abs(kpHolder[kpIndex].diff) &&
			!this->kpInRange(kpHolder, kp, kpPerRegion)
		) {
			if (kpIndex != kpPerRegion-1) {
				for (int swapIndex = kpPerRegion-1; swapIndex > kpIndex; swapIndex--) {
					kpHolder[swapIndex] = kpHolder[swapIndex - 1];
				}
			}
			
			kpHolder[kpIndex].diff = kp.diff;
			kpHolder[kpIndex].diffAbs = kp.diffAbs;
			kpHolder[kpIndex].x = kp.x;
			kpHolder[kpIndex].y = kp.y;
			break;
		}
	}
}