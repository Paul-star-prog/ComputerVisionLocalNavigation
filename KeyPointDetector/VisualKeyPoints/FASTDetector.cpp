#include "FASTDetector.h"

//void FASTDetector::detectPoints(cv::Mat image, KPInfo* keyPoints, double param) {
//	cv::Mat resultImage = cv::Mat(image.rows, image.cols, image.type());
//
//	for (int i = 4; i < image.rows-4; i++) {
//		for (int j = 4; j < image.cols-4; j++) {
//			int diffCounter = 0;
//			for (int pixelIndex = 0; pixelIndex < 16; pixelIndex++) {
//				if (image.at<uchar>(i, j) > image.at<uchar>(i + indexesX[pixelIndex], j + indexesY[pixelIndex]))
//					diffCounter++;
//				if (image.at<uchar>(i, j) < image.at<uchar>(i + indexesX[pixelIndex], j + indexesY[pixelIndex]))
//					diffCounter--;
//				if (abs(diffCounter) >= 9)
//					break;
//			}
//			if (abs(diffCounter) >= 9) {
//				resultImage.at<uchar>(i, j) = 255;
//			}
//			else {
//				resultImage.at<uchar>(i, j) = 0;
//			}
//		}
//	}
//	/*return resultImage;*/
//}
