// VisualKeyPoints.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "./FASTDetectorWithThreshold.h"

#include <opencv2/features2d.hpp>

#include <fstream>
#include <iostream>

#include <vector>

using namespace std::chrono;

/*
* 1 - input image
* 2 - output text file
* 3 - kp count
* 4 - threshold param
* 5 - ceiling delta
*/
int main(int argc, char* argv[])
{
	std::string imagePath = cv::samples::findFile(argv[1]);
	cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

	/*cv::imshow("Original image", image);
	int k = cv::waitKey(0);*/
	// threshold param, kp count, regions_x, regions_y, top ceiling
	double params[5] = { std::stod(argv[4]), std::stoi(argv[3]), 1, 1, std::stod(argv[5])};

	int kpPerRegion = int(params[1]);
	int xRegions = int(params[2]);
	int yRegions = int(params[3]);

	const int imageRegions = xRegions*yRegions;
	KPInfo **keyPoints = new KPInfo * [imageRegions];

	for (int i = 0; i < imageRegions; i++) {
		keyPoints[i] = new KPInfo[kpPerRegion];
		for (int j = 0; j < kpPerRegion; j++)
			keyPoints[i][j] = KPInfo();
	}

	//AnotherOneDetector detector = AnotherOneDetector();
 	FASTDetectorWithThreshold detector = FASTDetectorWithThreshold();
	//cv::Mat res = 
	detector.detectPoints(image, keyPoints, params); 

	// high_resolution_clock::time_point t2 = high_resolution_clock::now();

	// duration<double, std::milli> time_span = t2 - t1;
	// std::cout << "Time passed: " << time_span.count() << " milliseconds." << std::endl;

	std::ofstream out(argv[2]);

	for (int i = 0; i < imageRegions; i++) {
		for (int j = 0; j < kpPerRegion; j++) {
			KPInfo kp = keyPoints[i][j];
			out << kp.x << " " << kp.y << std::endl;
		}
	}

	out.close();

	std::vector<cv::KeyPoint> keypointsD;
	cv::Ptr<cv::FastFeatureDetector> detectorCV = cv::FastFeatureDetector::create();

	detectorCV->detect(image, keypointsD, cv::Mat());
	cv::drawKeypoints(image, keypointsD, image);
	cv::imwrite("cvRes.png", image);

	/*int x = 0;
	int y = 0;

	int deltaX = (image.rows - 6) / xRegions;
	int deltaY = (image.cols - 6) / yRegions;
	int* xStartPositions = new int[xRegions];
	int* yStartPositions = new int[yRegions];
	for (int xIndex = 0; xIndex < xRegions; xIndex++) {
		xStartPositions[xIndex] = 3 + xIndex * deltaX;
		yStartPositions[xIndex] = 3 + xIndex * deltaY;
	}

	for (int i = 0; i < xRegions; i++)
		cv::line(
			image,
			cv::Point(0, xStartPositions[i]),
			cv::Point(image.cols - 1, xStartPositions[i]),
			255
		);

	for (int j = 0; j < yRegions; j++)
		cv::line(
			image,
			cv::Point(yStartPositions[j], 0),
			cv::Point(yStartPositions[j], image.rows - 1),
			255
		);

	for (int index = 0; index < imageRegions; index++) {
		for (int kp = 0; kp < kpPerRegion; kp++) {
			if (abs(keyPoints[index][kp].diffAbs) > 0) {
				x = keyPoints[index][kp].x;
				y = keyPoints[index][kp].y;
				cv::circle(image, cv::Point(y, x), 3, cv::Scalar(0));
			}
			else {
				std::cout << "Key point at " << index << " is not presented" << std::endl;
			}
		}
	}
	cv::imshow("Test image", image);
	cv::waitKey(0);
	cv::imwrite("res.png", image);*/

}  