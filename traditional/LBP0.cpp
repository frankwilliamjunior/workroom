#include<opencv2\opencv.hpp>
using namespace cv;
using namespace std;
 
Mat src, gray_img;
int main(int arc, char** argv) {             
	src = imread("4.png");
	namedWindow("input", CV_WINDOW_AUTOSIZE);
	imshow("input", src);
	cvtColor(src, gray_img, CV_BGR2GRAY);
	int width = gray_img.cols;
	int height = gray_img.rows;
	Mat lbp_img = Mat::zeros(gray_img.rows - 2, gray_img.cols - 2, CV_8UC1);
	for (int row = 1; row < height-1; row++) {
		for (int col = 1; col < width - 1; col++) {
			uchar c = gray_img.at<uchar>(row, col);
			uchar tt = 0;
			if (gray_img.at<uchar>(row - 1, col - 1) > c) {
				tt += 1 << 7;
			}
			if (gray_img.at<uchar>(row - 1, col) > c) {
				tt += 1<< 6;
			}
			if (gray_img.at<uchar>(row - 1, col + 1) > c) {
				tt += 1 << 5;
			}
			if (gray_img.at<uchar>(row, col + 1) > c) {
				tt += 1 << 4;
			}
			if (gray_img.at<uchar>(row + 1, col + 1) > c) {
				tt += 1 << 3;
			}
			if (gray_img.at<uchar>(row + 1, col) > c) {
				tt += 1 << 2;
			}
			if (gray_img.at<uchar>(row + 1, col - 1) > c) {
				tt += 1 << 1;
			}
			if (gray_img.at<uchar>(row , col - 1) > c) {
				tt += 1 << 0;
			}
			lbp_img.at<uchar>(row - 1, col - 1) = tt;
		}
	}
	imshow("output", lbp_img);
	waitKey(0);
	return 0;
} 