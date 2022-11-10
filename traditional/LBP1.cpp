#include<opencv2\opencv.hpp>
using namespace cv;
using namespace std;
 
Mat src, gray_img;
int radius = 4;
void callback(int, void*);
 
int main(int arc, char** argv) {             
	src = imread("2.jpg");
	namedWindow("input", CV_WINDOW_AUTOSIZE);
	imshow("input", src);
	cvtColor(src, gray_img, CV_BGR2GRAY);
 
	namedWindow("output", CV_WINDOW_AUTOSIZE);
	createTrackbar("radius", "output", &radius, 20, callback);
	callback(0, 0);
	waitKey(0);
	return 0;
} 
void callback(int, void*) {
	int offset = radius * 2;
	Mat elbp_img = Mat::zeros(gray_img.rows - offset, gray_img.cols - offset, CV_8UC1);
	int width = gray_img.cols;
	int height = gray_img.rows;
 
	int numNeighbors = 8;
	for (int n = 0; n < numNeighbors; n++) {
		double x = static_cast<float>(radius)*cos(2.0 * CV_PI*n / static_cast<float>(numNeighbors));
		double y = static_cast<float>(radius)*sin(2.0 * CV_PI*n / static_cast<float>(numNeighbors));
 
		int fx = static_cast<int>(floor(x));
		int cx = static_cast<int>(ceil(x));
		int fy = static_cast<int>(floor(y));
		int cy = static_cast<int>(ceil(y));
 
		float ty = y - static_cast<float>(fy);
		float tx = x - static_cast<float>(fx);
		
		//周围四个点的权重
		float w1 = (1 - tx)*(1 - ty);
		float w2 = tx*(1 - ty);
		float w3 = tx*ty;
		float w4 = (1 - tx)*ty;
 
		for (int row = radius; row < height - radius; row++) {
			for (int col = radius; col < width - radius; col++) {
				float A = gray_img.at<uchar>(row - cy, col + fx);
				float B = gray_img.at<uchar>(row - cy, col + cx);				
				float C = gray_img.at<uchar>(row - fy, col + fx);
				float D = gray_img.at<uchar>(row - fy, col + cx);
				float center = w1*A + w2*B + w3*C + w4*D;//双线性插值
				elbp_img.at<uchar>(row - radius, col - radius) +=
				(center > gray_img.at<uchar>(row, col)) << 7-n;
			}
		}
	}
	imshow("output", elbp_img);
}