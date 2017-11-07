// stereo_calib.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include "videoInput.h"

//#define LOCAL_IMAGE

//down sample the input image by half
//#define DOWN_SAMPLE

#define FRAME_WIDTH 1080
#define FRAME_HEIGHT 1080
#define RENDER_WIDTH 2160
#define RENDER_HEIGHT 1080


//circle grids per row/comlun
#define CON_X 4
#define CON_Y 11

//print 
#define DEBUG_PRINT

//camera setup fliped 90 or not
//#define CAMERA_FLIP

//camera setup invert 180 or not
//#define CAMERA_INVERT

//fisheye model of camera
//#define FISHEYE

using namespace std;
using namespace cv;

void StereoCalib(Mat * imagelist, Size boardSize,int number_calib_image,float squareSize, Mat *mx1, Mat *mx2, Mat *my1, Mat *my2 ,Rect *validRoi, bool useCalibrated, bool showRectified,float *error);

int _tmain(int argc, char **argv)
{
	Mat frame(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3);
	Mat frame_right(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3);
	char image_path[50];
	char image_path_right[50];

	videoInput VI;
	int left_index;
	int right_index;

	int numDevices = videoInput::listDevices();	
	std::vector <std::string> list = videoInput::getDeviceList(); 
	for(int i = 0; i < list.size(); i++){
		printf("[%i] device is %s\n", i, list[i].c_str());

		if(list[i] == "VMG-CAM-R")
		{
			right_index = i;
			VI.setupDevice(i, FRAME_WIDTH, FRAME_HEIGHT, VI_COMPOSITE);
		}

		if(list[i] == "VMG-CAM-L")
		{
			left_index = i;
			VI.setupDevice(i, FRAME_WIDTH, FRAME_HEIGHT, VI_COMPOSITE);
		}
		
		// 

	}

	cout<<"left index : "<<left_index<<endl;
	cout<<"right index : "<<right_index<<endl;

	
	//VI.setupDevice(left_index, FRAME_WIDTH, FRAME_HEIGHT, VI_COMPOSITE); 
	//VI.setupDevice(right_index, FRAME_WIDTH, FRAME_HEIGHT, VI_COMPOSITE);
	



	//VideoCapture cap(1);
	//VideoCapture cap_right(0);
	//if(!cap.isOpened() || !cap_right.isOpened()) return -1;
	//cap.set(CV_CAP_PROP_FOURCC ,CV_FOURCC('M', 'J', 'P', 'G') );
	//cap.set(CV_CAP_PROP_FRAME_WIDTH,FRAME_WIDTH);
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT);
	//cap_right.set(CV_CAP_PROP_FOURCC ,CV_FOURCC('M', 'J', 'P', 'G') );
	//cap_right.set(CV_CAP_PROP_FRAME_WIDTH,FRAME_WIDTH);
	//cap_right.set(CV_CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT);

	namedWindow("camera_left",0);
	namedWindow("camera2_right",0);

	int count = 0;

#ifdef LOCAL_IMAGE
	count = 19;
#else
	//capture images
	for(;;)
	{
#ifdef DEBUG_PRINT
	double tic=(double)cvGetTickCount();
#endif
		//cap>>frame;
		//cap_right>>frame_right;

	if (VI.isFrameNew(left_index))
		{
			VI.getPixels(left_index, frame.data, false, true);
			//VI.getPixels(right_index, frame_right.data, false, true);
		}

	if (VI.isFrameNew(right_index))
	{
		//VI.getPixels(left_index, frame.data, false, true);
		VI.getPixels(right_index, frame_right.data, false, true);
	}

#ifdef CAMERA_FLIP
		flip(frame.t(), frame, 0);
		flip(frame_right.t(), frame_right, 0);
#endif

#ifdef CAMERA_INVERT
		flip(frame, frame, -1);
		flip(frame_right, frame_right, -1);
#endif
		//Mat(frame,Rect(0,0,RENDER_WIDTH/2,RENDER_HEIGHT)).copyTo(frame_crop);
		//Mat(frame_right,Rect(0,0,RENDER_WIDTH/2,RENDER_HEIGHT)).copyTo(frame_crop_right);

		#ifdef DEBUG_PRINT
	double toc=(double)cvGetTickCount();
	double detectionTime = (toc-tic)/((double) cvGetTickFrequency()*1000);
	//cout << " one frame in:  " << detectionTime << endl;
#endif

		if(waitKey(10) == 99)
		{
			sprintf(image_path, "./save_image/image_%d_left.jpg", count);
			sprintf(image_path_right, "./save_image/image_%d_right.jpg", count);
			imwrite(image_path,frame);
			imwrite(image_path_right,frame_right);
			frame = 255 - frame;
			frame_right = 255 - frame_right;
			printf("%s \n",image_path);
			count++;
		}


		//imshow("camera", frame_right);
		//imshow("camera2", frame);
		imshow("camera_left",frame);
		imshow("camera2_right",frame_right);

		if(waitKey(30) == 27) break;
	}
#endif

	//read image to a list
	//count = 12;//this is really bad code!!!!,bugs are here
	Mat* image_calib_list;
	image_calib_list = new Mat [count*2];

	for(int i = 0; i < count; i++)
	{
		sprintf(image_path, "./save_image/image_%d_left.jpg", i);
		sprintf(image_path_right, "./save_image/image_%d_right.jpg", i);
		Mat left_temp = imread(image_path);
		Mat right_temp = imread(image_path_right);

#ifdef DOWN_SAMPLE
		resize(left_temp,left_temp,Size(left_temp.size().width/2,left_temp.size().height/2));
		resize(right_temp,right_temp,Size(right_temp.size().width/2,right_temp.size().height/2));
#endif
		std::cout << i << " "<<left_temp.channels()<<endl;
		std::cout << i << " " << right_temp.channels() << endl;

		Mat left_temp_gray;
		Mat right_temp_gray;

		cvtColor(left_temp,left_temp_gray,CV_BGR2GRAY);
		cvtColor(right_temp,right_temp_gray,CV_BGR2GRAY);

		//imshow("left",left_temp_gray);
		//imshow("right",right_temp_gray);
		//waitKey();

		printf("read %s \n", image_path);

		image_calib_list[i].create(left_temp.size(),CV_8UC1);
		image_calib_list[i*2+1].create(right_temp.size(),CV_8UC1);

		//resize the images!
		//int chop = 50;
		//left_temp_gray = left_temp_gray.adjustROI(-chop, -chop, -chop, -chop).clone();
		//resize(left_temp_gray, left_temp_gray, cv::Size(FRAME_HEIGHT, FRAME_WIDTH));
		//right_temp_gray = right_temp_gray.adjustROI(-chop, -chop, -chop, -chop).clone();
		//resize(right_temp_gray, right_temp_gray, cv::Size(FRAME_HEIGHT, FRAME_WIDTH));

		//Mat left_resize(left_temp_gray.size(), CV_8UC1);
		//resize(left_temp_gray, left_temp_gray, cv::Size(left_temp_gray.size().width*0.7, left_temp_gray.size().height*0.7));
		//left_temp_gray.copyTo(left_resize(Rect((left_resize.cols - left_temp_gray.cols)/2 , (left_resize.rows - left_temp_gray.rows)/2,left_temp_gray.cols,left_temp_gray.rows)));

		//Mat right_resize(right_temp_gray.size(), CV_8UC1);
		//resize(right_temp_gray, right_temp_gray, cv::Size(right_temp_gray.size().width*0.7, right_temp_gray.size().height*0.7));
		//right_temp_gray.copyTo(right_resize(Rect((right_resize.cols - right_temp_gray.cols)/2 , (right_resize.rows - right_temp_gray.rows)/2,right_temp_gray.cols,right_temp_gray.rows)));

		left_temp_gray.copyTo(image_calib_list[i*2]);
		right_temp_gray.copyTo(image_calib_list[i*2+1]);

		//left_resize.copyTo(image_calib_list[i*2]);
		//right_resize.copyTo(image_calib_list[i*2+1]);
	}

	//stereo calib
	Size boardSize(CON_X, CON_Y);
	//Size boardSize(CON_Y, CON_X);
	int number_calib_image = count*2;
	float squareSize =18.0f;
	//float squareSize =16.0f;

	Mat mx1,my1,mx2,my2;
#ifdef DOWN_SAMPLE
	mx1.create( FRAME_HEIGHT/2, FRAME_WIDTH/2, CV_16S);
	my1.create( FRAME_HEIGHT/2, FRAME_WIDTH/2, CV_16S);
	mx2.create( FRAME_HEIGHT/2, FRAME_WIDTH/2, CV_16S);
	my2.create( FRAME_HEIGHT/2, FRAME_WIDTH/2, CV_16S);
#else
	mx1.create( FRAME_HEIGHT, FRAME_WIDTH, CV_16S);
	my1.create( FRAME_HEIGHT, FRAME_WIDTH, CV_16S);
	mx2.create( FRAME_HEIGHT, FRAME_WIDTH, CV_16S);
	my2.create( FRAME_HEIGHT, FRAME_WIDTH, CV_16S);
#endif
	
	Rect validRoi[2];
	float error1 = 0.0f;
	StereoCalib(image_calib_list, boardSize,number_calib_image,squareSize, &mx1,&mx2,&my1,&my2, validRoi, true, 1,&error1);

	//rectify images to check
#ifdef DOWN_SAMPLE
	//putting the downsampled image size in
	Mat frame_rectify(Size(frame.size().width/2,frame.size().height/2), CV_8UC3);
	Mat frame_rectify_right(Size(frame_right.size().width/2,frame_right.size().height/2), CV_8UC3);
#else
	Mat frame_rectify(frame.size(), CV_8UC3);
	Mat frame_rectify_right(frame_right.size(), CV_8UC3);
#endif
	

	bool bool_rectify = true;
#ifdef LOCAL_IMAGE
	
	for(int i = 0; i < count; i++)
	{
		frame = image_calib_list[i*2];
		frame_right = image_calib_list[2*i+1];
		
		//flip(frame.t(), frame, 1);
		//flip(frame_right.t(), frame_right, 1);
		if(bool_rectify)
		{
			remap(frame, frame_rectify, mx1, my1, CV_INTER_LINEAR);
			remap(frame_right, frame_rectify_right, mx2, my2, CV_INTER_LINEAR);
			//draw a few lines to check epipolar
			for(int j = 0; j < FRAME_WIDTH; j+=50)
			{
				line(frame_rectify,Point(0,j), Point(FRAME_HEIGHT,j), Scalar(255,0,0));
				line(frame_rectify_right,Point(0,j), Point(FRAME_HEIGHT,j), Scalar(255,0,0));
			}
			imshow("camera_left",frame_rectify);
			imshow("camera2_right",frame_rectify_right);
		}
		else
		{
			for(int j = 0; j < FRAME_WIDTH; j+=50)
			{
				line(frame,Point(0,j), Point(FRAME_HEIGHT,j), Scalar(255,0,0));
				line(frame_right,Point(0,j), Point(FRAME_HEIGHT,j), Scalar(255,0,0));
			}
			imshow("camera_left",frame);
			imshow("camera2_right",frame_right);
		}

		waitKey();
	}
#else



	for(;;)
	{
		//cap>>frame;
		//cap_right>>frame_right;

		if(VI.isFrameNew(left_index))
		{
			VI.getPixels(left_index, frame.data, false, true);
			VI.getPixels(right_index, frame_right.data, false, true);
		}

#ifdef CAMERA_FLIP
		flip(frame.t(), frame, 0);
		flip(frame_right.t(), frame_right, 0);
#endif
#ifdef CAMERA_INVERT
		flip(frame, frame, -1);
		flip(frame_right, frame_right, -1);
#endif
		if(waitKey(10) == 114)
		{bool_rectify = ! bool_rectify;}
		if(bool_rectify)
		{

#ifdef DOWN_SAMPLE
		Mat frame_down, frame_down_right;
		cv::resize(frame,frame_down,Size(frame.size().width/2,frame.size().height/2));
		cv::resize(frame_right,frame_down_right,Size(frame_right.size().width/2,frame_right.size().height/2));

		remap(frame_down, frame_rectify, mx1, my1, CV_INTER_LINEAR);
		remap(frame_down_right, frame_rectify_right, mx2, my2, CV_INTER_LINEAR);
#else
		remap(frame, frame_rectify, mx1, my1, CV_INTER_LINEAR);
		remap(frame_right, frame_rectify_right, mx2, my2, CV_INTER_LINEAR);
#endif


			//draw a few lines to check epipolar
			for(int j = 0; j < FRAME_WIDTH; j+=50)
			{
#ifdef CAMERA_FLIP
				line(frame_rectify,Point(0,j), Point(FRAME_HEIGHT,j), Scalar(255,0,0));
				line(frame_rectify_right,Point(0,j), Point(FRAME_HEIGHT,j), Scalar(255,0,0));
#else
				line(frame_rectify,Point(0,j), Point(FRAME_WIDTH,j), Scalar(255,0,0));
				line(frame_rectify_right,Point(0,j), Point(FRAME_WIDTH,j), Scalar(255,0,0));
#endif
			}
			imshow("camera_left",frame_rectify);
			imshow("camera2_right",frame_rectify_right);
		}
		else
		{
			for(int j = 0; j < FRAME_WIDTH; j+=50)
			{
				line(frame,Point(0,j), Point(FRAME_HEIGHT,j), Scalar(255,0,0));
				line(frame_right,Point(0,j), Point(FRAME_HEIGHT,j), Scalar(255,0,0));
			}
			imshow("camera_left",frame);
			imshow("camera2_right",frame_right);
		}

		if(waitKey(30) == 27) break;
	}
#endif

	FileStorage fs("./save_param/calib_para.yml", CV_STORAGE_WRITE);
	fs << "MX1" << mx1 << "MX2" << mx2 << "MY1" << my1 << "MY2" << my2;
	fs.release();

	VI.stopDevice(left_index);
	VI.stopDevice(right_index);

	delete[] image_calib_list;
	return 0;
}


void StereoCalib(Mat * imagelist, Size boardSize,int number_calib_image,float squareSize, Mat *mx1, Mat *mx2, Mat *my1, Mat *my2 ,Rect *validRoi, bool useCalibrated, bool showRectified,float *error)
{

	bool displayCorners = true;//true;
	const int maxScale = 2;
	//float squareSize = 25.f;  // Set this to your actual square size
	// ARRAY AND VECTOR STORAGE:
#ifdef FISHEYE
	//fiseeye calib requires 64 double
	vector<vector<Point2d> > imagePoints[2];
	vector<vector<Point3d> > objectPoints;
	//drawing function require 32f
	vector<vector<Point2f> > imagePoints_32f[2];
#else
	vector<vector<Point2f> > imagePoints[2];
	vector<vector<Point3f> > objectPoints;
#endif
	Size imageSize;

	int i, j, k, nimages = number_calib_image/2;

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
#ifdef FISHEYE
	imagePoints_32f[0].resize(nimages);
	imagePoints_32f[1].resize(nimages);
#endif

	for( i = j = 0; i < nimages; i++ )
	{
		for( k = 0; k < 2; k++ )
		{
			/*            const string& filename = imagelist[i*2+k];*/
			Mat img = imagelist[i*2+k];
			if(img.empty())
				break;
			if( imageSize == Size() )
				imageSize = img.size();
			else if( img.size() != imageSize )
			{
				// cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
				break;
			}
			bool found = false;
#ifdef FISHEYE
			vector<Point2d>& corners = imagePoints[k][j];
			vector<Point2f>& corners_32f = imagePoints_32f[k][j];
#else
			vector<Point2f>& corners = imagePoints[k][j];
#endif
			for( int scale = 1; scale <= maxScale; scale++ )
			{
				Mat timg;
				if( scale == 1 )
					timg = img;
				else
					resize(img, timg, Size(), scale, scale);
				found =findCirclesGrid(timg, boardSize, corners, CALIB_CB_ASYMMETRIC_GRID );
				if( found )
				{
					if( scale > 1 )
					{
						Mat cornersMat(corners);
						cornersMat *= 1./scale;
					}
					break;
				}
			}
			if( displayCorners )
			{
				Mat cimg, cimg1;
				cvtColor(img, cimg, CV_GRAY2BGR);
#ifdef FISHEYE
				
				for(int i = 0; i<(int)corners.size();i++)
				{
					corners_32f.push_back((Point2f)corners[i]);
				}
				drawChessboardCorners(cimg, boardSize, corners_32f, found);
#else
				drawChessboardCorners(cimg, boardSize, corners, found);
#endif

				double sf = (double)640.0/MAX(img.rows, img.cols);
				resize(cimg, cimg1, Size(), sf, sf);
				imshow("corners", cimg1);
				char c = (char)waitKey(1);
				if( c == 27 || c == 'q' || c == 'Q' ) //Allow ESC to quit
					exit(-1);
			}
			else
				putchar('.');
			if( !found )
				break;
		}
		if( k == 2 )
		{
			j++;
		}
	}
	cout << j << " pairs have been successfully detected.\n";
	nimages = j;
	if( nimages < 2 )
	{
		cout << "Error: too little pairs to run the calibration\n";
		return;
	}

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	objectPoints.resize(nimages);

	for( i = 0; i < nimages; i++ )
	{
		for( j = 0; j < boardSize.height; j++ )
			for( k = 0; k < boardSize.width; k++ )
#ifdef FISEHEYE
				objectPoints[i].push_back(Point3d((2*k+j%2)*squareSize, j*squareSize, 0));
#else
				objectPoints[i].push_back(Point3f((2*k+j%2)*squareSize, j*squareSize, 0));
#endif
	}

	cout<<"object pont:"<<objectPoints[0]<<endl;

	cout << "Running stereo calibration ...\n";

	Mat cameraMatrix[2], distCoeffs[2];
	cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
	cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
	Mat R, T, E, F;
#ifdef FISHEYE
	cv::Vec4d D;
	std::vector<cv::Vec3d> rvecs;
    std::vector<cv::Vec3d> tvecs;
#else
	vector<Mat> rvecs,tvecs;
#endif
	Mat R11,R22,T11,T22;

	
#ifdef FISHEYE
	//int flag1 = 0;
	//flag1 = fisheye::CALIB_RECOMPUTE_EXTRINSIC|fisheye::CALIB_FIX_SKEW|fisheye::CALIB_FIX_K1

		cv::fisheye::calibrate(objectPoints, imagePoints[0], imageSize, cameraMatrix[0],
		distCoeffs[0], rvecs, tvecs, fisheye::CALIB_RECOMPUTE_EXTRINSIC + fisheye::CALIB_CHECK_COND + fisheye::CALIB_FIX_SKEW+fisheye::CALIB_FIX_K4+fisheye::CALIB_FIX_K3+fisheye::CALIB_FIX_K2+fisheye::CALIB_FIX_K1 ,TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5));


		//cv::fisheye::calibrate(objectPoints, imagePoints[0], imageSize, cameraMatrix[0],
		//distCoeffs[0], rvecs, tvecs, fisheye::CALIB_RECOMPUTE_EXTRINSIC + fisheye::CALIB_CHECK_COND + fisheye::CALIB_FIX_SKEW+fisheye::CALIB_FIX_K4+fisheye::CALIB_FIX_K3+fisheye::CALIB_FIX_K2+fisheye::CALIB_FIX_K1 ,TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5));

	/*cv::fisheye::calibrate(objectPoints, imagePoints[0], imageSize, cameraMatrix[0],
		distCoeffs[0], rvecs, tvecs, fisheye::CALIB_RECOMPUTE_EXTRINSIC + fisheye::CALIB_CHECK_COND + fisheye::CALIB_FIX_SKEW ,TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5));*/
#else
	calibrateCamera(objectPoints, imagePoints[0], imageSize, cameraMatrix[0],
		distCoeffs[0], rvecs, tvecs,0,TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5));
	//calibrateCamera(objectPoints, imagePoints[0], imageSize, cameraMatrix[0],
	//	distCoeffs[0], rvecs, tvecs, CV_CALIB_FIX_K3|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5,TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5));
#endif
	Rodrigues(rvecs[0],R11);
	//tvecs[0].copyTo(T11);
	//cout<<" cameramatrix 0 0 "<<cameraMatrix[0].at<double>(0,0)<<endl;
	//cout<<" dist 0 "<<distCoeffs[0].at<double>(0,0)<<endl;
	//cout<<" dist 1 "<<distCoeffs[0].at<double>(0,1)<<endl;
	//cout<<" dist 2 "<<distCoeffs[0].at<double>(0,2)<<endl;
	//cout<<" dist 3 "<<distCoeffs[0].at<double>(0,3)<<endl;

#ifdef FISHEYE
	cv::fisheye::calibrate(objectPoints, imagePoints[1], imageSize, cameraMatrix[1],
		distCoeffs[1], rvecs, tvecs, fisheye::CALIB_RECOMPUTE_EXTRINSIC + fisheye::CALIB_CHECK_COND + fisheye::CALIB_FIX_SKEW+fisheye::CALIB_FIX_K4+fisheye::CALIB_FIX_K3+fisheye::CALIB_FIX_K2+fisheye::CALIB_FIX_K1,TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5));

	/*cv::fisheye::calibrate(objectPoints, imagePoints[1], imageSize, cameraMatrix[1],
		distCoeffs[1], rvecs, tvecs, fisheye::CALIB_RECOMPUTE_EXTRINSIC + fisheye::CALIB_CHECK_COND + fisheye::CALIB_FIX_SKEW,TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5));*/
#else
	calibrateCamera(objectPoints, imagePoints[1], imageSize, cameraMatrix[1],
		distCoeffs[1], rvecs, tvecs, 0,TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5));

	/*calibrateCamera(objectPoints, imagePoints[1], imageSize, cameraMatrix[1],
		distCoeffs[1], rvecs, tvecs, CV_CALIB_FIX_K3|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5,TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5));*/
#endif
	Rodrigues(rvecs[0],R22);
	//tvecs[0].copyTo(T22);


//#ifdef FISHEYE
	//Mat undist_left, undist_right;
	//fisheye::undistortImage(imagelist[0], undist_left, cameraMatrix[0],distCoeffs[0],cameraMatrix[0]);
	//namedWindow("undistort_left",0);
	//imshow("undistort_left",undist_left);
	//waitKey();

	//vector<vector<Point2d> > imagePoints_undist[2];
	//imagePoints_undist[0].resize(nimages);
	//imagePoints_undist[1].resize(nimages);

	//for(int i = 0; i < imagePoints[0].size(); i++)
	//{
	//	fisheye::undistortPoints(imagePoints[0][i], imagePoints_undist[0][i], cameraMatrix[0], distCoeffs[0]);
	//	fisheye::undistortPoints(imagePoints[1][i], imagePoints_undist[1][i], cameraMatrix[1], distCoeffs[1]);
	//}

//#endif

	//cameraMatrix[0].at<double>(0,0) += 300;
	//cameraMatrix[0].at<double>(1,1) += 300;

	//cameraMatrix[1].at<double>(0,0) += 300;
	//cameraMatrix[1].at<double>(1,1) += 300;
#ifdef FISHEYE
		double rms = cv::fisheye::stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
		cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T,
		fisheye::CALIB_FIX_INTRINSIC + fisheye::CALIB_USE_INTRINSIC_GUESS + fisheye::CALIB_CHECK_COND + fisheye::CALIB_FIX_SKEW +fisheye::CALIB_FIX_K4+fisheye::CALIB_FIX_K3+fisheye::CALIB_FIX_K2+fisheye::CALIB_FIX_K1,
		TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,200, 1e-5));
#else
	double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
		cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, E, F,
		TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,200, 1e-5),
		CV_CALIB_FIX_INTRINSIC  +
		CV_CALIB_RATIONAL_MODEL );

	//double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
	//	cameraMatrix[0], distCoeffs[0],
	//	cameraMatrix[1], distCoeffs[1],
	//	imageSize, R, T, E, F,
	//	TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,200, 1e-5),
	//	CV_CALIB_FIX_INTRINSIC  +
	//	CV_CALIB_RATIONAL_MODEL +CV_CALIB_FIX_K1 + CV_CALIB_FIX_K2+
	//	CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5);
	//
	// 					CV_CALIB_FIX_INTRINSIC  +CV_CALIB_RATIONAL_MODEL
	//                     );
#endif
	cout << "done with RMS error=" << rms << endl;
	error[0] = rms;
	//cout<<" cameramatrix 0 0 "<<cameraMatrix[0].at<double>(0,0)<<endl;
	//cout<<" dist 0 "<<distCoeffs[0].at<double>(0,0)<<endl;
	//cout<<" R "<<R.at<double>(0,0)<<endl;
	//cout<<" T "<<T.at<double>(0,0)<<endl;

	//FileStorage fs_rt("./save_param/rt_vectors.yml", CV_STORAGE_WRITE);
	//if( fs_rt.isOpened() )
	//{
	//	fs_rt << "R1" << R11 << "T1" << T11 << "R2" << R22 << "T2" << T22;
	//	fs_rt.release();
	//}



#ifndef FISHEYE
	// CALIBRATION QUALITY CHECK
	double err = 0;
	int npoints = 0;
	vector<Vec3f> lines[2];
	for( i = 0; i < nimages; i++ )
	{
		int npt = (int)imagePoints[0][i].size();
		Mat imgpt[2];
		for( k = 0; k < 2; k++ )
		{
			imgpt[k] = Mat(imagePoints[k][i]);
			undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
			computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
		}
		for( j = 0; j < npt; j++ )
		{
			double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
				imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
				fabs(imagePoints[1][i][j].x*lines[0][j][0] +
				imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
			err += errij;
		}
		npoints += npt;
	}
	cout << "average reprojection err = " <<  err/npoints << endl;
	error[1] = err/npoints;
#endif
	// save intrinsic parameters
	FileStorage fs("./save_param/intrinsics.yml", CV_STORAGE_WRITE);
	if( fs.isOpened() )
	{

#ifdef DOWN_SAMPLE
		fs<<"image_width" << FRAME_WIDTH/2;
        fs<<"image_height" << FRAME_HEIGHT/2;
#else
		fs<<"image_width" << FRAME_WIDTH;
        fs<<"image_height" << FRAME_HEIGHT;
#endif
		fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
			"M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
		fs.release();
	}
	else
		cout << "Error: can not save the intrinsic parameters\n";

	Mat R1, R2, P1, P2, Q;
	/*    Rect validRoi[2];*/

	std::cout<<"camera matrix 0 "<<cameraMatrix[0]<<endl;
	std::cout<<"camera matrix 1"<<cameraMatrix[1]<<endl;
	//std::cout<<"dist 0"<<distCoeffs[0]<<endl;
	//std::cout<<"dist 1"<<distCoeffs[1]<<endl;

#ifdef FISHEYE
	cv::fisheye::stereoRectify(cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY,
		imageSize);
#else
	stereoRectify(cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, R1, R2, P1, P2, Q,CALIB_ZERO_DISPARITY,
		-1, imageSize, &validRoi[0], &validRoi[1]);
#endif

	fs.open("./save_param/validRoi.yml", CV_STORAGE_WRITE);
	if( fs.isOpened() )
	{
		Mat temp1,temp2;
		temp1.create(1,4,CV_32S);
		temp2.create(1,4,CV_32S);

		temp1.at<int>(0,0) = validRoi[0].x;
		temp1.at<int>(0,1) = validRoi[0].y;
		temp1.at<int>(0,2) = imageSize.width-(validRoi[0].x+validRoi[0].width-1);
		temp1.at<int>(0,3) = imageSize.height-(validRoi[0].y+validRoi[0].height-1);
		temp2.at<int>(0,0) = validRoi[1].x;
		temp2.at<int>(0,1) = validRoi[1].y;
		temp2.at<int>(0,2) = imageSize.width-(validRoi[1].x+validRoi[1].width-1);
		temp2.at<int>(0,3) = imageSize.height-(validRoi[1].y+validRoi[1].height-1);
		fs << "validRoi1" << temp1 << "validRoi2" << temp2;
		fs.release();
		temp1.release();
		temp2.release();
	}
	//else
		//cout << "Error: can not save the intrinsic parameters\n";


	fs.open("./save_param/extrinsics.yml", CV_STORAGE_WRITE);
	if( fs.isOpened() )
	{
		fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
		fs.release();
	}

	//std::cout<<"R:"<<R<<endl;
	//std::cout<<"T:"<<T<<endl;
	//std::cout<<"P1"<<P1<<endl;
	//else
		//cout << "Error: can not save the intrinsic parameters\n";

	// OpenCV can handle left-right
	// or up-down camera arrangements
	bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

	// COMPUTE AND DISPLAY RECTIFICATION
	if( !showRectified )
		return;

	if( useCalibrated )
	{
		// we already computed everything
	}

	else
		// use intrinsic parameters of each camera, but
		// compute the rectification transformation directly
		// from the fundamental matrix
	{
		vector<Point2f> allimgpt[2];
		for( k = 0; k < 2; k++ )
		{
			for( i = 0; i < nimages; i++ )
				std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
		}
		F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
		//std::cout<<"Fundamental:"<<F<<endl;
		Mat H1, H2;
		stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);
		//std::cout<<"H1:"<<H1<<endl;
		//std::cout<<"H2"<<H2<<endl;

		R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
		R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
		P1 = cameraMatrix[0];
		P2 = cameraMatrix[1];
	}

	//Precompute maps for cv::remap()
#ifdef FISHEYE
	cv::fisheye::initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, mx1[0], my1[0]);
	cv::fisheye::initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, mx2[0], my2[0]);
#else
	initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, mx1[0], my1[0]);
	initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, mx2[0], my2[0]);
#endif

}

























//	for(;;)
//	{
//		//cap>>frame;
//		//cap_right>>frame_right;
//
//		if(VI.isFrameNew(0))
//		{
//			VI.getPixels(left_index, frame.data, false, true);
//			VI.getPixels(right_index, frame_right.data, false, true);
//		}
//
//#ifdef CAMERA_FLIP
//		flip(frame.t(), frame, 0);
//		flip(frame_right.t(), frame_right, 0);
//#endif
//#ifdef CAMERA_INVERT
//		flip(frame, frame, -1);
//		flip(frame_right, frame_right, -1);
//#endif
//		if(waitKey(10) == 114)
//		{bool_rectify = ! bool_rectify;}
//		if(bool_rectify)
//		{
//
//#ifdef DOWN_SAMPLE
//		Mat frame_down, frame_down_right;
//		cv::resize(frame,frame_down,Size(frame.size().width/2,frame.size().height/2));
//		cv::resize(frame_right,frame_down_right,Size(frame_right.size().width/2,frame_right.size().height/2));
//
//		remap(frame_down, frame_rectify, mx1, my1, CV_INTER_LINEAR);
//		remap(frame_down_right, frame_rectify_right, mx2, my2, CV_INTER_LINEAR);
//#else
//		remap(frame, frame_rectify, mx1, my1, CV_INTER_LINEAR);
//		remap(frame_right, frame_rectify_right, mx2, my2, CV_INTER_LINEAR);
//#endif
//
//
//			//draw a few lines to check epipolar
//			for(int j = 0; j < FRAME_WIDTH; j+=50)
//			{
//#ifdef CAMERA_FLIP
//				line(frame_rectify,Point(0,j), Point(FRAME_HEIGHT,j), Scalar(255,0,0));
//				line(frame_rectify_right,Point(0,j), Point(FRAME_HEIGHT,j), Scalar(255,0,0));
//#else
//				line(frame_rectify,Point(0,j), Point(FRAME_WIDTH,j), Scalar(255,0,0));
//				line(frame_rectify_right,Point(0,j), Point(FRAME_WIDTH,j), Scalar(255,0,0));
//#endif
//			}
//			imshow("camera_left",frame_rectify);
//			imshow("camera2_right",frame_rectify_right);
//		}
//		else
//		{
//			for(int j = 0; j < FRAME_WIDTH; j+=50)
//			{
//				line(frame,Point(0,j), Point(FRAME_HEIGHT,j), Scalar(255,0,0));
//				line(frame_right,Point(0,j), Point(FRAME_HEIGHT,j), Scalar(255,0,0));
//			}
//			imshow("camera_left",frame);
//			imshow("camera2_right",frame_right);
//		}
//
//		if(waitKey(30) == 27) break;
//	}
//#endif
//
//	FileStorage fs("./save_param/calib_para.yml", CV_STORAGE_WRITE);
//	fs << "MX1" << mx1 << "MX2" << mx2 << "MY1" << my1 << "MY2" << my2;
//	fs.release();
//
//	VI.stopDevice(0);
//	VI.stopDevice(1);
//
//	delete[] image_calib_list;
//	return 0;
//}
//
//
//void StereoCalib(Mat * imagelist, Size boardSize,int number_calib_image,float squareSize, Mat *mx1, Mat *mx2, Mat *my1, Mat *my2 ,Rect *validRoi, bool useCalibrated, bool showRectified,float *error)
//{
//
//	bool displayCorners = true;//true;
//	const int maxScale = 2;
//	//float squareSize = 25.f;  // Set this to your actual square size
//	// ARRAY AND VECTOR STORAGE:
//#ifdef FISHEYE
//	//fiseeye calib requires 64 double
//	vector<vector<Point2d> > imagePoints[2];
//	vector<vector<Point3d> > objectPoints;
//	//drawing function require 32f
//	vector<vector<Point2f> > imagePoints_32f[2];
//#else
//	vector<vector<Point2f> > imagePoints[2];
//	vector<vector<Point3f> > objectPoints;
//#endif
//	Size imageSize;
//
//	int i, j, k, nimages = number_calib_image/2;
//
//	imagePoints[0].resize(nimages);
//	imagePoints[1].resize(nimages);
//#ifdef FISHEYE
//	imagePoints_32f[0].resize(nimages);
//	imagePoints_32f[1].resize(nimages);
//#endif
//
//	for( i = j = 0; i < nimages; i++ )
//	{
//		for( k = 0; k < 2; k++ )
//		{
//			/*            const string& filename = imagelist[i*2+k];*/
//			Mat img = imagelist[i*2+k];
//			if(img.empty())
//				break;
//			if( imageSize == Size() )
//				imageSize = img.size();
//			else if( img.size() != imageSize )
//			{
//				// cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
//				break;
//			}
//			bool found = false;
//#ifdef FISHEYE
//			vector<Point2d>& corners = imagePoints[k][j];
//			vector<Point2f>& corners_32f = imagePoints_32f[k][j];
//#else
//			vector<Point2f>& corners = imagePoints[k][j];
//#endif
//			for( int scale = 1; scale <= maxScale; scale++ )
//			{
//				Mat timg;
//				if( scale == 1 )
//					timg = img;
//				else
//					resize(img, timg, Size(), scale, scale);
//				found =findCirclesGrid(timg, boardSize, corners, CALIB_CB_ASYMMETRIC_GRID );
//				if( found )
//				{
//					if( scale > 1 )
//					{
//						Mat cornersMat(corners);
//						cornersMat *= 1./scale;
//					}
//					break;
//				}
//			}
//			if( displayCorners )
//			{
//				Mat cimg, cimg1;
//				cvtColor(img, cimg, CV_GRAY2BGR);
//#ifdef FISHEYE
//				
//				for(int i = 0; i<(int)corners.size();i++)
//				{
//					corners_32f.push_back((Point2f)corners[i]);
//				}
//				drawChessboardCorners(cimg, boardSize, corners_32f, found);
//#else
//				drawChessboardCorners(cimg, boardSize, corners, found);
//#endif
//
//				double sf = (double)640.0/MAX(img.rows, img.cols);
//				resize(cimg, cimg1, Size(), sf, sf);
//				imshow("corners", cimg1);
//				char c = (char)waitKey(1);
//				if( c == 27 || c == 'q' || c == 'Q' ) //Allow ESC to quit
//					exit(-1);
//			}
//			else
//				putchar('.');
//			if( !found )
//				break;
//		}
//		if( k == 2 )
//		{
//			j++;
//		}
//	}
//	cout << j << " pairs have been successfully detected.\n";
//	nimages = j;
//	if( nimages < 2 )
//	{
//		cout << "Error: too little pairs to run the calibration\n";
//		return;
//	}
//
//	imagePoints[0].resize(nimages);
//	imagePoints[1].resize(nimages);
//	objectPoints.resize(nimages);
//
//	for( i = 0; i < nimages; i++ )
//	{
////		for( j = 0; j < boardSize.height; j++ )
////			for( k = 0; k < boardSize.width; k++ )
////#ifdef FISEHEYE
////				objectPoints[i].push_back(Point3d((2*k+j%2)*squareSize, j*squareSize, 0));
////#else
////				objectPoints[i].push_back(Point3f((2*k+j%2)*squareSize, j*squareSize, 0));
////#endif
//			for( k = 0; k < boardSize.width; k++ )
//				for( j = 0; j < boardSize.height; j++ ){
//			
//#ifdef FISEHEYE
//				objectPoints[i].push_back(Point3d((2*k+j%2)*squareSize, j*squareSize, 0));
//#else
//				objectPoints[i].push_back(Point3f((2*k+j%2)*squareSize, j*squareSize, 0));
//#endif
//			}
//	}
//
//	cout<<"object pont:"<<objectPoints[0]<<endl;
//
//	cout << "Running stereo calibration ...\n";
//
//	Mat cameraMatrix[2], distCoeffs[2];
//	cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
//	cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
//	Mat R, T, E, F;
//#ifdef FISHEYE
//	cv::Vec4d D;
//	std::vector<cv::Vec3d> rvecs;
//    std::vector<cv::Vec3d> tvecs;
//#else
//	vector<Mat> rvecs,tvecs;
//#endif
//	Mat R11,R22,T11,T22;
//
//	
//#ifdef FISHEYE
//	/*cv::fisheye::calibrate(objectPoints, imagePoints[0], imageSize, cameraMatrix[0],
//		distCoeffs[0], rvecs, tvecs, fisheye::CALIB_RECOMPUTE_EXTRINSIC + fisheye::CALIB_CHECK_COND + fisheye::CALIB_FIX_SKEW ,TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5));
//*/
//	int flag_1 = 0;
//    flag_1 |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
//    flag_1 |= cv::fisheye::CALIB_CHECK_COND;
//    flag_1 |= cv::fisheye::CALIB_FIX_SKEW;
//
//	/*cv::fisheye::calibrate(objectPoints, imagePoints[0], imageSize, cameraMatrix[0],
//		distCoeffs[0], rvecs, tvecs, flag_1 ,TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 20, 1e-6));*/
//	cv::fisheye::calibrate(objectPoints, imagePoints[0], imageSize, cameraMatrix[0],
//		D, rvecs, tvecs, flag_1 ,TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 20, 1e-6));
//#else
//	calibrateCamera(objectPoints, imagePoints[0], imageSize, cameraMatrix[0],
//		distCoeffs[0], rvecs, tvecs, CV_CALIB_FIX_K3|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5,TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5));
//#endif
//	Rodrigues(rvecs[0],R11);
//	//tvecs[0].copyTo(T11);
//	//cout<<" cameramatrix 0 0 "<<cameraMatrix[0].at<double>(0,0)<<endl;
//	//cout<<" dist 0 "<<distCoeffs[0].at<double>(0,0)<<endl;
//	//cout<<" dist 1 "<<distCoeffs[0].at<double>(0,1)<<endl;
//	//cout<<" dist 2 "<<distCoeffs[0].at<double>(0,2)<<endl;
//	//cout<<" dist 3 "<<distCoeffs[0].at<double>(0,3)<<endl;
//
//#ifdef FISHEYE
//	int flag_2 = 0;
//    flag_2 |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
//    flag_2 |= cv::fisheye::CALIB_CHECK_COND;
//    flag_2 |= cv::fisheye::CALIB_FIX_SKEW;
//	cv::fisheye::calibrate(objectPoints, imagePoints[1], imageSize, cameraMatrix[1],
//		distCoeffs[1], rvecs, tvecs, flag_2,TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 20, 1e-6));
//#else
//	calibrateCamera(objectPoints, imagePoints[1], imageSize, cameraMatrix[1],
//		distCoeffs[1], rvecs, tvecs, CV_CALIB_FIX_K3|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5,TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-6));
//#endif
//	Rodrigues(rvecs[0],R22);
//	//tvecs[0].copyTo(T22);
//
//
////#ifdef FISHEYE
//	//Mat undist_left, undist_right;
//	//fisheye::undistortImage(imagelist[0], undist_left, cameraMatrix[0],distCoeffs[0],cameraMatrix[0]);
//	//namedWindow("undistort_left",0);
//	//imshow("undistort_left",undist_left);
//	//waitKey();
//
//	//vector<vector<Point2d> > imagePoints_undist[2];
//	//imagePoints_undist[0].resize(nimages);
//	//imagePoints_undist[1].resize(nimages);
//
//	//for(int i = 0; i < imagePoints[0].size(); i++)
//	//{
//	//	fisheye::undistortPoints(imagePoints[0][i], imagePoints_undist[0][i], cameraMatrix[0], distCoeffs[0]);
//	//	fisheye::undistortPoints(imagePoints[1][i], imagePoints_undist[1][i], cameraMatrix[1], distCoeffs[1]);
//	//}
//
////#endif
//
//	//cameraMatrix[0].at<double>(0,0) += 300;
//	//cameraMatrix[0].at<double>(1,1) += 300;
//
//	//cameraMatrix[1].at<double>(0,0) += 300;
//	//cameraMatrix[1].at<double>(1,1) += 300;
//#ifdef FISHEYE
//		double rms = cv::fisheye::stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
//		cameraMatrix[0], distCoeffs[0],
//		cameraMatrix[1], distCoeffs[1],
//		imageSize, R, T,
//		fisheye::CALIB_FIX_INTRINSIC + fisheye::CALIB_USE_INTRINSIC_GUESS + fisheye::CALIB_CHECK_COND + fisheye::CALIB_FIX_SKEW ,
//		TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,200, 1e-5));
//		/*fisheye::CALIB_FIX_INTRINSIC + fisheye::CALIB_USE_INTRINSIC_GUESS +  fisheye::CALIB_FIX_SKEW ,
//		TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,200, 1e-5));*/
//#else
//	double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
//		cameraMatrix[0], distCoeffs[0],
//		cameraMatrix[1], distCoeffs[1],
//		imageSize, R, T, E, F,
//		TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,200, 1e-5),
//		CV_CALIB_FIX_INTRINSIC  +
//		CV_CALIB_RATIONAL_MODEL +CV_CALIB_FIX_K1 + CV_CALIB_FIX_K2+
//		CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5);
//	// 					CV_CALIB_FIX_INTRINSIC  +CV_CALIB_RATIONAL_MODEL
//	//                     );
//#endif
//	cout << "done with RMS error=" << rms << endl;
//	error[0] = rms;
//	//cout<<" cameramatrix 0 0 "<<cameraMatrix[0].at<double>(0,0)<<endl;
//	//cout<<" dist 0 "<<distCoeffs[0].at<double>(0,0)<<endl;
//	//cout<<" R "<<R.at<double>(0,0)<<endl;
//	//cout<<" T "<<T.at<double>(0,0)<<endl;
//
//	//FileStorage fs_rt("./save_param/rt_vectors.yml", CV_STORAGE_WRITE);
//	//if( fs_rt.isOpened() )
//	//{
//	//	fs_rt << "R1" << R11 << "T1" << T11 << "R2" << R22 << "T2" << T22;
//	//	fs_rt.release();
//	//}
//
//
//
//#ifndef FISHEYE
//	// CALIBRATION QUALITY CHECK
//	double err = 0;
//	int npoints = 0;
//	vector<Vec3f> lines[2];
//	for( i = 0; i < nimages; i++ )
//	{
//		int npt = (int)imagePoints[0][i].size();
//		Mat imgpt[2];
//		for( k = 0; k < 2; k++ )
//		{
//			imgpt[k] = Mat(imagePoints[k][i]);
//			undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
//			computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
//		}
//		for( j = 0; j < npt; j++ )
//		{
//			double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
//				imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
//				fabs(imagePoints[1][i][j].x*lines[0][j][0] +
//				imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
//			err += errij;
//		}
//		npoints += npt;
//	}
//	cout << "average reprojection err = " <<  err/npoints << endl;
//	error[1] = err/npoints;
//#endif
//	// save intrinsic parameters
//	FileStorage fs("./save_param/intrinsics.yml", CV_STORAGE_WRITE);
//	if( fs.isOpened() )
//	{
//
//#ifdef DOWN_SAMPLE
//		fs<<"image_width" << FRAME_WIDTH/2;
//        fs<<"image_height" << FRAME_HEIGHT/2;
//#else
//		fs<<"image_width" << FRAME_WIDTH;
//        fs<<"image_height" << FRAME_HEIGHT;
//#endif
//		fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
//			"M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
//		fs.release();
//	}
//	else
//		cout << "Error: can not save the intrinsic parameters\n";
//
//	Mat R1, R2, P1, P2, Q;
//	/*    Rect validRoi[2];*/
//
//	std::cout<<"camera matrix 0 "<<cameraMatrix[0]<<endl;
//	std::cout<<"camera matrix 1"<<cameraMatrix[1]<<endl;
//	//std::cout<<"dist 0"<<distCoeffs[0]<<endl;
//	//std::cout<<"dist 1"<<distCoeffs[1]<<endl;
//
//#ifdef FISHEYE
//	cv::fisheye::stereoRectify(cameraMatrix[0], distCoeffs[0],
//		cameraMatrix[1], distCoeffs[1],
//		imageSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY,
//		imageSize);
//#else
//	stereoRectify(cameraMatrix[0], distCoeffs[0],
//		cameraMatrix[1], distCoeffs[1],
//		imageSize, R, T, R1, R2, P1, P2, Q,CALIB_ZERO_DISPARITY,
//		-1, imageSize, &validRoi[0], &validRoi[1]);
//#endif
//
//	fs.open("./save_param/validRoi.yml", CV_STORAGE_WRITE);
//	if( fs.isOpened() )
//	{
//		Mat temp1,temp2;
//		temp1.create(1,4,CV_32S);
//		temp2.create(1,4,CV_32S);
//
//		temp1.at<int>(0,0) = validRoi[0].x;
//		temp1.at<int>(0,1) = validRoi[0].y;
//		temp1.at<int>(0,2) = imageSize.width-(validRoi[0].x+validRoi[0].width-1);
//		temp1.at<int>(0,3) = imageSize.height-(validRoi[0].y+validRoi[0].height-1);
//		temp2.at<int>(0,0) = validRoi[1].x;
//		temp2.at<int>(0,1) = validRoi[1].y;
//		temp2.at<int>(0,2) = imageSize.width-(validRoi[1].x+validRoi[1].width-1);
//		temp2.at<int>(0,3) = imageSize.height-(validRoi[1].y+validRoi[1].height-1);
//		fs << "validRoi1" << temp1 << "validRoi2" << temp2;
//		fs.release();
//		temp1.release();
//		temp2.release();
//	}
//	//else
//		//cout << "Error: can not save the intrinsic parameters\n";
//
//
//	fs.open("./save_param/extrinsics.yml", CV_STORAGE_WRITE);
//	if( fs.isOpened() )
//	{
//		fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
//		fs.release();
//	}
//
//	//std::cout<<"R:"<<R<<endl;
//	//std::cout<<"T:"<<T<<endl;
//	//std::cout<<"P1"<<P1<<endl;
//	//else
//		//cout << "Error: can not save the intrinsic parameters\n";
//
//	// OpenCV can handle left-right
//	// or up-down camera arrangements
//	bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));
//
//	// COMPUTE AND DISPLAY RECTIFICATION
//	if( !showRectified )
//		return;
//
//	if( useCalibrated )
//	{
//		// we already computed everything
//	}
//
//	else
//		// use intrinsic parameters of each camera, but
//		// compute the rectification transformation directly
//		// from the fundamental matrix
//	{
//		vector<Point2f> allimgpt[2];
//		for( k = 0; k < 2; k++ )
//		{
//			for( i = 0; i < nimages; i++ )
//				std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
//		}
//		F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
//		//std::cout<<"Fundamental:"<<F<<endl;
//		Mat H1, H2;
//		stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);
//		//std::cout<<"H1:"<<H1<<endl;
//		//std::cout<<"H2"<<H2<<endl;
//
//		R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
//		R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
//		P1 = cameraMatrix[0];
//		P2 = cameraMatrix[1];
//	}
//
//	//Precompute maps for cv::remap()
//#ifdef FISHEYE
//	cv::fisheye::initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, mx1[0], my1[0]);
//	cv::fisheye::initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, mx2[0], my2[0]);
//#else
//	initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, mx1[0], my1[0]);
//	initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, mx2[0], my2[0]);
//#endif
//
//}
//
