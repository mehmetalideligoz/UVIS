#ifndef MOTION_DETECTOR_H
#define MOTION_DETECTOR_H

#include <assert.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

const cv::Mat SE5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
const cv::Mat SE3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

class MotionDetector{

private:
	cv::Mat m_frame;
	cv::Mat m_binary_image;
	cv::Mat m_fgMaskMOG2;
	cv::Mat m_background_frame;
	cv::Ptr<cv::BackgroundSubtractor> m_pMOG2;
	

	int status = WAIT;
	double MIN_BR_HEIGHT = 300;
	double MIN_BR_WIDTH = 60;
	double MAX_BINARY_THRESHOLD = 255;
	double MIN_BINARY_THRESHOLD = 30;
	void read_config_file();

public:
	MotionDetector();
	~MotionDetector();
	bool MotionDetector::detect_motion();
	void set_frame(const cv::Mat &frame);
	int passive_frame_counter = 0;
	int get_passive_frame_counter(){
		return passive_frame_counter;
	}
	int get_motion_status(){
		return status;
	}
	enum MotionStatus{
		WAIT,
		START,
		CONTD,
		FINISH,
	};
	cv::Mat get_bg_mask(){
		return m_binary_image;
	}




};

#endif