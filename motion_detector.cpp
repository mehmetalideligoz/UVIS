#include "motion_detector.h"




MotionDetector::MotionDetector(){
	m_pMOG2 = cv::createBackgroundSubtractorMOG2(100, 10, false);
}

MotionDetector::~MotionDetector(){


}


/*
*	detect_motion:
*
*/

bool MotionDetector::detect_motion(){
	assert(!m_frame.empty());
	

	// 1. Apply MoG2 BG subtraction
	cv::cvtColor(m_frame, m_frame, CV_BGR2GRAY);
	cv::resize(m_frame, m_frame, cv::Size(m_frame.cols / 4, m_frame.rows / 4));

	m_pMOG2->apply(m_frame, m_fgMaskMOG2,0.001); // learning rate: 0.001 ~=105 frames
	m_binary_image = m_fgMaskMOG2;

	cv::imshow("thresh", m_binary_image);
		

	// 2. Find contours
	std::vector< std::vector< cv::Point> > contours;
	cv::findContours(m_binary_image, contours,
		CV_RETR_EXTERNAL, // retrieve the external contours
		CV_CHAIN_APPROX_SIMPLE); // all pixels of each contours
	std::vector< std::vector< cv::Point> >::iterator itc = contours.begin();

	// If motion detected ("large" bounding box detected): update status START 
	while (itc != contours.end()) {
		cv::Rect br = cv::boundingRect(cv::Mat(*itc));
		
		if (br.height > MIN_BR_HEIGHT 
			&& br.width > MIN_BR_WIDTH)
		{
			status = START;
			passive_frame_counter = 0;
			return true;
		}

		++itc;
	}

	// If there are more contours than threshold=10: update status CONTD 
	if (status == START || status == CONTD){
		if (contours.size() > 10){
			status = CONTD;
			return true;
		}
	}
	
	// If there is not sufficient contours for "20" frames, update status WAIT
	passive_frame_counter++;
	if (passive_frame_counter >= 20){
		status = WAIT;
		return false;
	}

	return false;
}


void MotionDetector::set_frame(const cv::Mat &frame){
	m_frame = frame;
}

