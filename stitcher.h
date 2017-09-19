#ifndef STITCHER_H
#define STITCHER_H

#include "opencv2/opencv.hpp"
#include <fstream>
#include <iostream>

class Stitcher{

public:
	Stitcher(int pano_width, int pano_height, int type);
	~Stitcher();
	bool stitch(cv::Mat &previous_frame, cv::Mat &current_frame, cv::Mat &bg_mask);
	void set_frame_type(cv::Mat &f);
	cv::Mat& get_pano();
	void reset_pano();
	char get_pano_status(){
		return pano_status;
	}
	void set_background_image(cv::Mat& bg_image){
		m_background_image = bg_image;
	}


private:
	cv::Mat m_final_pano;
	cv::Mat m_slice;
	cv::Mat m_background_image;

	// pano properties
	int pano_width;
	int pano_height;
	int pano_type;
	int pano_right;
	int pano_top = 400;
	int idle_counter = 0;
	bool car_counter = 0;
	char pano_status = 0;


	// delta_x delta_y
	int delta_x;
	int delta_y;
	int prev_delta_x;
	int prev_delta_y;
	
	// ROI Properties
	float template_roi_width_mult= 0.40;
	float template_roi_height_mult = 0.7;
	 
	//const cv::Mat SE3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));


};


#endif