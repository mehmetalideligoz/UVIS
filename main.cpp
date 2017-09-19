/*
	main.cpp


*/

#include <iostream>
#include "stitcher.h"
#include "motion_detector.h"

// Utility functions for rotation
void rotate90(cv::Mat &frame){
	cv::flip(frame.t(), frame, 1);
}
void rotate270(cv::Mat &frame){
	cv::flip(frame.t(), frame, 0);
	
}



int main(){
	bool passive_frames = 0;
	cv::Mat current_frame;
	cv::Mat previous_frame;

	std::string video_path = "C:/Users/bunyaminA2/Desktop/06.07.2017/";
	std::string filename = "4831701330404"; // 4831539749755 -4831701330404 -4831590745528

	cv::VideoCapture cap(video_path + filename + ".avi");
	if (!cap.isOpened()){
		std::cout << "Cannot open the video file" << std::endl;
		return -1;
	}
	cv::namedWindow("m_final_pano", CV_WINDOW_KEEPRATIO);
	cv::namedWindow("current_frame", CV_WINDOW_KEEPRATIO);

	// Rotate frames either 90 degree or 270 degree, according to car's direction
	cap >> previous_frame;
	rotate90(previous_frame); //rotate270(previous_frame);

	MotionDetector *md = new MotionDetector();
	Stitcher *st = new Stitcher(8500,3500,previous_frame.type()); // create stitcher with 8000x3500 pano

	// Set the first frame as background image
	st->set_background_image(previous_frame);

	// Skip frames
	while (cap.get(CV_CAP_PROP_POS_FRAMES) < 550 ) cap >> previous_frame;
	rotate90(previous_frame);	//rotate270(previous_frame);
	md->set_frame(previous_frame);


	// Start capturing
	while (true){
		std::cout << "\r frame: " << cap.get(CV_CAP_PROP_POS_FRAMES) << std::flush;

		cap >> current_frame;
		if (current_frame.empty())
			break;
		
		rotate90(current_frame); //rotate270(current_frame);
		cv::Mat cf;
		current_frame.copyTo(cf);

		md->set_frame(current_frame);

		// Start stitching if motion is detected
		if (md->detect_motion()){
			st->stitch(previous_frame, current_frame, md->get_bg_mask());
		}

		// if pano has something on it, and there is no motion
		else if (st->get_pano_status() == 1 && md->get_motion_status() == md->WAIT){
			cv::imwrite(filename + "_" +std::to_string(cap.get(CV_CAP_PROP_POS_FRAMES)) + ".jpg", st->get_pano());
			st->reset_pano();
		}

		cv::imshow("m_final_pano", st->get_pano());
		cv::imshow("current_frame", current_frame);
		cv::imshow("current_frame", previous_frame);
		previous_frame = cf;

		if (cv::waitKey(50) >= 0) break;
	}	

	system("pause");
	return 0;
}