/*
	stitcher.cpp
	Image stitching using template matching

	*/
#include "stitcher.h"


Stitcher::Stitcher(int pano_width, int pano_height, int type){
	this->pano_width = pano_width;
	this->pano_height = pano_height;
	this->pano_type = type;
	pano_right = pano_width;
	m_final_pano = cv::Mat::zeros(cv::Size(pano_width, pano_height), type);

}

Stitcher::~Stitcher(){
}


/*
	reset_pano:
		- reset old values after each car
*/
void Stitcher::reset_pano(){
	if (pano_status){
		std::cout << "Resetting pano" << std::endl;
		m_final_pano = cv::Mat::zeros(cv::Size(pano_width, pano_height), pano_type);
		pano_top = 400;
		pano_right = 8000;
		pano_status = 0;
		delta_x = 0;
		prev_delta_x = 0;
		delta_y = 0;
		prev_delta_y = 0;
	}
}


/*
*	stitch:
*		- takes two consecutive frames and optionally bg_mask as parameters
*		- stitches two frames using template matching 
*
*
*/
bool Stitcher::stitch(cv::Mat &previous_frame, cv::Mat &current_frame, cv::Mat &bg_mask){

	if (previous_frame.empty() || current_frame.empty()){
		std::cout << "Empty frames" << std::endl;
		return false;
	}

	// Due to illumination, region of interest for template matching should be close to the middle. 
	float x_roi = previous_frame.cols * 0.35;
	float y_roi = previous_frame.rows * 0.01;


	cv::Rect roi_rect(x_roi, y_roi,
		previous_frame.cols * template_roi_width_mult, previous_frame.rows* template_roi_height_mult);
	cv::Mat template_image = previous_frame(roi_rect);

	// Create the result matrix
	int result_cols = current_frame.cols - template_image.cols + 1;
	int result_rows = current_frame.rows - template_image.rows + 1;
	cv::Mat result(result_cols, result_rows, CV_32FC1);

	// Search template in right side of the roi (assuming that the car only moves forward)
	cv::Rect match_roi(x_roi, y_roi, current_frame.cols - x_roi, current_frame.rows - y_roi);

	// Do the Matching and Normalize
	cv::matchTemplate(current_frame(match_roi), template_image, result, CV_TM_CCOEFF_NORMED);
	cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

	// Localizing the best match with minMaxLoc
	double minVal, maxVal;
	cv::Point matchLoc, minLoc, maxLoc;
	cv::minMaxLoc(result(cv::Rect(0,0,result.cols,3)), &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

	// minLoc for CV_TM_SQDIFF_NORMED
	matchLoc = maxLoc;

	// Calculate displacement in x and y direction,
	int delta_x = matchLoc.x; 
	int delta_y = matchLoc.y ;

	// Template matching is not useful here, use another method
	float tm_confidence_thresh = 0.3;
	if (minVal < tm_confidence_thresh){

		cv::Mat gray, prevGray;
		cv::cvtColor(previous_frame,prevGray, CV_BGR2GRAY);
		cv::cvtColor(current_frame, gray, CV_BGR2GRAY);
		std::vector<cv::Point2f> PreviousFrameFeatures;
		cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);

		if (prevGray.empty() == false)
		{
			cv::goodFeaturesToTrack(prevGray, PreviousFrameFeatures, 500, 0.01, 3, cv::Mat(), 3, 0, 0.04);

			std::vector<uchar> status;
			std::vector<float> err;
			std::vector<cv::Point2f> CurrentFrameFeatures;
			cv::calcOpticalFlowPyrLK(prevGray, gray, PreviousFrameFeatures, CurrentFrameFeatures, status, err, cv::Size(31,31), 3, termcrit, 0, 0.001);

			std::vector<int> deltas;

			// Filter bad LK Points
			for (size_t index = 0; index < PreviousFrameFeatures.size(); index++)
			{
				cv::Point2f & PreviewPoint = PreviousFrameFeatures.at(index);
				cv::Point2f & CurrentPoint = CurrentFrameFeatures.at(index);

				if (status[index] 
					&& CurrentPoint.x >= 0
					&& CurrentPoint.y >= 0 
					&& CurrentPoint.x < gray.cols 
					&& CurrentPoint.y < gray.rows
					&& CurrentPoint.x >= PreviewPoint.x
					&& abs(CurrentPoint.y - PreviewPoint.y) < 5) 
				{
					deltas.push_back(CurrentPoint.x - PreviewPoint.x);
				}

			}
		
			// -Each LK prediction votes for different delta_x value, therefore it uses the median as delta_x  
			// -Reliable LK calculation requires more than "30" votes
			double median = 0;
			if (deltas.size() > 30){
				std::sort(deltas.begin(), deltas.end());
				if (deltas.size() % 2 == 0)
				{
					median = (deltas[deltas.size() / 2 - 1] + deltas[deltas.size() / 2]) / 2;
				}
				else
				{
					median = deltas[deltas.size() / 2];
				}

				delta_x = median;
			}
		}	
	}

	// Store previous deltas
	prev_delta_x = delta_x;
	prev_delta_y = delta_y;

	// Slice from middle of the current frame with width of delta_x 
	float x_slice_location = current_frame.cols * 0.50;
	cv::Rect slice_rect(x_slice_location, 0, delta_x, current_frame.rows);
	m_slice = current_frame(slice_rect);
	
	if (pano_right - m_slice.cols <= 0){
		std::cout << "Pano is full" << std::endl;
		return false;
	}

	if (m_slice.rows <= 0 && m_slice.cols <= 0){
		std::cout << "Template matching failed, empty slice" << std::endl;
		return false;
	}

	// Remove background: Using the output of MOG2 mask does the same thing. 
	// If the difference of values in both 3 channels less than	threshold values remove the pixel.  
	int R_thresh = 20;
	int G_thresh = 20;
	int B_thresh = 20;

	for (size_t y = 0; y < m_slice.rows; y++){
		for (size_t x = 0; x < m_slice.cols ; x++){

				int b_diff = m_slice.at<cv::Vec3b>(y, x)[0] - m_background_image(slice_rect).at<cv::Vec3b>(y, x)[0];
				int g_diff = m_slice.at<cv::Vec3b>(y, x)[1] - m_background_image(slice_rect).at<cv::Vec3b>(y, x)[1];
				int r_diff = m_slice.at<cv::Vec3b>(y, x)[2] - m_background_image(slice_rect).at<cv::Vec3b>(y, x)[2];

				if (abs(b_diff) < B_thresh && abs(g_diff) < G_thresh && abs(r_diff) < R_thresh){
					m_slice.at<cv::Vec3b>(y, x)[0] = 0;
					m_slice.at<cv::Vec3b>(y, x)[1] = 0;
					m_slice.at<cv::Vec3b>(y, x)[2] = 0;
				}
		}
	}

	m_slice.copyTo(m_final_pano(cv::Rect(pano_right - m_slice.cols, pano_top, m_slice.cols, m_slice.rows)));
	pano_status = 1; // indicating that pano is not empty

	// Interpolate colors on the stitching lines.
	for (size_t index_y = pano_top; index_y < m_slice.rows - 1; index_y++){
		for (size_t index_x = pano_right - 1; index_x < pano_right + 1; index_x++){

			m_final_pano.at<cv::Vec3b>(index_y, index_x)[0] = (m_final_pano.at<cv::Vec3b>(index_y, index_x + 1)[0] + m_final_pano.at<cv::Vec3b>(index_y, index_x - 1)[0]) / 2;
			m_final_pano.at<cv::Vec3b>(index_y, index_x)[1] = (m_final_pano.at<cv::Vec3b>(index_y, index_x + 1)[1] + m_final_pano.at<cv::Vec3b>(index_y, index_x - 1)[1]) / 2;
			m_final_pano.at<cv::Vec3b>(index_y, index_x)[2] = (m_final_pano.at<cv::Vec3b>(index_y, index_x + 1)[2] + m_final_pano.at<cv::Vec3b>(index_y, index_x - 1)[2]) / 2;
		}
	}

	// Update stitching locations
	pano_top = pano_top - delta_y;
	pano_right = pano_right - delta_x;

	return true;
}

cv::Mat& Stitcher::get_pano(){
	return m_final_pano;
}


