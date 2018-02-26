#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include <opencv2/features2d/features2d.hpp>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <time.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "opencv2/calib3d/calib3d.hpp"
#include <boost/math/distributions/normal.hpp>
#include <boost/random.hpp>
#include <pangolin/pangolin.h>
#include <algorithm>
class GridHarrisFast{
public:
	GridHarrisFast(){

	}
	GridHarrisFast(cv::Size img_size, int grid_size){
		col = img_size.width;
		row = img_size.height;
		grid_size_ = grid_size;
		margin_c_ = col%grid_size;
		if (margin_c_ < 20) //TODO 20 can be tuned
			margin_c_ += grid_size;
		grid_x_ = (col - margin_c_)/grid_size;
		margin_c_ /= 2;
		margin_r_ = row%grid_size;
		if (margin_r_ < 20)
			margin_r_ += grid_size;
		grid_y_ = (row - margin_r_)/grid_size;
		margin_r_ /= 2;
		fast_thres_ = 12;
	}

	void detect(cv::Mat &src, std::vector<std::vector<cv::KeyPoint> > &kps, std::vector<cv::Point2f> &old_pts){
		int index = 0;
		// std::cout << "size: " << grid_x_*grid_y_ << std::endl;
		double t = (double)cv::getTickCount();

		std::vector<cv::KeyPoint> temp_kp;
		cv::FAST(src, temp_kp, fast_thres_, true);
		kps.resize(grid_x_*grid_y_);

		for(int i=0; i<temp_kp.size(); ++i){
			int x = temp_kp[i].pt.x;
			int y = temp_kp[i].pt.y;
			if(x+margin_c_+1>col || x-margin_c_<0 || y+margin_r_+1>row || y-margin_r_<0)
				continue;
			int x_index = (x-margin_c_)/grid_size_;
			int y_index = (y-margin_r_)/grid_size_;
			kps[y_index*grid_x_ + x_index].push_back(temp_kp[i]);
		}


		for(int i=0; i<old_pts.size(); ++i){
			int x = old_pts[i].x;
			int y = old_pts[i].y;
			if(x+margin_c_+1>col || x-margin_c_<0 || y+margin_r_+1>row || y-margin_r_<0)
				continue;
			int x_index = (x-margin_c_)/grid_size_;
			int y_index = (y-margin_r_)/grid_size_;
			kps[y_index*grid_x_ + x_index].clear();
		}
		for(int i=0; i<old_pts.size(); ++i){
			int x = old_pts[i].x;
			int y = old_pts[i].y;
			int x_index = (x-margin_c_)/grid_size_;
			int y_index = (y-margin_r_)/grid_size_;

			int square = grid_size_*grid_size_/9; //TODO 9 can be tuned
			for(int j=-1; j<2; ++j){
				for(int k=-1; k<2; ++k){
					if(x_index+j<0 || x_index+j>=grid_x_ || y_index+k<0 || y_index+k>=grid_y_)
						continue;
					std::vector<cv::KeyPoint> &chosed = kps[(y_index+k)*grid_x_+x_index+j];
					for(auto iter=chosed.begin(); iter!=chosed.end();){
						int dx = iter->pt.x - x;
						int dy = iter->pt.y - y;
						if(dx*dx+dy*dy < square){
							iter = chosed.erase(iter);
						}else{
							++iter;
						}
					}
				}
			}
		}

		shiTomasiScore(src, kps);

		t = ((double)cv::getTickCount() - t)/cv::getTickFrequency(); 
  		std::cout << "Times passed in seconds for detect: " << t << std::endl;
	}

	void draw_grid(cv::Mat &src){
		for(int x=margin_c_, i=0; i<=grid_x_; x+=grid_size_, ++i)
			line(src, cv::Point2f(x, 0), cv::Point2f(x, src.rows-1), CV_RGB(255, 0, 0), 1);
		for(int y=margin_r_, j=0; j<=grid_y_; y+=grid_size_, ++j){
			line(src, cv::Point2f(0, y), cv::Point2f(src.cols-1, y), CV_RGB(255, 0, 0), 1);
		}
	}
	void draw_feature(cv::Mat &src, std::vector<std::vector<cv::KeyPoint> > &kps){
		for(int i=0; i<kps.size(); ++i)
			for(int j=0; j<kps[i].size(); ++j)
				circle(src, kps[i][j].pt, 2, CV_RGB(0, 255, 0), 1);
	}
	void draw_feature(cv::Mat &src, std::vector<cv::Point2f> &max_pt, bool hh = false){
		for(int i=0; i<max_pt.size(); ++i)
			if(!hh)
				circle(src, max_pt[i], 2, CV_RGB(255, 0, 0), 1);
			else
				circle(src, max_pt[i], 2, CV_RGB(255, 255, 255), 1);
	}

	void draw_feature(cv::Mat &src, std::vector<cv::Point2f> &max_pt, int radius){
		for(int i=0; i<max_pt.size(); ++i)
			circle(src, max_pt[i], radius, CV_RGB(255, 0, 0), 1);
	}

	void find_max(std::vector<std::vector<cv::KeyPoint> > &kps, std::vector<cv::Point2f> &max_pt){
		max_pt.clear();
		int max = -1;
		float x, y;
		for(int i=0; i<kps.size(); ++i){
			max = -1;
			for(int j=0; j<kps[i].size(); ++j){
				if(kps[i][j].response>max){
					max = kps[i][j].response;
					x = kps[i][j].pt.x;
					y = kps[i][j].pt.y;
				}
			}
			if(kps[i].size()>0)
				max_pt.push_back(cv::Point2f(x, y));
		}
	}

	// from vikit 
	void shiTomasiScore(const cv::Mat& img, std::vector<std::vector<cv::KeyPoint> > &kps)
	{
	  assert(img.type() == CV_8UC1);
	  float score;
	  for(int i=0; i<kps.size(); ++i){
	  	for(int j=0; j<kps[i].size(); ++j){
	  		int u = kps[i][j].pt.x;
	  		int v = kps[i][j].pt.y;
	  		float dXX = 0.0;
			float dYY = 0.0;
			float dXY = 0.0;
			const int halfbox_size = 4;
			const int box_size = 2*halfbox_size;
			const int box_area = box_size*box_size;
			const int x_min = u-halfbox_size;
			const int x_max = u+halfbox_size;
			const int y_min = v-halfbox_size;
			const int y_max = v+halfbox_size;

			if(x_min < 1 || x_max >= img.cols-1 || y_min < 1 || y_max >= img.rows-1)
				score = 0.0; // patch is too close to the boundary

			const int stride = img.step.p[0];
			for( int y=y_min; y<y_max; ++y ){
				const uint8_t* ptr_left   = img.data + stride*y + x_min - 1;
				const uint8_t* ptr_right  = img.data + stride*y + x_min + 1;
				const uint8_t* ptr_top    = img.data + stride*(y-1) + x_min;
				const uint8_t* ptr_bottom = img.data + stride*(y+1) + x_min;
				for(int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom){
					float dx = *ptr_right - *ptr_left;
					float dy = *ptr_bottom - *ptr_top;
					dXX += dx*dx;
					dYY += dy*dy;
					dXY += dx*dy;
				}
			}

			// Find and return smaller eigenvalue:
			dXX = dXX / (2.0 * box_area);
			dYY = dYY / (2.0 * box_area);
			dXY = dXY / (2.0 * box_area);
			score = 0.5 * (dXX + dYY - sqrt( (dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY) ));
			kps[i][j].response = score;
	  	}
	  }
	}

private:
	int grid_size_;
	int grid_x_;
	int grid_y_;
	int margin_c_;
	int margin_r_;
	int fast_thres_;
	int row;
	int col;
};
struct imu_measurement{
	imu_measurement(){
		memset(this,0,sizeof(imu_measurement));
	}

	imu_measurement(Eigen::Vector3d &gyro, Eigen::Vector3d &acc, double time){
		_gyro = gyro;
		_acc = acc;
		_time_stamp = time;
	}
	Eigen::Vector3d _gyro;
	Eigen::Vector3d _acc;
	double _time_stamp;
};

struct state{
	Eigen::Vector3d _position;
	Eigen::Vector3d _velocity;
	Eigen::Quaterniond _pose;
	Eigen::Vector3d _acc_bias;
	Eigen::Vector3d _gyro_bias;
	Eigen::Vector3d _gravity;
};

class frame;
class feature{
public:
	feature(){
		_visible_frame.clear();
		_location.clear();
	}
	void add_measurement(frame *frame_ptr, cv::Point2f &location){
		_visible_frame.push_back(frame_ptr);
		_location.push_back(location);
	}
	void delete_measurement(frame *frame_ptr){
		std::vector<frame*>::iterator iter = std::find(_visible_frame.begin(), _visible_frame.end(), frame_ptr);
		if(iter != _visible_frame.end()){
			_location.erase(_location.begin()+(iter-_visible_frame.begin()));
			_visible_frame.erase(iter);
		}
		else{
			std::cout << "fatal logical error, the code should not come here " << std::endl;
			exit(0);
		}
	}
	std::vector<frame*> _visible_frame;
	std::vector<cv::Point2f> _location;
	Eigen::Vector3d _position;
};

class camera_model{
public:
	camera_model(float fx, float fy, float cx, float cy, float k1, float k2, float p1, float p2){
		cv::Mat K = cv::Mat::eye(3,3,CV_32F);
	    K.at<float>(0,0) = fx;
	    K.at<float>(1,1) = fy;
	    K.at<float>(0,2) = cx;
	    K.at<float>(1,2) = cy;
	    K.copyTo(_K);
	    Eigen::Matrix3d temp;
	    temp << fx, 0, cx, 0, fy, cy, 0, 0, 1;
	    _K_inv = temp.inverse();

	    cv::Mat DistCoef(4,1,CV_32F);
	    DistCoef.at<float>(0) = k1;
	    DistCoef.at<float>(1) = k2;
	    DistCoef.at<float>(2) = p1;
	    DistCoef.at<float>(3) = p2;
	    DistCoef.copyTo(_DisCoef);
	}

	Eigen::Vector3d to3d(cv::Point2f &point){
		// Fill matrix with points
	    cv::Mat mat(1,2,CV_32F);
	    mat.at<float>(0,0)=point.x;
	    mat.at<float>(0,1)=point.y;

	    // Undistort points
	    mat=mat.reshape(2);
	    cv::undistortPoints(mat,mat,_K,_DisCoef,cv::Mat(),_K);
	    mat=mat.reshape(1);
	    Eigen::Vector3d return_vec;
	    return_vec << (double)(mat.at<float>(0,0)),double(mat.at<float>(0,1)),1;
	    return_vec = (_K_inv*return_vec).eval();
	    return_vec = (return_vec/return_vec(2)).eval();
	    return return_vec;
	}

	cv::Mat _K;
	cv::Mat _DisCoef;
	Eigen::Matrix3d _K_inv;
};

class frame{
public:
	// for normal frame
	frame(double time, cv::Mat &img){
		_time_stamp = time;
		static size_t INDEX = 0;
		_id = INDEX++;
		// if(_id==0){
			_detector = GridHarrisFast(img.size(), 100); //50 as grid_size
		// }
		_img = img.clone();
		_visible_feature.clear();
		_feature_location.clear();
	}

	void add_feature(feature *feature_ptr, cv::Point2f point_location){
		_visible_feature.push_back(feature_ptr);
		_feature_location.push_back(point_location);
	}

	void delete_feature(feature *feature_ptr){
		std::vector<feature*>::iterator iter = std::find(_visible_feature.begin(), _visible_feature.end(), feature_ptr);
		if(iter != _visible_feature.end()){
			_feature_location.erase(_feature_location.begin()+(iter-_visible_feature.begin()));
			_visible_feature.erase(iter);
		}
		else{
			std::cout << "fatal logical error, the code should not come here " << std::endl;
			exit(0);
		}
	}

	void detect_feature(){
		std::vector< std::vector<cv::KeyPoint> > kps;
		std::vector< cv::Point2f> max_pt;
		_detector.detect(_img, kps, _feature_location);
		_detector.find_max(kps, max_pt);
		std::cout << "max_pt: " << max_pt.size() << std::endl;
		for(int i=0; i<max_pt.size(); ++i){
			feature *new_feature = new feature;
			new_feature->add_measurement(this, max_pt[i]);
			add_feature(new_feature, max_pt[i]);
		}
	}

	GridHarrisFast _detector;
	double _time_stamp;
	size_t _id;
	cv::Mat _img;
	std::vector<feature*> _visible_feature;
	std::vector<cv::Point2f> _feature_location;
	Eigen::Matrix3d _r_propagated;
	Eigen::Matrix3d _r_updated;
	Eigen::Vector3d _t_propagated;
	Eigen::Vector3d _t_updated;
	Eigen::Matrix<double, 6, 6> _covariance;
};


class msckf{
public:
	msckf(double acc_noise, double acc_random, double gyro_noise, double gyro_random, Eigen::Matrix3d &r_imu_camera, Eigen::Vector3d &t_imu_camera, 
		state &inital_state, double initial_time, camera_model *camera){
		_window.clear();
		_acc_noise_density = acc_noise;
		_acc_random_walk = acc_random;
		_gyro_noise_density = gyro_noise;
		_gyro_random_walk = gyro_random;
		_r_imu_camera = r_imu_camera;
		_t_imu_camera = t_imu_camera;
		_state_updated = inital_state;
		_state_propagated = inital_state;
		_time_stamp = initial_time;
		_camera = camera;
		_covariance.resize(18,18);
		_covariance.setZero();
		_covariance.block(15,15,3,3) = Eigen::Matrix3d::Identity()*0.005; //initial noise for gravity
	}

	// from sophus
	Eigen::Matrix3d exp_map(Eigen::Vector3d omega){
		double theta;
		theta = omega.norm();
		double half_theta = 0.5*theta;
		double imag_factor;
		double real_factor = cos(half_theta);
		if(theta<1e-10){
			double theta_sq = theta*theta;
			double theta_po4 = theta_sq*theta_sq;
    		imag_factor = 0.5-0.0208333*theta_sq+0.000260417*theta_po4;
		}else{
			double sin_half_theta = sin(half_theta);
			imag_factor = sin_half_theta/theta;
		}

		return Eigen::Quaterniond(real_factor,
                         imag_factor*omega.x(),
                         imag_factor*omega.y(),
                         imag_factor*omega.z()).toRotationMatrix();
	}

	void halt_naive(){
		int n;
		std::cout << "cin number to continue" << std::endl;
		std::cin >> n;
	}

	void show_vector(std::string str, Eigen::VectorXd vec){
		std::cout << str << ": ";
		for(int i=0; i<vec.size(); ++i)
			std::cout << vec(i) << " ";
		std::cout << std::endl;
	}

	void show_matrix(std::string str, Eigen::MatrixXd mat){
		std::cout << str << std::endl << mat << std::endl;
	}

	void propagate(Eigen::Vector3d &gyro, Eigen::Vector3d &acc, double time){
		// for nominal
		state state_temp;
		double dt = time - _time_stamp;
		std::cout << "dt: " << dt << std::endl;
		Eigen::Vector3d gyro_unbias = gyro - _state_updated._gyro_bias;
		Eigen::Vector3d acc_unbias = acc - _state_updated._acc_bias;
		Eigen::Matrix3d rotation = _state_updated._pose.toRotationMatrix();
		Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
		Eigen::Matrix3d acc_cross;
		acc_cross << 0,-acc_unbias(2),acc_unbias(1),acc_unbias(2),0,-acc_unbias(0),-acc_unbias(1),acc_unbias(0),0;
		Eigen::Quaterniond qua_inc(1, gyro_unbias(0)*dt *0.5, gyro_unbias(1)*dt *0.5, gyro_unbias(2)*dt *0.5);
		state_temp._pose = _state_updated._pose * qua_inc;
		Eigen::Vector3d temp_val = _state_updated._pose * (acc - _state_updated._acc_bias) + _state_updated._gravity;
		state_temp._velocity = _state_updated._velocity + temp_val * dt;
		state_temp._position = _state_updated._position + state_temp._velocity * dt + temp_val * dt * dt * 0.5;
		state_temp._gravity = _state_updated._gravity;
		state_temp._gyro_bias = _state_updated._gyro_bias;
		state_temp._acc_bias = _state_updated._acc_bias;
		// std::cout << "position: " << state_temp._position << std::endl;
		_time_stamp = time;
		_state_propagated  = state_temp;
		_state_updated = state_temp;
		// show_vector("gyro_bias", gyro);
		// show_vector("gyro_unbias", gyro_unbias);
		// std::cout << "qua_inc: " << qua_inc.w() << " " << qua_inc.x() << " " << qua_inc.y() << " " << qua_inc.z() << std::endl;
		// std::cout << "state_temp._position: " << std::endl << state_temp._position << std::endl;
		// std::cout << "state_temp._velocity: " << std::endl << state_temp._velocity << std::endl;
		// std::cout << "state_temp._gravity: " << std::endl << state_temp._gravity << std::endl;
		// std::cout << "state_temp._pose: " << std::endl << state_temp._pose.toRotationMatrix() << std::endl;
		// for covariance
		Eigen::Matrix<double, 18, 18> jacobian = Eigen::Matrix<double, 18, 18>::Zero();
		jacobian.block(0,0,3,3) = identity;
		jacobian.block(0,3,3,3) = identity*dt;
		jacobian.block(3,3,3,3) = identity;
		jacobian.block(3,6,3,3) = -rotation*acc_cross*dt;
		jacobian.block(3,9,3,3) = -rotation*dt;
		jacobian.block(3,15,3,3) = identity*dt;
		jacobian.block(6,6,3,3) = exp_map(gyro_unbias*dt);
		jacobian.block(6,12,3,3) = -identity*dt;
		jacobian.block(9,9,3,3) = identity;
		jacobian.block(12,12,3,3) = identity;
		jacobian.block(15,15,3,3) = identity;
		if(_window.size() == 0){
			_covariance = (jacobian * _covariance * jacobian.transpose()).eval();
		}else{
			size_t size = _covariance.rows();
			Eigen::MatrixXd large_jacobian(size, size);
			large_jacobian.setZero();
			Eigen::MatrixXd iden(size-18, size-18);
			iden.setIdentity();
			large_jacobian.block(0,0,18,18) = jacobian;
			large_jacobian.block(18,18,size-18,size-18) = iden;
			_covariance = (large_jacobian * _covariance * large_jacobian.transpose()).eval();
		}

		//add noise: impulse and random walk
		_covariance.block(3,3,3,3) += identity*_acc_noise_density*dt*dt;
		_covariance.block(6,6,3,3) += identity*_gyro_noise_density*dt*dt;
		_covariance.block(9,9,3,3) += identity*_acc_random_walk*dt;
		_covariance.block(12,12,3,3) += identity*_gyro_random_walk*dt;

		_covariance = 0.5*(_covariance+_covariance.transpose().eval());
		// std::cout << "_covariance determinant: " << _covariance.determinant() << std::endl << _covariance << std::endl;
		// halt_naive();
	}

	void measure(double time, frame *frame_ptr){
		std::cout << "frame id: " << frame_ptr->_id << std::endl;
		// std::cout << "ok " << __LINE__ << std::endl;
		augumentation(frame_ptr);
		// std::cout << "ok " << __LINE__ << std::endl;
		std::vector<feature*> lost_feature;
		// std::cout << "ok " << __LINE__ << std::endl;
		tracking(lost_feature);
		std::cout << "ok " << __LINE__ << std::endl;
		if(lost_feature.size() > 0){
			// std::cout << "lost_feature size: " << lost_feature.size() << " " << __LINE__ << std::endl;
			// std::cout << "frame id: " << frame_ptr->_id << std::endl;
			update(lost_feature);
		}
		frame_ptr->detect_feature();
		adjust_window();
	}

	void adjust_window(){
		if(_window.size() < 15) // 15 can be tuned
			return;
		std::vector<bool> clear_flag(_window.size(), false);
		int clear_cnt = 0;
		for(int i=1; i<clear_flag.size()-3; ++i){
			if(i%3==0 || _window[i]->_visible_feature.size() == 0){
				clear_flag[i] = true;
				++clear_cnt;
			}
		}
		int new_size = _window.size() - clear_cnt;
		std::cout << "new size: " << new_size << std::endl;
		Eigen::MatrixXd new_cov(18+new_size*6, 18+new_size*6);
		new_cov.block(0,0,18,18) = _covariance.block(0,0,18,18);
		int index_begin = 18;
		std::cout << "ok " << __LINE__ << std::endl;
		for(int i=0; i<_window.size(); ++i){
			if(clear_flag[i])
				continue;
			new_cov.block(0,index_begin,18,6) = _covariance.block(0,18+i*6,18,6);
			new_cov.block(index_begin,0,6,18) = _covariance.block(18+i*6,0,6,18);
			index_begin += 6;
		}
		std::cout << "ok " << __LINE__ << std::endl;
		int i_begin=18, j_begin=18;
		for(int i=0; i<_window.size(); ++i){
			if(clear_flag[i])
				continue;
			for(int j=0; j<_window.size(); ++j){
				if(clear_flag[j])
					continue;
				new_cov.block(i_begin, j_begin, 6, 6) = _covariance.block(18+i*6, 18+j*6, 6, 6);
				j_begin += 6;
			}
			j_begin = 18;
			i_begin += 6;
		}
		std::cout << "ok " << __LINE__ << std::endl;
		_covariance = new_cov;

		for(int i=0; i<_window.size(); ++i){
			if(!clear_flag[i])
				continue;
			frame *frame_ptr = _window[i];
			auto feature_vec = frame_ptr->_visible_feature;
			for(int j=0; j<feature_vec.size(); ++j){
				feature_vec[j]->delete_measurement(frame_ptr);
				frame_ptr->delete_feature(feature_vec[j]);
			}
			delete frame_ptr;
		}
		int index = 0;
		std::cout << "window size: " << _window.size() << std::endl;
		for(auto iter=_window.begin(); iter!=_window.end();){
			if(clear_flag[index]){
				iter = _window.erase(iter);
				++index;
			}else{
				++iter;
				++index;
			}
		}
		std::cout << "window size: " << _window.size() << std::endl;

	}

	cv::Mat drawmatch(cv::Mat &im1, cv::Mat &im2, std::vector<cv::Point2f> &p1, std::vector<cv::Point2f> &p2, std::vector<uchar> &status){
		int rows = im1.rows, cols = im1.cols;
		cv::Mat im_out(rows, cols*2, CV_8UC1);
		// Copy images in correct position
		im1.copyTo(im_out(cv::Rect(0, 0, cols, rows)));
		im2.copyTo(im_out(cv::Rect(cols, 0, cols, rows)));
		for(int i=0; i<status.size(); ++i)
			if(status[i]>0)
				line(im_out, p1[i], cv::Point2f(p2[i].x+cols, p2[i].y), CV_RGB(255, 255, 255), 1);
		return im_out;
	}

	void augumentation(frame *frame_ptr){
		// for covariance
		size_t size = _covariance.rows();
		Eigen::MatrixXd large_jacobian(size+6, size);
		Eigen::MatrixXd iden(size, size);
		iden.setIdentity();
		Eigen::MatrixXd jacobian(6, size);
		jacobian.setZero();
		jacobian.block(0,6,3,3) = _r_imu_camera.transpose(); // TODO need to check
		jacobian.block(3,6,3,3) = -_state_updated._pose.toRotationMatrix()*crossMat(_t_imu_camera); //t over r
		jacobian.block(3,0,3,3) = Eigen::Matrix3d::Identity(); //t over t
		large_jacobian.block(0,0,size,size) = iden;
		large_jacobian.block(size,0,6,size) = jacobian;
		_covariance = (large_jacobian*_covariance*large_jacobian.transpose()).eval();

		// log pose to frame
		frame_ptr->_r_propagated = _state_updated._pose.toRotationMatrix()*_r_imu_camera;
		frame_ptr->_r_updated = frame_ptr->_r_propagated;
		frame_ptr->_t_propagated = _state_updated._position + _state_updated._pose.toRotationMatrix()*_t_imu_camera;
		frame_ptr->_t_updated = frame_ptr->_t_propagated;
		// std::cout << "_state_updated._position: " << std::endl << _state_updated._position << std::endl;
		// std::cout << "augumentation: t" << std::endl << frame_ptr->_t_updated << std::endl;
		frame_ptr->_covariance = _covariance.block(size,size,6,6);

		// window second
		_window.push_back(frame_ptr);
	}


	void tracking(std::vector<feature*> &lost_feature){
		lost_feature.clear();
		if(_window.size() == 1)
			return;
		std::vector<cv::Point2f> prev_points, next_points;
		int prev_index = _window.size()-2;
		frame *prev_frame = _window[prev_index];
		frame *next_frame = _window[prev_index+1];
		prev_points = prev_frame->_feature_location;

		std::cout << "prev_points size: " << prev_frame->_feature_location.size() << std::endl;

		std::vector<uchar> status;
	  	std::vector<float> err;
	  	std::cout << "ok " << __LINE__ << std::endl;
	  	cv::imshow("tt", prev_frame->_img);
	  	// cv::waitKey(0);
	  	cv::calcOpticalFlowPyrLK(prev_frame->_img, next_frame->_img , prev_points, next_points, status, err, cv::Size(21,21), 3);
	  	std::cout << "ok " << __LINE__ << std::endl;
	  	for(int i=0; i<prev_points.size(); ++i){
	  		if(status[i] > 0 && err[i] < 30){ //TODO 30 as threahold
	  			next_frame->add_feature(prev_frame->_visible_feature[i], next_points[i]);
	  			prev_frame->_visible_feature[i]->add_measurement(next_frame, next_points[i]);
	  		}else{
	  			status[i] = 0;
	  			lost_feature.push_back(prev_frame->_visible_feature[i]);
	  		}
	  	}
	  	cv::Mat temp = drawmatch(prev_frame->_img, next_frame->_img, prev_points, next_points, status);
	  	cv::imshow("matching", temp);
	  	cv::waitKey(0);
	}

	void summarize_feature(feature *feature_ptr, std::vector<Eigen::Vector3d> normalized_point, std::vector<Eigen::Matrix<double, 3, 4>> transform){
		std::cout << "SUMMARIZE" << std::endl;
		std::cout << "feature_ptr: " << feature_ptr << std::endl;
		for(int i=0; i<feature_ptr->_location.size(); ++i){
			std::cout << "location: " << feature_ptr->_location[i].x << " " << feature_ptr->_location[i].y << std::endl;
			show_vector("normalized_point", normalized_point[i]);
			std::cout << "transform: " << std::endl << transform[i] << std::endl;
		}
		show_vector("feature 3d position", feature_ptr->_position);
		for(int i=0; i<feature_ptr->_visible_frame.size(); ++i){
			Eigen::Vector3d temp = transform[i].block(0,0,3,3)*feature_ptr->_position+transform[i].block(0,3,3,1);
			show_vector("position wrt frame", temp);
		}
	}
	void update(std::vector<feature*> &lost_feature){
		// delete feature having measurements less than 3 TODO 3 can be tuned
		for(auto iter=lost_feature.begin(); iter!=lost_feature.end();){
			if((*iter)->_visible_frame.size() < 3){
				for(int j=0; j<(*iter)->_visible_frame.size(); ++j)
					(*iter)->_visible_frame[j]->delete_feature(*iter);
				delete (*iter);
				iter = lost_feature.erase(iter);
			}else{
				 ++iter;
			}
		}
		std::cout << "lost_feature size: " << lost_feature.size() << std::endl;
		// std::cout << "ok " << __LINE__ << std::endl;
		// triangulate
		std::vector<Eigen::Vector3d> feature_position;
		std::vector<Eigen::Vector3d> normalized_point;
		std::vector<Eigen::Matrix<double, 3, 4>> transform;
		for(auto iter=lost_feature.begin(); iter!=lost_feature.end();){
			normalized_point.clear();
			transform.clear();
			feature *feature_ptr = *iter;
			for(int j=0; j<feature_ptr->_location.size(); ++j){
				normalized_point.push_back(_camera->to3d(feature_ptr->_location[j]));
				Eigen::Matrix<double, 3, 4> temp;
				temp.block(0,0,3,3) = feature_ptr->_visible_frame[j]->_r_updated.transpose();
				temp.block(0,3,3,1) = -temp.block(0,0,3,3)*feature_ptr->_visible_frame[j]->_t_updated;
				transform.push_back(temp);
			}
			// std::cout << "ok " << __LINE__ << std::endl;
			// std::cout << "normalized_point size: " << normalized_point.size() << " _position size: " <<feature_ptr->_position.size() << std::endl;
			bool good = triangulate(normalized_point, transform, feature_ptr->_position);
			if(!good){
				for(int j=0; j<(*iter)->_visible_frame.size(); ++j)
					(*iter)->_visible_frame[j]->delete_feature(*iter);
				delete (*iter);
				iter = lost_feature.erase(iter);
			}else{
				++iter;
			}
			// summarize_feature(feature_ptr, normalized_point, transform);
			// std::cout << "ok " << __LINE__ << std::endl;
		}
		std::cout << "lost_feature size: " << lost_feature.size() << std::endl;
		// std::cout << "ok " << __LINE__ << std::endl;
		// delete feature triangulate fail
		for(auto iter=lost_feature.begin(); iter!=lost_feature.end(); ){
			if((bool) std::isnan((*iter)->_position(2)) ){
				for(int j=0; j<(*iter)->_visible_frame.size(); ++j)
					(*iter)->_visible_frame[j]->delete_feature(*iter);
				delete (*iter);
				iter = lost_feature.erase(iter); 
			}else{
				++iter;
			}
		}
		std::cout << "lost_feature size: " << lost_feature.size() << std::endl;
		// std::cout << "ok " << __LINE__ << std::endl;
		// build per feature measurement
		std::vector<Eigen::VectorXd> residuals_vec;
		std::vector<Eigen::MatrixXd> H_vec;
		// std::vector<Eigen::MatrixXd> noise_vec;
		for(int i=0; i<lost_feature.size(); ++i){
			Eigen::VectorXd res_temp;
			Eigen::MatrixXd H_temp;
			// Eigen::MatrixXd noise_temp;
			bool success = build_measure_equation(lost_feature[i], res_temp, H_temp);
			if(success){
				residuals_vec.push_back(res_temp);
				H_vec.push_back(H_temp);
				// noise_vec.push_back(noise_temp);
			}
		}
		std::cout << "update size: " << H_vec.size() << std::endl;
		// std::cout << "ok " << __LINE__ << std::endl;
		// build on all feature
		size_t res_row = 0;
		size_t state_length = _covariance.rows();
		for(int i=0; i<residuals_vec.size(); ++i){
			res_row += residuals_vec[i].size();
		}
		// std::cout << "ok " << __LINE__ << std::endl;
		if(res_row == 0) 
			return; // no update
		Eigen::VectorXd residual_all(res_row);
		Eigen::MatrixXd H_all(res_row, state_length);
		size_t row_add = 0;
		for(int i=0; i<residuals_vec.size(); ++i){
			int row_temp = residuals_vec[i].size();
			residual_all.segment(row_add, row_temp) = residuals_vec[i];
			H_all.block(row_add,0,row_temp,state_length) = H_vec[i];
			row_add += row_temp;
		}
		// std::cout << "ok " << __LINE__ << std::endl;
		// QR reduction
		const int m = H_all.rows();
	    const int n = H_all.cols();
	    if (m == 0 || n == 0 || n > m) {
	        //no qr
	        int axiba = 0;
	    }else{
			Eigen::HouseholderQR<Eigen::MatrixXd> qr = H_all.householderQr();
	    	Eigen::MatrixXd Q1 = qr.householderQ() * Eigen::MatrixXd::Identity(m, n);
		    H_all = qr.matrixQR().topLeftCorner(n, n).template triangularView<Eigen::Upper>();
		    residual_all = Q1.transpose() * residual_all;
	    }
	    int num_row = residual_all.size();
	    Eigen::MatrixXd noise_all(num_row, num_row);
	    noise_all.setIdentity();
	    noise_all *= 10; // pixel noise TODO can be tuned

	    //ekf update
	    // show_matrix("H_all", H_all);
	    Eigen::MatrixXd kalman_gain = _covariance * H_all.transpose() * (H_all * _covariance * H_all.transpose() + noise_all).inverse();
	    // show_matrix("kalman_gain", kalman_gain);
	    Eigen::VectorXd delta_x = kalman_gain * residual_all;
	    // show_vector("residual_all", residual_all);
	    // show_vector("delta_x", delta_x);
	    Eigen::MatrixXd KTH = kalman_gain * H_all;
	    int cov_dim = KTH.rows();
	    Eigen::MatrixXd I(cov_dim, cov_dim);
	    I.setIdentity();
	    _covariance = (I - KTH) * _covariance * (I - KTH).transpose() + kalman_gain * noise_all * kalman_gain.transpose();

	    //update state
	    _state_updated._position += delta_x.segment(0,3);
		_state_updated._velocity += delta_x.segment(3,3);
	    Eigen::Quaterniond qua_inc(1, delta_x(6) * 0.5, delta_x(7) * 0.5, delta_x(8) * 0.5);
		_state_updated._pose = _state_updated._pose * qua_inc;
		_state_updated._acc_bias += delta_x.segment(9,3);
		_state_updated._gyro_bias += delta_x.segment(12,3);
		_state_updated._gravity += delta_x.segment(15,3);
	    for(int i=0; i<_window.size(); ++i){
	    	Eigen::Vector3d theta = delta_x.segment(18+i*6,3)*0.5;
	    	Eigen::Quaterniond q_inc(1, theta(0), theta(1), theta(2));
	    	_window[i]->_r_updated *= q_inc.toRotationMatrix();
	    	_window[i]->_t_updated += delta_x.segment(18+i*6+3,3);
	    	_window[i]->_covariance = _covariance.block(18+i*6,18+i*6,6,6);
	    }

	    //delete feature
	    for(int i=0; i<lost_feature.size(); ++i){
			for(int j=0; j<lost_feature[i]->_visible_frame.size(); ++j)
				lost_feature[i]->_visible_frame[j]->delete_feature(lost_feature[i]);
			delete lost_feature[i];
		}

	}

	bool triangulate(std::vector<Eigen::Vector3d> &normalized_point, std::vector<Eigen::Matrix<double, 3, 4>> &transform, Eigen::Vector3d &position){
		int frame_num = normalized_point.size();
		// std::cout << "ok " << __LINE__ << std::endl;
		bool return_flag = true;
		// DLT
		Eigen::MatrixXd equation(2*frame_num, 4);
		for(int i=0; i<frame_num; ++i){
			equation.block(i*2, 0, 1, 4) = normalized_point[i](0) * transform[i].row(2) - transform[i].row(0); //TODO need to check x and y
			equation.block(i*2+1, 0, 1, 4) = normalized_point[i](1) * transform[i].row(2) - transform[i].row(1); //TODO need to check x and y
			// show_vector("normalized_point", normalized_point[i]);
			// show_matrix("transform", transform[i]);
		}
		// std::cout << "ok " << __LINE__ << std::endl;
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(equation, Eigen::ComputeFullV);
  		Eigen::Vector4d initial_position = svd.matrixV().col(3);
  		initial_position = (initial_position/initial_position(3)).eval();
  		
  		// LM from svo lsd
  		int iteration_num = 10;//TODO 10 can be tuned
  		Eigen::Vector4d old_point = initial_position;
  		// show_vector("initial_position: ", initial_position);
  		double chi2 = 0.0;
  		Eigen::Matrix3d A;
  		Eigen::Vector3d b;
  		for(int i=0; i<iteration_num; ++i){
  			// get error
  			chi2 = 0;
  			for(int j=0; j<frame_num; ++j){
  				const Eigen::Vector3d p_in_f = transform[j]*initial_position;
  				chi2 += (normalized_point[j].segment(0,2) - p_in_f.segment(0,2)/p_in_f(2)).squaredNorm();
  			}

  			// get jacobian and build system
   			A.setZero();
    		b.setZero();
		    // compute residuals
		    for(int j=0; j<frame_num; ++j){
		      Eigen::Matrix<double, 2, 3> J;
		      const Eigen::Vector3d p_in_f = transform[j]*initial_position;
		      double z_inv = 1/p_in_f(2), x = p_in_f(0), y = p_in_f(1);
		      J << z_inv, 0 , -x*z_inv*z_inv , 0, z_inv, -y*z_inv*z_inv;
		      J = (-J*transform[j].block(0,0,3,3)).eval();
		      const Eigen::Vector2d e(normalized_point[j].segment(0,2) - p_in_f.segment(0,2)/p_in_f(2));
		      A.noalias() += J.transpose() * J;
		      b.noalias() -= J.transpose() * e;
		    }
		    // std::cout << "ok " << __LINE__ << std::endl;
    		// solve linear system
    		const Eigen::Vector3d dp(A.ldlt().solve(b));
    		// recompute error
    		Eigen::Vector4d new_point = initial_position;
			new_point.block(0,0,3,1) = new_point.block(0,0,3,1)+dp;
    		double new_chi2 = 0.0;
    		for(int j=0; j<frame_num; ++j){
  				const Eigen::Vector3d p_in_f = transform[j]*new_point;
  				new_chi2 += (normalized_point[j].segment(0,2) - p_in_f.segment(0,2)/p_in_f(2)).squaredNorm();
  			}
    		if((i > 0 && new_chi2 > chi2) || (bool) std::isnan((double)dp(0))) {
				// initial_position = old_point; // roll-back
				break;
			}
			// std::cout << "chi2: " << chi2 << " new_chi2: " << new_chi2 << std::endl;
			// show_vector("initial_position", initial_position);
			// show_vector("optimized_position", new_point);
			initial_position = new_point;
  		}
  		position = initial_position.block(0,0,3,1);
  		return return_flag;
  		// std::cout << "ok " << __LINE__ << std::endl;
  	}
	
	Eigen::Matrix3d crossMat(Eigen::Vector3d &vec){
		Eigen::Matrix3d returnMat;
		returnMat << 0,		-vec(2),vec(1),
					 vec(2), 0,		-vec(0),
					 -vec(1),vec(0),0;
		return returnMat;	
	}

	// the noise can be computed directly
	bool build_measure_equation(feature *feature_ptr, Eigen::VectorXd &res_temp, Eigen::MatrixXd &H_temp){//, Eigen::MatrixXd &noise_temp){
		// std::cout << "ok " << __LINE__ << std::endl;
		size_t frame_num = feature_ptr->_visible_frame.size();
		size_t state_length = _covariance.rows();
		res_temp.resize(frame_num*2);
		H_temp.resize(frame_num*2, state_length);
		Eigen::MatrixXd H_feature(frame_num*2, 3);
		// noise_temp.resize(num*2, num*2);
		res_temp.setZero();
		H_temp.setZero();
		H_feature.setZero();
		// noise_temp.setZero();
		//build initial
		// std::cout << "ok " << __LINE__ << std::endl;
		for(int i=0; i<feature_ptr->_visible_frame.size(); ++i){
			frame *frame_ptr = feature_ptr->_visible_frame[i];
			Eigen::Vector3d point_in_frame = frame_ptr->_r_updated.transpose() * (feature_ptr->_position - frame_ptr->_t_updated);
			// std::cout << "feature postion: " << std::endl << feature_ptr->_position << std::endl;
			// std::cout << "frame t: " << std::endl << frame_ptr->_t_updated << std::endl;
			double x = point_in_frame(0), y = point_in_frame(1), z = point_in_frame(2), z_inv = 1/z;
			// std::cout << "ok " << __LINE__ << std::endl;
			// Eigen::Vector3d vec_cross = frame_ptr->_r_updated.transpose() * feature_ptr->_position;
			Eigen::Vector2d xy = point_in_frame.segment(0,2)/point_in_frame(2);
			Eigen::Vector2d actual;
			actual << (double)(feature_ptr->_location[i].x), (double)(feature_ptr->_location[i].y);
			//res
			res_temp.segment(i*2,2) =  _camera->to3d(feature_ptr->_location[i]).segment(0,2) - xy;
			show_vector("residual", res_temp.segment(i*2,2));
			Eigen::Matrix<double, 2, 3> J, xy_over_r, xy_over_t, xy_over_feature;
			J << z_inv, 0 , -x*z_inv*z_inv , 0, z_inv, -y*z_inv*z_inv;
			xy_over_r = J * crossMat(point_in_frame); // minus for _r_updated transpose TODO not confirm
			xy_over_t = -J * frame_ptr->_r_updated.transpose();
			xy_over_feature = J * frame_ptr->_r_updated.transpose();

			//H_temp
			size_t index = _window.size()-(feature_ptr->_visible_frame.size()-i-1)-1-1; //newest frame push back in _window, and one more -1 for augmented state

			H_temp.block(i*2,15+6*index,2,3) = xy_over_r;
			H_temp.block(i*2,15+6*index+3,2,3) = xy_over_t;

			//H_feature
			H_feature.block(i*2,0,2,3) = xy_over_feature;
			// std::cout << "ok " << __LINE__ << std::endl;
			// //noise_temp
			// noise_temp.block(2*i,2*i,2,2) = Eigen::Matrix2d::Identity() * 2; //2 as pixel variance TODO can be tuned
		}
		// std::cout << "ok " << __LINE__ << std::endl;
		bool return_flag = true;
		//project null space
		Eigen::FullPivLU<Eigen::MatrixXd> lu(H_feature.transpose());
		// std::cout << "ok " << __LINE__ << std::endl;
		// std::cout << "H_feature.transpose(): " << std::endl << H_feature.transpose() <<std::endl;
		// std::cout << "lu.kernel(): " << std::endl << lu.kernel() <<std::endl;
		Eigen::MatrixXd left_null_space_t = lu.kernel().transpose();
		// std::cout << "ok " << __LINE__ << std::endl;
		if(left_null_space_t.rows() != 2*frame_num - 3){
			//no enough basis
			return_flag = false;
		}
		res_temp = (left_null_space_t*res_temp).eval();
		H_temp = (left_null_space_t*H_temp).eval();
		// noise_temp = (left_null_space_t*noise_temp*left_null_space_t.transpose()).eval();
		// std::cout << "ok " << __LINE__ << std::endl;
		return return_flag;
	}


	std::vector<frame*> _window;
	Eigen::MatrixXd _covariance;
	Eigen::Matrix3d _r_imu_camera;
	Eigen::Vector3d _t_imu_camera;
	state _state_propagated;
	state _state_updated;
	double _time_stamp;
	double _acc_noise_density;
	double _gyro_noise_density;
	double _acc_random_walk;
	double _gyro_random_walk;
	camera_model *_camera;

};

int main(){
	Eigen::Matrix3d r_imu_camera = Eigen::Matrix3d::Identity();
	r_imu_camera << 0.0148655429818, -0.999880929698, 0.00414029679422,
					0.999557249008, 0.0149672133247, 0.025715529948,
					-0.0257744366974, 0.00375618835797, 0.999660727178;
	Eigen::Vector3d t_imu_camera = Eigen::Vector3d::Zero();
	t_imu_camera << -0.0216401454975,-0.064676986768,0.00981073058949;
	camera_model *camera = new camera_model(458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);
	double acc_noise = 2.0000e-3;
	double acc_random = 2.0000e-3;
	double gyro_noise = 1.6968e-04;
	double gyro_random = 1.9393e-05;

	double initial_time = 1.40363658083856E+018/1e9;
	Eigen::Vector3d position;
	position << 4.688319,-1.786938,0.783338;
	Eigen::Quaterniond pose(0.534108,-0.153029,-0.827383,-0.082152);
	Eigen::Vector3d velocity;
	velocity << -0.027876,0.033207,0.800006;
	Eigen::Vector3d gyro_bias;
	gyro_bias << -0.003172,0.021267,0.078502;
	Eigen::Vector3d acc_bias;
	acc_bias << -0.025266,0.136696,0.075593;
	Eigen::Vector3d gravity;
	gravity << 0,0,-9.81;
	state initial_state;
	
	// initial_time = 0;
	// position << 20, 5, 5;
	// pose = Eigen::Quaterniond(0.99875, 0.0499792, 0, 0);
	// velocity << -0, 6.28319, 3.14159;
	// gyro_bias.setZero();
	// acc_bias.setZero();

	initial_state._position = position;
	initial_state._pose = pose;
	initial_state._velocity = velocity;
	initial_state._acc_bias = acc_bias;
	initial_state._gyro_bias = gyro_bias;
	initial_state._gravity = gravity;
	msckf test_kf(acc_noise, acc_random, gyro_noise, gyro_random, r_imu_camera, t_imu_camera, initial_state, initial_time, camera);
	std::vector<std::string> gt_line;
	std::vector<std::string> data_line;
	std::string gt_file = "/home/chenchr/Desktop/mav0/imu0/data.csv";
	std::string data_file = "/home/chenchr/Desktop/mav0/imu0/data.csv";
	std::string img_csv = "/home/chenchr/Desktop/mav0/cam0/data.csv";
	std::vector<cv::Mat> img_vec;
	// data_file = "/home/chenchr/project/vio_data_simulation/bin/imu_pose.txt";
	std::ifstream file(gt_file.c_str()), file2(data_file.c_str()), file3(img_csv.c_str());
	std::vector<imu_measurement> imu_m;
	std::vector<Eigen::Vector3d> int_position;
	std::string line;
	for(int i=0; i<21; ++i)
		std::getline(file3, line);
	std::cout << "ok " << __LINE__ << std::endl;
	for(int i=0; i<81; ++i){
		std::getline(file3, line);
		std::string name =  line.substr(20,23);
		name = "/home/chenchr/Desktop/mav0/cam0/data/" + name;
		cv::Mat img = cv::imread(name.c_str());
		cv::Mat gray;
		cv::cvtColor(img, gray, CV_BGR2GRAY);
		img_vec.push_back(gray);
	}
	std::cout << "ok " << __LINE__ << std::endl;
	for(int i=0; i<=218; ++i)
		std::getline(file2, line);
	for(int i=0; i<20000; ++i){
		double time, d1,d2,d3;
		Eigen::Vector3d temp;
		imu_measurement temp_m;
		char comma;
		file2 >> time;
		temp_m._time_stamp = time/1e9;
		file2 >> comma >> d1 >> comma >> d2 >> comma >> d3;
		temp << d1,d2,d3;
		temp_m._gyro = temp;
		file2 >> comma >> d1 >> comma >> d2 >> comma >> d3;
		temp << d1,d2,d3;
		temp_m._acc = temp;
		imu_m.push_back(temp_m);
	}
	// std::getline(file2, line);
	// for(int i=1; i<4000; ++i){
	// 	double time, d1,d2,d3;
	// 	Eigen::Vector3d temp;
	// 	imu_measurement temp_m;
	// 	char comma;
	// 	file2 >> time;
	// 	temp_m._time_stamp = time;
	// 	file2 >> d1 >> d1 >> d1 >> d1 >> d1 >> d1 >>d1;
	// 	file2 >> d1 >> d2 >> d3;
	// 	temp << d1,d2,d3;
	// 	temp_m._gyro = temp;
	// 	file2 >> d1 >> d2 >> d3;
	// 	temp << d1,d2,d3;
	// 	temp_m._acc = temp;
	// 	imu_m.push_back(temp_m);
	// }
	file.close();
	file2.close();
	file3.close();
	
	for(int i=0; i<301; ++i){
	// for(int i=0; i<5; ++i){
		// std::cout << "ok " << __LINE__ << std::endl;
		test_kf.propagate(imu_m[i]._gyro, imu_m[i]._acc, imu_m[i]._time_stamp);
		int_position.push_back(test_kf._state_updated._position);
		if(i%10 == 0){
			frame *frame_ptr = new frame(imu_m[i]._time_stamp, img_vec[i/10]);
			test_kf.measure(imu_m[i]._time_stamp, frame_ptr);
			int_position.push_back(test_kf._state_updated._position);
		}
	}
	//visualize
	pangolin::CreateWindowAndBind("naive",1600,1600);
	glEnable(GL_DEPTH_TEST);
    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024*5,768*5,500,500,512,389,0.1,1000),
                pangolin::ModelViewLookAt(0,-0.7,-1.8, 0,0,0,0.0,-1.0, 0.0)
                );
    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    std::cout << "size: " << int_position.size() << std::endl;
    for(int i=0; i<301; i+=5){
    	std::cout << i << "\t" << int_position[i](0) << " " << int_position[i](1) << " " << int_position[i](2) << std::endl;
    }
    while(1){
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	    glClearColor(1.0f,1.0f,1.0f,1.0f);

	    d_cam.Activate(s_cam);

	    int pointsize = 2;
		glPointSize(pointsize);
		glBegin(GL_POINTS);
		glColor3f(0.0,0.0,0.0);
	    for(size_t i=0; i<261; i++){
	        glVertex3f(int_position[i](0), int_position[i](1), int_position[i](2));
	    }
	    glEnd();
	    glPointSize(pointsize);
		glBegin(GL_POINTS);
		glColor3f(1.0,0.0,0.0);
	    for(size_t i=261; i<int_position.size(); i++){
	        glVertex3f(int_position[i](0), int_position[i](1), int_position[i](2));
	    }
	    glEnd();
	    pangolin::FinishFrame();
    }
    return 0;
}