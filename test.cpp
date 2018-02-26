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
		std::cout << "size: " << grid_x_*grid_y_ << std::endl;
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
					for(int num=0; num<chosed.size(); ++num){
						int dx = chosed[num].pt.x - x;
						int dy = chosed[num].pt.y - y;
						if(dx*dx+dy*dy < square)
							chosed.erase(chosed.begin()+num);
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
			if(kps.size()>0)
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

struct imu_state{
	imu_state(){
		p_ << 0, 0, 0;
		v_ << 0, 0, 0;
		q_ = Eigen::Quaterniond(1, 0, 0, 0);
		// std::cout << "sdfdsafaddsafdsa: " << q_.w() << " " << q_.x() << " " << q_.y() << " " << q_.z() << std::endl;
 		a_bias_ << 0, 0, 0;
		w_bias_ << 0, 0, 0;
		g_ << 0, 0, -9.81;
		time_ = 0;
		temp_val << 0, 0, 0;
	}

	imu_state(Eigen::Vector3d p, Eigen::Vector3d v, Eigen::Quaterniond q, Eigen::Vector3d a_bias, Eigen::Vector3d w_bias, Eigen::Vector3d g, double time){
		p_ = p;
		v_ = v;
		q_ = q;
		a_bias_ = a_bias;
		w_bias_ = w_bias;
		g_ = g;
		time_ = time;
		temp_val << 0, 0, 0;
	}
	Eigen::Vector3d temp_val;
	Eigen::Vector3d p_;
	Eigen::Vector3d v_;
	Eigen::Quaterniond q_;
	Eigen::Vector3d a_bias_;
	Eigen::Vector3d w_bias_;
	Eigen::Vector3d g_;
	double time_;
};

struct imu_reading{
	imu_reading(){
		memset(this,0,sizeof(imu_reading));
	}

	imu_reading(Eigen::Vector3d w, Eigen::Vector3d a, double time){
		w_ = w;
		a_ = a;
		time_ = time;
	}
	Eigen::Vector3d w_;
	Eigen::Vector3d a_;
	double time_;
};



imu_state propagate(imu_state &pre, imu_reading &read){
	static int cnt = 0;
	imu_state cur;
	float delta_t = (read.time_ - pre.time_)/1;
	Eigen::Vector3d w = read.w_-pre.w_bias_;
	w = w*delta_t*0.5;
	Eigen::Quaterniond qua_inc = Eigen::Quaterniond(1, w(0), w(1), w(2));
	cur.q_ = pre.q_ * qua_inc;
	Eigen::Vector3d temp_val = pre.q_*(read.a_-pre.a_bias_) + pre.g_;
	cur.v_ = pre.v_ + temp_val * delta_t;
	cur.p_ = pre.p_ + cur.v_*delta_t + temp_val*delta_t*delta_t*0.5;
	cur.g_ = pre.g_;
	cur.time_ = read.time_;
	cnt++;
	if(cnt < 100){
		// std::cout << "time: " << read.time_ << std::endl;
		// std::cout << cnt << std::endl;
		// std::cout << "pos: " << cur.p_(0) << " " << cur.p_(1) << " " << cur.p_(2) << std::endl;
		// std::cout << "qua: " << cur.q_.w() << " " << cur.q_.x() << " " << cur.q_.y() << " " << cur.q_.z() << std::endl;
		// std::cout << "delta_t: " << delta_t << std::endl;
		// std::cout << "pre.a_bias_: " << pre.a_bias_(0) << " " << pre.a_bias_(1) << " " << pre.a_bias_(2) << std::endl;
		// std::cout << "pre.w_bias_: " << pre.w_bias_(0) << " " << pre.w_bias_(1) << " " << pre.w_bias_(2) << std::endl;
	}
	return cur;
}

void read_data(std::string file_name, std::vector<imu_reading> &data){
	std::ifstream file(file_name.c_str());
	std::string line;
	// std::getline(file, line);
	data.clear();
	Eigen::Vector3d w_temp, a_temp;
	double time;
	char comma;
	while(file>>time){
		float a;
		// file >> comma >> w_temp(0) >> comma >> w_temp(1) >> comma >> w_temp(2) >> comma >> a_temp(0) >> comma >> a_temp(1) >> comma >> a_temp(2);
		file >> a >> a >> a >> a >> a >> a >> a >> 
		w_temp(0) >> w_temp(1) >> w_temp(2) >> a_temp(0) >>a_temp(1) >> a_temp(2);
		data.push_back(imu_reading(w_temp, a_temp, time));
	}
	file.close();
}

void read_gt(std::string file_name, std::vector<imu_state> &gt){
	std::ifstream file(file_name.c_str());
	std::string line;
	// std::getline(file, line);
	gt.clear();
	double time;
	Eigen::Vector3d p, v, w_bias, a_bias, g;
	float qw, qx, qy, qz;
	char comma;
	while(file>>time){
		float a; 
		file >> qw >> qx >> qy >> qz >> p(0) >> p(1) >> p(2) >> a >> a >> a >> a >> a >> a >> a;
		Eigen::Quaterniond qua(qw, qx, qy, qz);
		gt.push_back(imu_state(p, v, qua, a_bias, w_bias, g, time));
	}
}

void generate_trace(std::vector<imu_state> &state, std::vector<imu_reading> data, std::vector<imu_state> &gt){
	imu_state first;
	Eigen::Quaterniond qua(0.99875, 0.0499792, 0, 0);
	first.q_ = qua;
	Eigen::Vector3d position(20, 5, 5);
	first.p_ = position;
	first.v_ << -0, 6.28319, 3.14159;

	first.time_ = 0;
	first.g_ << 0,0,-9.81;
	state.clear();
	state.push_back(propagate(first, data[0]));
	for(int i=1; i<data.size(); ++i)
		state.push_back(propagate(state[i-1], data[i]));
}

void visualize(std::vector<imu_state> &estimate, std::vector<imu_state> &gt){
	pangolin::CreateWindowAndBind("naive",1024,768);
	glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024,768,500,500,512,389,0.1,1000),
                pangolin::ModelViewLookAt(0,-0.7,-1.8, 0,0,0,0.0,-1.0, 0.0)
                );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    size_t index = 0, big_index=0;
    while( !pangolin::ShouldQuit() )
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	    glClearColor(1.0f,1.0f,1.0f,1.0f);

	    d_cam.Activate(s_cam);
	    // if(index<estimate.size()){
	    	// std::cout << "estimate: " << std::endl << estimate[0].q_ << std::endl;
	    	// std::cout << "gt: " << std::endl << gt[0].q_ << std::endl;
	    	// index = 0;
		//     int pointsize = 1;
		//     Eigen::Vector3f x(1,0,0), y(0,1,0),z(0,0,1), o(0,0,0), temp;
		//     glLineWidth(pointsize);
		//     glBegin(GL_LINES);
		//     glColor3f(1.0,0.0,0.0);
		//     temp = estimate[index].q_*o;
		//     glVertex3f(temp(0), temp(1), temp(2));
		//     temp = estimate[index].q_*x;
		//     glVertex3f(temp(0), temp(1), temp(2));
		//     glEnd();
		//     glBegin(GL_LINES);
		//     glColor3f(0.0,1.0,0.0);
		//     temp = estimate[index].q_*o;
		//     glVertex3f(temp(0), temp(1), temp(2));
		//     temp = estimate[index].q_*y;
		//     glVertex3f(temp(0), temp(1), temp(2));
		//     glEnd();
		//     glBegin(GL_LINES);
		//     glColor3f(0.0,0.0,1.0);
		//     temp = estimate[index].q_*o;
		//     glVertex3f(temp(0), temp(1), temp(2));
		//     temp = estimate[index].q_*z;
		//     glVertex3f(temp(0), temp(1), temp(2));
		//     glEnd();

		//     glBegin(GL_LINES);
		//     glColor3f(0.0,0.0,0.0);
		//     temp = gt[index].q_*o;
		//     glVertex3f(temp(0), temp(1), temp(2));
		//     temp = gt[index].q_*x;
		//     glVertex3f(temp(0), temp(1), temp(2));
		//     glEnd();
		//     glBegin(GL_LINES);
		//     glColor3f(0.0,1.0,1.0);
		//     temp = gt[index].q_*o;
		//     glVertex3f(temp(0), temp(1), temp(2));
		//     temp = gt[index].q_*y;
		//     glVertex3f(temp(0), temp(1), temp(2));
		//     glEnd();
		//     glBegin(GL_LINES);
		//     glColor3f(1.0,0.0,1.0);
		//     temp = gt[index].q_*o;
		//     glVertex3f(temp(0), temp(1), temp(2));
		//     temp = gt[index].q_*z;
		//     glVertex3f(temp(0), temp(1), temp(2));
		//     glEnd();

		//     big_index++;
		//     if(big_index % 50 == 0)
		//     	++index;
		// }

	    int pointsize = 2;
		glPointSize(pointsize);
		glBegin(GL_POINTS);
		glColor3f(0.0,0.0,0.0);
	    for(size_t i=0, iend=7000; i<iend;i++)
	    {
	        glVertex3f(estimate[i].p_(0), estimate[i].p_(1), estimate[i].p_(2));
	    }
	    glEnd();

	    glPointSize(pointsize);
		glBegin(GL_POINTS);
		glColor3f(1.0,0.0,0.0);
		for(size_t i=0, iend=7500; i<iend;i++)
		{
		    glVertex3f(gt[i].p_(0), gt[i].p_(1), gt[i].p_(2));
		}
		glEnd();
	    pangolin::FinishFrame();
    }
}

cv::Mat drawmatch(cv::Mat &im1, cv::Mat &im2, std::vector<cv::Point2f> &p1, std::vector<cv::Point2f> &p2, std::vector<uchar> &status){
   int rows = im1.rows, cols = im1.cols;
   cv::Mat im_out(rows, cols*2, CV_8UC3);
    // Copy images in correct position
    im1.copyTo(im_out(cv::Rect(0, 0, cols, rows)));
    im2.copyTo(im_out(cv::Rect(cols, 0, cols, rows)));
    for(int i=0; i<status.size(); ++i)
      if(status[i]>0)
        line(im_out, p1[i], cv::Point2f(p2[i].x+cols, p2[i].y), CV_RGB(255, 255, 255), 1);
    return im_out;
}

int main(){
	cv::Mat src_ori = cv::imread("/home/chenchr/project/msckf/1.png");
	cv::Mat src;
	double kk = (double)cv::getTickCount();
  	kk = ((double)cv::getTickCount() - kk)/cv::getTickFrequency(); 
  	std::cout << "Times passed in seconds for blur: " << kk << std::endl;
	cv::medianBlur ( src_ori, src, 3 );
	cv::Mat src2_ori = cv::imread("/home/chenchr/project/msckf/2.png");
	cv::Mat src2;
	cv::medianBlur ( src2_ori, src2, 3 );
	cv::Mat gray, gray2;
	cv::cvtColor(src, gray, CV_BGR2GRAY);
	cv::cvtColor(src2, gray2, CV_BGR2GRAY);
	GridHarrisFast det(gray.size(), 60);
	std::vector< std::vector<cv::KeyPoint> > kps, kps2, kps3;
	std::vector< cv::Point2f> max_pt, max_pt2, max_pt3;
	std::vector<cv::Point2f> old;
	det.detect(gray, kps, old);
	det.draw_grid(src);
	det.draw_feature(src, kps);
	det.find_max(kps, max_pt);
	det.detect(gray, kps3, old);
	det.find_max(kps3, max_pt3);
	det.draw_feature(src, max_pt);
	det.draw_feature(src, max_pt3, true);
	

	std::vector< cv::Point2f> g2_pt(max_pt);
	std::vector<uchar> status;
  	std::vector<float> err;
  	cv::TermCriteria criteria=cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);
  	double t = (double)cv::getTickCount();
  	cv::calcOpticalFlowPyrLK(gray, gray2, max_pt, g2_pt, status, err, cv::Size(21,21), 3, 
  		criteria, cv::OPTFLOW_USE_INITIAL_FLOW);
  	t = ((double)cv::getTickCount() - t)/cv::getTickFrequency(); 
  	std::cout << "Times passed in seconds for lk track: " << t << std::endl;

  	std::vector<cv::Point2f> good_g2;
  	for(int i=0; i<status.size(); ++i)
  		if(status[i]>0)
  			good_g2.push_back(g2_pt[i]);
	det.detect(gray2, kps2, good_g2);
	det.find_max(kps2, max_pt2);
  	det.draw_feature(src2, kps2);
  	det.draw_feature(src2, good_g2, 60/3);
  	det.draw_grid(src2);
	det.draw_feature(src2, max_pt2, true);

  	cv::Mat match = drawmatch(src, src2, max_pt, g2_pt, status);
  	cv::imshow("match", match);

	std::cout << "size: " << kps.size() << std::endl;
	cv::imshow("test", src);
	cv::waitKey(0);
	Eigen::Vector3d vec = Eigen::Vector3d::Random();
	std::cout << "vec: " << vec << std::endl;
	vec(0) = 100;
	Eigen::Vector3d vec_nor = vec.normalized();
	float norm = vec.norm();
	std::cout << "before: " << vec << std::endl;
	std::cout << "axis: " << vec_nor << std::endl;
	std::cout << "angle: " << norm << std::endl;
	Eigen::AngleAxisd tt(norm, vec);
	std::cout << "axis2: " << tt.axis() << std::endl;
	std::cout << "angle2: " << tt.angle() << std::endl;

	std::cout << "test: " << std::endl;
	Eigen::Vector4d test_q(1,2,3,4);
	Eigen::Quaterniond qua(1, 2, 3, 4);
	// std::cout << "qua: " << qua << std::endl;
	std::cout << "w: " << qua.w() << std::endl;

	// std::string	file_name = "/home/chenchr/Desktop/mav0/imu0/data.csv";
	std::string	file_name = "/home/chenchr/project/vio_data_simulation/bin/imu_pose_noise.txt";
	std::vector<imu_reading> data;
	read_data(file_name, data);
	std::cout << "size: " << data.size() << std::endl;
	for(int i=0; i<30; ++i)
		std::cout << data[i].time_ << " " << data[i].w_(0) << " " << data[i].w_(1) << " " << data[i].w_(2) << " " << data[i].a_(0) << " " 
					<< data[i].a_(1) << " " << data[i].a_(2) << std::endl;

	// std::string file_gt = "/home/chenchr/Desktop/mav0/state_groundtruth_estimate0/data.csv";
	std::string file_gt = "/home/chenchr/project/vio_data_simulation/bin/imu_int_pose_noise.txt";
	std::vector<imu_state> gt;
	read_gt(file_gt, gt);
	std::vector<imu_state> state;
	generate_trace(state, data, gt);

	std::cout << "gt size: " << gt.size() << std::endl;
	std::cout << "state size: " << state.size() << std::endl;
	// for(int i=0; i<10000; ++i){
	// 	std::cout << "estimate: \t" << state[i].time_ << " \t\t" << state[i].p_(0) << " \t\t" << state[i].p_(1) << " \t\t" << state[i].p_(2) << std::endl;
	// 	std::cout << "gt:       \t" << gt[i].time_ << " \t\t" << gt[i].p_(0) << " \t\t" << gt[i].p_(1) << " \t\t" << gt[i].p_(2) << std::endl;
	// }
	// for(int i=0; i<30; ++i)
	
	std::cout << "Hello: " << std::endl;
	Eigen::Quaterniond tt_qua(1,0,0,0);
	Eigen::Vector3d qua_mul = Eigen::Vector3d::Random();
	std::cout << "qua_mul: " << qua_mul << std::endl;
	std::cout << "after: " << tt_qua*qua_mul << std::endl;

	Eigen::Vector3d acc_unbias(1,2,3);
	Eigen::Matrix3d acc_cross;
		acc_cross << 0,-acc_unbias(2),acc_unbias(1),acc_unbias(2),0,-acc_unbias(0),-acc_unbias(1),acc_unbias(0),0;
	std::cout << "acc_unbias: " << acc_unbias << std::endl;
	std::cout << "acc_cross: " << acc_cross << std::endl;
	std::cout << "acc_cross trans: " << acc_cross.transpose() << std::endl;
	std::cout << "acc_cross: " << acc_cross << std::endl;

	std::vector<int> int1 = {1,2,3,4,5};
	std::vector<int> int2 = {6,7,8,9,0};
	for(int i=0; i<int1.size(); ++i){
		std::cout << int1[i] << " " << std::endl;
	}
	for(int i=0; i<int2.size(); ++i){
		std::cout << int2[i] << " " << std::endl;
	}
	std::vector<int>::iterator iter = std::find(int1.begin(), int1.end(), 3);
	if(iter != int1.end()){
		int1.erase(iter);
		int2.erase(int2.begin()+(iter-int1.begin()));
	}
	for(int i=0; i<int1.size(); ++i){
		std::cout << int1[i] << " " << std::endl;
	}
	for(int i=0; i<int2.size(); ++i){
		std::cout << int2[i] << " " << std::endl;
	}
	int size = 4;
	Eigen::MatrixXd iden(size, size+1);
	std::cout << "before: " << iden << std::endl;
	iden.setIdentity();
	iden.block(0,0,2,2) = Eigen::Matrix2d::Random();
	std::cout << "after: " << iden << std::endl;

	Eigen::Matrix<double, 5, 3> test_null;
	test_null.setZero();
	test_null.block(2,0,3,3) = Eigen::Matrix3d::Identity();
	std::cout << "test_null: " << std::endl << test_null.transpose() << std::endl;
	Eigen::FullPivLU<Eigen::MatrixXd> lu(test_null.transpose());
	Eigen::MatrixXd A_null_space = lu.kernel();
	std::cout << "after: " << std::endl << A_null_space << std::endl;

	Eigen::MatrixXd test_x = Eigen::Matrix3d::Random();
	std::cout << "test_x: " << std::endl << test_x << std::endl;
	test_x = Eigen::Matrix2d::Identity();
	std::cout << "after: " << std::endl << test_x << std::endl;

	Eigen::VectorXd test_bl = Eigen::Vector4d::Random();
	std::cout << "test_bl: " << std::endl << test_bl << std::endl;
	test_bl.segment(0,2) = Eigen::Vector2d::Zero();
	std::cout << "after: " << std::endl << test_bl << std::endl;
	// visualize(state, gt);
	Eigen::Matrix<double, 2, 3> rrr = Eigen::Matrix<double, 2, 3>::Random();
	std::cout << "after: " << std::endl << rrr << std::endl;

	Eigen::VectorXd ttvec = Eigen::Vector4d::Random();
	ttvec.setZero();
	std::cout << "after: " << std::endl << ttvec << std::endl;
	std::cout << "size: " << ttvec.size() << std::endl;

	int length = 8;
	Eigen::VectorXd ttlength(length);
	// ttlength.setZero();
	Eigen::VectorXd ttlength1(length/2), ttlength2(length/2);
	ttlength1.setIdentity();
	ttlength2.setIdentity();
	ttlength << ttlength1,ttlength2;
	std::cout << "after: " << std::endl << ttlength << std::endl;
	std::cout << "size: " << ttlength.size() << std::endl;

	Eigen::MatrixXd H = Eigen::Matrix<double, 6, 3>::Random();
	Eigen::HouseholderQR<Eigen::MatrixXd> qr = H.householderQr();
    const int m = H.rows();
    const int n = H.cols();
    if (m == 0 || n == 0 || n > m) {
        return 0;
    }
    std::cout << "matrix bebefore qr: " << std::endl << H << std::endl;

    Eigen::MatrixXd Q1 = qr.householderQ() * Eigen::MatrixXd::Identity(m, n);
    std::cout << "q1: " << std::endl << Q1 << std::endl;
    std::cout << "ok" << std::endl;
    Eigen::MatrixXd H1 = qr.matrixQR().topLeftCorner(n, n).template triangularView<Eigen::Upper>();
    std::cout << "matrixqr: " << std::endl << H1 << std::endl;
    H = (Q1.transpose()*H).eval();
    std::cout << "matrixqr: " << std::endl << H << std::endl;
    // r = Q1.transpose() * r;
    // R = Q1.transpose() * R * Q1;

    std::string str = "1403636763813555456,1403636763813555456.png";
    std::cout << "str: " << str << std::endl;
    std::string substring = str.substr(20,23);
    std::cout << "substring: " << substring << std::endl;

    cv::Mat rt1 = cv::Mat::zeros(3, 4, CV_32FC1), rt2 = cv::Mat::zeros(3, 4, CV_32FC1);
    std::vector<float> rt1_vec = {
    	0.358704,   0.933451, -0.00117436 ,  -0.312113,
   0.382241 ,  -0.148032,   -0.912136 ,   -4.88009,
  -0.851609 ,   0.326737 ,  -0.409902  ,  -1.36471
};
    std::vector<float> rt2_vec = {
    	 0.366256 ,  0.93046, 0.0101508, -0.332294,
 0.371587 , -0.13624, -0.918371 , -4.86585,
-0.853126,  0.340125, -0.395646 , -1.42255
};
    for(int i=0; i<3; ++i){
    	for(int j=0; j<4; ++j){
    		rt1.at<float>(i,j) = rt1_vec[i*4+j];
    		rt2.at<float>(i,j) = rt2_vec[i*4+j];
    	}
    }
    std::cout << "rt1: " << std::endl << rt1 << std::endl;
    std::cout << "rt2: " << std::endl << rt2 << std::endl;
    cv::Mat p1 = cv::Mat::zeros(2, 1, CV_32FC1), p2 = cv::Mat::zeros(2, 1, CV_32FC1);
    p1.at<float>(0,0) = 0.501072;
    p1.at<float>(1,0) = 0.4044;
    p2.at<float>(0,0) = 0.472263;
    p2.at<float>(1,0) = 0.521477;
    cv::Mat point4d;
    cv::triangulatePoints(rt1, rt2, p1, p2, point4d);
    std::cout << "point4d: " << point4d << std::endl;
	for(int i=0; i<4; ++i)
		point4d.at<float>(i,0) = point4d.at<float>(i,0)/point4d.at<float>(3,0);
	std::cout << "point4d: " << point4d << std::endl; 
	return 0;

}
