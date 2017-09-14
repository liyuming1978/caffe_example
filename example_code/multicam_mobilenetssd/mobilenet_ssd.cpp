#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include "opencv2/core/ocl.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#endif  // USE_OPENCV
//#include <algorithm>
//#include <iomanip>
//#include <iosfwd>
//#include <memory>
//#include <string>
//#include <utility>
#include <vector>
#include <pthread.h>
#include <queue>
#include <semaphore.h>
#include <glog/logging.h>


#if defined(USE_OPENCV) && defined(HAS_HALF_SUPPORT)
using namespace caffe;  // NOLINT(build/namespaces)
using caffe::Timer;
using std::queue;  

#define Dtype half

class Detector {
public:
	typedef struct __detect_result {
		int imgid;
		float classid;
		float confidence;
		float left;
		float right;
		float top;
		float bottom;
	}detect_result;
	
	Detector(const string& model_file,
		const string& weights_file,int min_batch);
	~Detector();

	bool Detect(vector<vector<Detector::detect_result> >& objects);
	inline int GetCurBatch(){return  num_batch_;}
	bool InsertImage(const cv::Mat& orgimg);
	void SetBatch(int batch);
	bool TryDetect();

private:
	void WrapInputLayer(Blob<Dtype>* pdata);
	cv::Mat PreProcess(const cv::Mat& img);
	void CreateMean();
	void EmptyQueue(queue<Blob<Dtype>*>& que);
	void EmptyQueue(queue<cv::Size>& que);
	shared_ptr<Net<Dtype> > net_;
	std::vector<Dtype *> input_channels;
	cv::Size input_geometry_;
	queue<Blob<Dtype>*> batchque_;
	queue<cv::Size> imgsizeque_;
	pthread_mutex_t mutex ; 
	int nbatch_index_;
	int num_channels_;
	int min_batch_;
	int num_batch_;
	cv::Mat mean_;
};

// Get all available GPU devices
static void get_gpus(vector<int>* gpus) {
	int count = 0;
#ifndef CPU_ONLY
	count = Caffe::EnumerateDevices(true);
#else
	NO_GPU;
#endif
	for (int i = 0; i < count; ++i) {
		gpus->push_back(i);
	}
}

Detector::Detector(const string& model_file,
	const string& weights_file,int min_batch) {
	// Set device id and mode
	vector<int> gpus;
	get_gpus(&gpus);
	Caffe::SetDevices(gpus);
	bool bGpumode = false;
	if (gpus.size() != 0) {
#ifndef CPU_ONLY
		for (int i = 0; i < gpus.size(); i++) {
			if (Caffe::GetDevice(gpus[i], true)->backend() == BACKEND_OpenCL) {
				if (Caffe::GetDevice(gpus[i], true)->CheckVendor("Intel")
					&& Caffe::GetDevice(gpus[i], true)->CheckType("GPU")) {
					//&& Caffe::GetDevice(gpus[i], true)->CheckCapability("cl_intel_subgroups")) {
						Caffe::set_mode(Caffe::GPU);
						Caffe::SetDevice(gpus[i]);
						bGpumode = true;
						LOG(INFO) << "Use GPU=" << gpus[i];
						break;
				}
			}
		}
#endif  // !CPU_ONLY
	}
	if(!bGpumode){
		LOG(INFO) << "Use CPU!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
		Caffe::set_mode(Caffe::CPU);
	}
	
	pthread_mutex_init(&mutex,NULL); 
	min_batch_=min_batch;
	/* Load the network. */
	net_.reset(new Net<Dtype>(model_file, TEST, Caffe::GetDefaultDevice()));
	net_->CopyTrainedLayersFrom(weights_file);

	Blob<Dtype>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
	SetBatch(min_batch_);
	CreateMean();
	nbatch_index_ = 0;
}

Detector::~Detector() {
	pthread_mutex_destroy(&mutex);  
	EmptyQueue(batchque_);
	EmptyQueue(imgsizeque_);
}

void Detector::EmptyQueue(queue<Blob<Dtype>*>& que)
{
	while(!que.empty()){
		Blob<Dtype>* pdata = que.front();
		que.pop();
		delete pdata;
	}
}

void Detector::EmptyQueue(queue<cv::Size>& que)
{
	while(!que.empty()){
		que.pop();
	}
}

//note! do not call it if Detect not finished
void Detector::SetBatch(int batch)
{
	if(batch<min_batch_ ||batch==num_batch_)
		return;
	Blob<Dtype>* input_layer = net_->input_blobs()[0];
	num_batch_=batch;
	LOG(INFO) << "Change Batch to " << num_batch_ ;
	input_layer->Reshape(num_batch_, num_channels_,input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();
}

bool Detector::TryDetect() {
	bool hasdata;
	pthread_mutex_lock(&mutex); 
	hasdata = !batchque_.empty();
	pthread_mutex_unlock(&mutex); 	
	return hasdata;
}

bool Detector::Detect(vector<vector<Detector::detect_result> >& objects) {
	Blob<Dtype>* pdata;
	pthread_mutex_lock(&mutex); 
	if(!batchque_.empty()){
		pdata = batchque_.front();
		batchque_.pop();
	}
	else{
		pthread_mutex_unlock(&mutex); 	
		return false;
	}
	pthread_mutex_unlock(&mutex); 	
	net_->input_blobs()[0]->ShareData(*pdata);
	
	vector<cv::Size> vimgsize;
	pthread_mutex_lock(&mutex); 
	for (int i=0;i<pdata->num();i++) {
		if(!imgsizeque_.empty()){
			vimgsize.push_back(imgsizeque_.front());
			imgsizeque_.pop();
		}
		else
			vimgsize.push_back(cv::Size(0,0));
	}
	pthread_mutex_unlock(&mutex); 
	
	net_->Forward();
	/* get the result */
	Blob<Dtype>* result_blob = net_->output_blobs()[0];
	const Dtype* result = result_blob->cpu_data();
	const int num_det = result_blob->height();
	for (int k = 0; k < num_det * 7; k += 7) {
		detect_result object;
		object.imgid = (int)result[k + 0];
		int w=vimgsize[object.imgid].width;
		int h=vimgsize[object.imgid].height;		
		object.classid = (int)result[k + 1];
		object.confidence = result[k + 2];
		object.left = (int)(result[k + 3] * w);
		object.top = (int)(result[k + 4] * h);
		object.right = (int)(result[k + 5] * w);
		object.bottom = (int)(result[k + 6] * h);
		if (object.left < 0) object.left = 0;
		if (object.top < 0) object.top = 0;
		if (object.right >= w) object.right = w - 1;
		if (object.bottom >= h) object.bottom = h - 1;
		objects[result[k + 0]].push_back(object);
	}
	delete pdata;
	return true;
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Detector::WrapInputLayer(Blob<Dtype>* pdata) {
	Blob<Dtype>* input_layer = pdata;
	input_channels.clear();
	int width = input_layer->width();
	int height = input_layer->height();
	Dtype* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels()*num_batch_; ++i) {
		input_channels.push_back(input_data);
		input_data += width * height;
	}
}

cv::Mat Detector::PreProcess(const cv::Mat& img) {
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2RGB);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2RGB);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_) {
		cv::resize(sample, sample_resized, input_geometry_);
	}
	else
		sample_resized = sample;

	cv::Mat sample_float;
	//cv::Mat sample_float_sub;
	cv::Mat sample_float_sub_scale;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::scaleAdd (sample_float, 0.007843, mean_, sample_float_sub_scale); //scaleAdd or (add+multiply)? which speed : ans is scaleAdd
	//cv::add(sample_float, -127.5, sample_float_sub);
	//cv::multiply(sample_float_sub, 0.007843, sample_float_sub_scale);
	
	return sample_float_sub_scale;	
}

void Detector::CreateMean() {
	if (num_channels_ == 3)
		mean_= cv::Mat(input_geometry_, CV_32FC3, cv::Scalar(-127.5*0.007843,-127.5*0.007843,-127.5*0.007843));
	else
		mean_= cv::Mat(input_geometry_, CV_32FC1, cv::Scalar(-127.5*0.007843));	
}

/*
	InsertImage will fill a blob until blob full, if full return the blob point
*/
bool Detector::InsertImage(const cv::Mat& orgimg) {
	bool retvalue=false;
	
	cv::Mat img = PreProcess(orgimg);
	
	if(nbatch_index_==0){  //new a blob
		Blob<Dtype>* pbatch_element_=new Blob<Dtype>();
		pbatch_element_->Reshape(num_batch_, num_channels_,input_geometry_.height, input_geometry_.width);		
		WrapInputLayer(pbatch_element_);
		pthread_mutex_lock(&mutex); 
		batchque_.push(pbatch_element_);
		pthread_mutex_unlock(&mutex); 
	}
	pthread_mutex_lock(&mutex); 
	imgsizeque_.push(orgimg.size());
	pthread_mutex_unlock(&mutex); 	
	
	/* Convert the input image to the input image format of the network. */
	int indexB=0+nbatch_index_*num_channels_;
	int indexG=1+nbatch_index_*num_channels_;
	int indexR=2+nbatch_index_*num_channels_;
	for (int i = 0; i < input_geometry_.height; i++) {
		for (int j = 0; j < input_geometry_.width; j++) {
			int pos = i * input_geometry_.width + j;
			if (num_channels_ == 3) {
				cv::Vec3f pixel = img.at<cv::Vec3f>(i, j);
				input_channels[indexB][pos] = pixel.val[2];
				input_channels[indexG][pos] = pixel.val[1];
				input_channels[indexR][pos] = pixel.val[0];  //RGB2BGR
			}
			else {
				cv::Scalar pixel = img.at<float>(i, j);
				input_channels[indexB][pos] = pixel.val[0];
			}
		}
	}
	//if full return pbatch_element_
	if(++nbatch_index_>=num_batch_){
		nbatch_index_=0;
		retvalue = true;	
	}
	return retvalue;
}

//--------------------------------------------------------------------
#define CAMNUM	6
const int FIXED_BATCH=CAMNUM;  //you can set higher than CAMNUM
const int MAXQUEUE=(FIXED_BATCH*2);  //must larger or equal to FIXED_BATCH*2

bool grunning=false;
queue<cv::Mat> gpicque; //if you donot need show, just del all gpicque
queue<int> gpicindexque;
Detector* gpdetector;
pthread_mutex_t mutex ; 
sem_t g_semt;
int total_frame;

void safesleep(int nms)  //I need accurate sleep, so use select
{
	struct timeval delay;
	delay.tv_sec = 0;
	delay.tv_usec = nms * 1000; // n ms
	select(0, NULL, NULL, NULL, &delay);
}

void safewakeup()
{
	/*pthread_mutex_lock(&mutex_condition);  
	pthread_cond_signal(&condition); 
	pthread_mutex_unlock(&mutex_condition);	*/
	sem_post(&g_semt);
}

void safewait()
{
	/*pthread_mutex_lock(&mutex_condition);  
	pthread_cond_wait(&condition,&mutex_condition); 
	pthread_mutex_unlock(&mutex_condition);	*/
	sem_wait(&g_semt);
}

void *thr_camera(void *arg)
{
	int id=*((int*)arg);
	if(id==0)
	{
		cv::VideoCapture cap(0);
		if (!cap.isOpened()) {
			LOG(FATAL) << "can not open camera";
			cap.release(); 
			grunning=false;
			safewakeup();
			return NULL;
		}
		cv::Mat frame;
		while (grunning){
			cap >> frame;
			pthread_mutex_lock(&mutex);  
			if(gpicindexque.size()<MAXQUEUE){
				gpicque.push(frame);
				gpicindexque.push(id);
				if(gpdetector->InsertImage(frame)){
					safewakeup(); 
				}
			}
			pthread_mutex_unlock(&mutex);  
		}
		cap.release(); 
	}
	else  //just fake , demo , I have no 6 cameras
	{
		safesleep(id+3);
		while (grunning){
			safesleep(20);  //50 fps
			pthread_mutex_lock(&mutex);  
			if(gpicindexque.size()<MAXQUEUE && !gpicindexque.empty()){
				gpicque.push(gpicque.back().clone());
				gpicindexque.push(id);
				if(gpdetector->InsertImage(gpicque.back())){
					safewakeup(); 
				}
			}		
			pthread_mutex_unlock(&mutex);  
		}
	}
}

void *thr_detector(void *arg)
{
	int fc=0;
	while (grunning){
		safewait();
		if(!grunning) break;
		if(!gpdetector->TryDetect()){
			std::cout << "no data to detect \n" ;
			continue;
		}		
		pthread_mutex_lock(&mutex); 
		for(int i=0;i<FIXED_BATCH;i++){
			gpicque.pop();
			gpicindexque.pop();
		}
		pthread_mutex_unlock(&mutex); 
		
		vector<vector<Detector::detect_result> > objects(gpdetector->GetCurBatch());
		gpdetector->Detect(objects);
		
		total_frame+=FIXED_BATCH;	
	}
}

void *thr_fps(void *arg)
{
	while(grunning){
		safesleep(1000*10);//10s
		std::cout << "Cur fps=" << total_frame/10.0 << "\n";
		total_frame = 0;
	}
}

void emptyqueue(queue<Blob<Dtype>*>& que)
{
	while(!que.empty()){
		Blob<Dtype>* pdata = que.front();
		que.pop();
		delete pdata;
	}
}

int main(int argc, char** argv) {

	int i;
	if (argc < 3) {
		return 1;
	}
	google::InitGoogleLogging(argv[0]);
	FLAGS_minloglevel =3; //0~3, 0 will enable all logs
	const string& model_file = argv[1];
	const string& weights_file = argv[2];

	grunning=true;
	total_frame = 0;
	pthread_t nctid[CAMNUM];
	pthread_t nrtid;
	pthread_t nttid;
	pthread_mutex_init(&mutex,NULL);  
	sem_init(&g_semt, 0, 0);
	
	std::cout << "Opencv is using Opencl? " << cv::ocl::useOpenCL() << "\n";
	cv::ocl::setUseOpenCL(false);
	std::cout << "Now set to " << cv::ocl::useOpenCL() << "\n";
	// Initialize the network.
	Detector detector(model_file, weights_file,FIXED_BATCH);
	gpdetector = &detector;
	pthread_create(&nrtid, NULL, thr_detector, (void*)(&detector));
	pthread_create(&nttid, NULL, thr_fps, NULL);
	
	//open input
	int id[CAMNUM];
	for(i=0;i<CAMNUM;i++){
		id[i]=i; //void* --> int must add -fpermissive...
		pthread_create(&nctid[i], NULL, thr_camera, (void*)&id[i]);
	}
	
	//wait quit
	while(true){
		std::cout << "Enter q to quite: " ;
		int c=getchar();
		if (c=='q' || c=='Q'){
			grunning=false;
			safewakeup();
			break;
		}
	}
	
	for(i=0;i<CAMNUM;i++)
		pthread_join(nctid[i],NULL);
	pthread_join(nrtid,NULL);
	pthread_cancel(nttid);
	pthread_join(nttid,NULL);
	pthread_mutex_destroy(&mutex);  
	sem_destroy(&g_semt);
	//emptyqueue(gpicque);
	//emptyqueue(gpicindexque);

	LOG(INFO) << "Done";
	return 0;
}
#else
int main(int argc, char** argv) {
	LOG(FATAL) << "This example requires OpenCV and half floating point support."
		<< "compile with USE_OPENCV and USE_ISAAC.";
}
#endif  // USE_OPENCV
