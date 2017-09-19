#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/ocl.hpp>
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
#include "kernels.hpp"

#if defined(USE_OPENCV) && defined(HAS_HALF_SUPPORT)
using namespace caffe;  // NOLINT(build/namespaces)
using caffe::Timer;
using std::queue;  

#define Dtype float

class Detector {
public:
	typedef struct __resultbox {
		int imgid;
		float classid;
		float confidence;
		float left;
		float right;
		float top;
		float bottom;
	}resultbox;
	
	typedef struct __Result {
		vector<resultbox> boxs;
		cv::Mat orgimg;
		cv::Size imgsize;
	}Result;
	
	Detector(const string& model_file,
		const string& weights_file,int min_batch,bool keep_orgimg);
	~Detector();

	bool Detect(vector<Result>& objects);
	inline int GetCurBatch(){return  num_batch_;}
	bool InsertImage(const cv::Mat& orgimg);
	void SetBatch(int batch);
	int TryDetect();
	void PreprocessGPU(float* imgdata);

private:
	void WrapInputLayer(Blob<Dtype>* pdata);
	cv::Mat PreProcess(const cv::Mat& img);
	void CreateMean();
	void EmptyQueue(queue<Blob<Dtype>*>& que);
	void EmptyQueue(queue<cv::Size>& que);
	void EmptyQueue(queue<cv::Mat>& que);
	shared_ptr<Net<Dtype> > net_;
	std::vector<Dtype *> input_channels;
	cv::Size input_geometry_;
	queue<Blob<Dtype>*> batchque_;
	queue<cv::Size> imgsizeque_;
	queue<cv::Mat> imgque_;
	pthread_mutex_t mutex ; 
	int nbatch_index_;
	int num_channels_;
	int min_batch_;
	int num_batch_;
	cv::Mat mean_;
	int gpuid_;
	bool keep_orgimg_;
	int max_imgqueue_;
	int curdata_batch_;
	viennacl::ocl::program fp16_ocl_program_;
	Blob<Dtype>* pbatch_element_;
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
	const string& weights_file,int min_batch,bool keep_orgimg) {
	// Set device id and mode
	vector<int> gpus;
	get_gpus(&gpus);
	Caffe::SetDevices(gpus);
	gpuid_ = -1;
	if (gpus.size() != 0) {
#ifndef CPU_ONLY
		for (int i = 0; i < gpus.size(); i++) {
			if (Caffe::GetDevice(gpus[i], true)->backend() == BACKEND_OpenCL) {
				if (Caffe::GetDevice(gpus[i], true)->CheckVendor("Intel")
					&& Caffe::GetDevice(gpus[i], true)->CheckType("GPU")) {
					//&& Caffe::GetDevice(gpus[i], true)->CheckCapability("cl_intel_subgroups")) {
						Caffe::set_mode(Caffe::GPU);
						Caffe::SetDevice(gpus[i]);
						gpuid_ = gpus[i];
						//fp16_ocl_program_ = RegisterKernels<Dtype>(&(viennacl::ocl::get_context(static_cast<uint64_t>(gpuid_))));
						LOG(INFO) << "Use GPU=" << gpus[i];
						break;
				}
			}
		}
#endif  // !CPU_ONLY
	}
	if(gpuid_<0){
		LOG(FATAL) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!Use CPU!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
		Caffe::set_mode(Caffe::CPU);
	}
	
	pthread_mutex_init(&mutex,NULL); 
	min_batch_=min_batch;
	keep_orgimg_ = keep_orgimg;
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
	net_->Forward(); //just warm up
}

Detector::~Detector() {
	pthread_mutex_destroy(&mutex);  
	EmptyQueue(batchque_);
	EmptyQueue(imgsizeque_);
	EmptyQueue(imgque_);
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

void Detector::EmptyQueue(queue<cv::Mat>& que)
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
	max_imgqueue_=num_batch_*2;
	LOG(INFO) << "Change Batch to " << num_batch_ ;
	input_layer->Reshape(num_batch_, num_channels_,input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();
}

int Detector::TryDetect() {
	int curbatch=0;
	pthread_mutex_lock(&mutex); 
	if(!batchque_.empty())
		curbatch=batchque_.front()->num();
	pthread_mutex_unlock(&mutex); 	
	return curbatch;
}

bool Detector::Detect(vector<Detector::Result>& objects) {
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

	for (int i=0;i<pdata->num();i++) {
		if(!imgsizeque_.empty()){
			objects[i].imgsize = imgsizeque_.front();
			imgsizeque_.pop();
		}
		else
			objects[i].imgsize = cv::Size(0,0);
		
		if(keep_orgimg_ && !imgque_.empty()){
			objects[i].orgimg = imgque_.front();
			imgque_.pop();
		}
	}
	pthread_mutex_unlock(&mutex); 
	
	net_->input_blobs()[0]->ShareData(*pdata);
	net_->Forward();
	/* get the result */
	Blob<Dtype>* result_blob = net_->output_blobs()[0];
	const Dtype* result = result_blob->cpu_data();
	const int num_det = result_blob->height();
	for (int k = 0; k < num_det * 7; k += 7) {
		resultbox object;
		object.imgid = (int)result[k + 0];
		int w=objects[object.imgid].imgsize.width;
		int h=objects[object.imgid].imgsize.height;		
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
		objects[object.imgid].boxs.push_back(object);
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

void Detector::PreprocessGPU(float* imgdata)
{
	int n=input_geometry_.height*input_geometry_.width*num_channels_;
	float scale = 0.007843;
	float mean = -127.5;
	float* in_data;
	Dtype* out_data=pbatch_element_->mutable_gpu_data()+nbatch_index_*n;
	
	Blob<float> in_blob;
	in_blob.Reshape(1, num_channels_,input_geometry_.height, input_geometry_.width);
	in_blob.set_cpu_data(imgdata);
	in_blob.data().get()->async_gpu_push();
	in_data = in_blob.mutable_gpu_data();
	
	viennacl::ocl::context &ctx = viennacl::ocl::get_context(gpuid_);
	// Execute kernel
	viennacl::ocl::kernel &oclk_preprocess = fp16_ocl_program_.get_kernel(
		CL_KERNEL_SELECT("preprocess"));
	viennacl::ocl::enqueue(
		oclk_preprocess(n, scale, mean, WrapHandle((cl_mem)in_data, &ctx),
		WrapHandle((cl_mem)out_data, &ctx)),
		ctx.get_queue());
}

/*
	InsertImage will fill a blob until blob full, if full return the blob point
*/
bool Detector::InsertImage(const cv::Mat& orgimg) {
	bool retvalue=false;

	pthread_mutex_lock(&mutex); 
	if(imgsizeque_.size()>=max_imgqueue_){
		pthread_mutex_unlock(&mutex); 	
		return false;
	}
	pthread_mutex_unlock(&mutex); 		
	
	cv::Mat img = PreProcess(orgimg);	
	
	if(nbatch_index_==0){  //new a blob
		pbatch_element_=new Blob<Dtype>();
		pbatch_element_->Reshape(num_batch_, num_channels_,input_geometry_.height, input_geometry_.width);		
		curdata_batch_ = num_batch_;
		WrapInputLayer(pbatch_element_);
		pthread_mutex_lock(&mutex); 
		batchque_.push(pbatch_element_);
		pthread_mutex_unlock(&mutex); 
	}
	
	pthread_mutex_lock(&mutex); 
	imgsizeque_.push(orgimg.size());
	if(keep_orgimg_)
		imgque_.push(orgimg);
	pthread_mutex_unlock(&mutex); 	
	
	/* Convert the input image to the input image format of the network. */
	if(true){
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
	}
	else{
		CHECK(img.isContinuous())
			<< "Model size must 4X, if not, cv::Mat will not Continuous";
		PreprocessGPU((float*)img.data);
	}
	//if full return pbatch_element_
	if(++nbatch_index_>=curdata_batch_){
		nbatch_index_=0;
		retvalue = true;	
	}
	return retvalue;
}

//--------------------------------------------------------------------
#define CAMNUM	6
const int FIXED_BATCH=CAMNUM;  //you can set higher than CAMNUM
bool grunning=false;
Detector* gpdetector;
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
		if(gpdetector->InsertImage(frame)){
			safewakeup(); 
		}
		for(int i=0;i<CAMNUM-1;i++){
			//safesleep(5);  //different time sleep ,  fps from 80~130 ...  in fp16
			if(gpdetector->InsertImage(frame)){
				safewakeup(); 
			}			
		}
	}
	cap.release(); 
}

void *thr_detector(void *arg)
{
	int fc=0;
	int curbatch;
	while (grunning){
		safewait();
		if(!grunning) break;
		if((curbatch=gpdetector->TryDetect())<=0){
			std::cout << "no data to detect \n" ;
			continue;
		}		
		vector<Detector::Result> objects(curbatch);
		gpdetector->Detect(objects);
		total_frame+=curbatch;	
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
	pthread_t nctid;
	pthread_t nrtid;
	pthread_t nttid;
	sem_init(&g_semt, 0, 0);
	
	std::cout << "Opencv is using Opencl? " << cv::ocl::useOpenCL() << "\n";
	//cv::ocl::setUseOpenCL(false);
	//std::cout << "Now set to " << cv::ocl::useOpenCL() << "\n";
	// Initialize the network.
	Detector detector(model_file, weights_file,FIXED_BATCH,true);
	gpdetector = &detector;
	pthread_create(&nrtid, NULL, thr_detector, (void*)(&detector));
	pthread_create(&nttid, NULL, thr_fps, NULL);
	pthread_create(&nctid, NULL, thr_camera, NULL);
	
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
	
	pthread_join(nctid,NULL);
	pthread_join(nrtid,NULL);
	pthread_cancel(nttid);
	pthread_join(nttid,NULL);
	sem_destroy(&g_semt);

	LOG(INFO) << "Done";
	return 0;
}
#else
int main(int argc, char** argv) {
	LOG(FATAL) << "This example requires OpenCV and half floating point support."
		<< "compile with USE_OPENCV and USE_ISAAC.";
}
#endif  // USE_OPENCV
