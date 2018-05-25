//
// Created by chaoz on 23/05/18.
//

#include <ros/ros.h>
#include <ros/package.h>
#include <tiny_dnn/tiny_dnn.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>

using namespace boost::filesystem;
using namespace tiny_dnn;
using namespace tiny_dnn::layers;
using namespace tiny_dnn::activation;

std::string pathCones;
int numCones;
std::string pathNonCones;
int numNonCones;
int image_w;
int image_h;
double process_scale;
double train_ratio;
int batch_size;
int epochs;
double alpha;


// convert image to vec_t
void convert_image(const std::string& imagefilename,
                   double scale,
                   int w,
                   int h,
                   std::vector<vec_t>& data)
{
    auto img = cv::imread(imagefilename, cv::IMREAD_GRAYSCALE);
    if (img.data == nullptr) return; // cannot open, or it's not an image

    cv::Mat_<uint8_t> resized;
    cv::resize(img, resized, cv::Size(w, h));
    vec_t d;

    std::transform(resized.begin(), resized.end(), std::back_inserter(d),
                   [=](uint8_t c) { return c * scale; });
    data.push_back(d);
}

// convert all images found in directory to vec_t
void convert_images(const std::string& directory,
                    int num_images,
                    double scale,
                    int w,
                    int h,
                    std::vector<vec_t>& data)
{
    for(int i =0;i<num_images;i++) {
        std::string img_path = directory + "image" + boost::lexical_cast<std::string>(i) +".png";
        convert_image(img_path, scale, w, h, data);
    }
}

void Configure(ros::NodeHandle n) {
    if(!n.getParam("/parser/path_cones",pathCones)) {
        ROS_ERROR_STREAM("path_cones is missing!"<<pathCones);
//        exit(0);
    }
    if(!n.getParam("/parser/num_cones",numCones)) {
        ROS_ERROR("num_cones is missing!");
        exit(0);
    }
    if(!n.getParam("/parser/path_non_cones",pathNonCones)) {
        ROS_ERROR("path_non_cones is missing!");
        exit(0);
    }
    if(!n.getParam("/parser/num_non_cones",numNonCones)) {
        ROS_ERROR("num_non_cones is missing!");
        exit(0);
    }
    if(!n.getParam("/parser/image_width",image_w)) {
        ROS_ERROR("image width is missing!");
        exit(0);
    }
    if(!n.getParam("/parser/image_height",image_h)) {
        ROS_ERROR("image height is missing!");
        exit(0);
    }
    if(!n.getParam("/parser/process_scale",process_scale)) {
        ROS_ERROR("process_scale is missing!");
        exit(0);
    }
    if(!n.getParam("/parser/train_ratio",train_ratio)) {
        ROS_ERROR("train_ratio is missing!");
        exit(0);
    }
    if(!n.getParam("/parser/batch_size",batch_size)) {
        ROS_ERROR("batch_size is missing!");
        exit(0);
    }
    if(!n.getParam("/parser/epochs",epochs)) {
        ROS_ERROR("epochs is missing!");
        exit(0);
    }
    if(!n.getParam("/parser/alpha",alpha)) {
        ROS_ERROR("alpha is missing!");
        exit(0);
    }
    bool absolute_path = true;
    if(!n.getParam("/parser/absolute_path",absolute_path)) {
        ROS_ERROR("absolute_path is missing!");
        exit(0);
    }
    if(!absolute_path) {
        std::string pkg_path = ros::package::getPath("vision")+"/";
        pathCones.insert(0,pkg_path);
        pathNonCones.insert(0,pkg_path);
    }
    ROS_INFO("configure done");
}

int main(int argc, char** argv) {
    ros::init(argc,argv,"parser");
    ros::NodeHandle n;
    Configure(n);
    std::vector<vec_t> alldata;
    std::vector<label_t > alllabel;
    std::vector<vec_t> data;
    std::vector<label_t> label;
    std::vector<vec_t> test_data;
    std::vector<label_t > test_label;
    convert_images(pathCones,numCones,process_scale,image_w,image_h,data);
    for(int i = 0; i<data.size();i++) {
        label_t tmp;
        tmp = 1;
        label.push_back(tmp);
    }
    convert_images(pathNonCones,numNonCones,process_scale,image_w,image_h,data);
    for(int i = 0;data.size()>label.size();i++) {
        label_t tmp;
        tmp = 0;
        label.push_back(tmp);
    }
    std::cout<<"data loaded: " << data.size() << " | " << data.front().size() << "x" << data.front().front()<<std::endl;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle(data.begin(), data.end(), std::default_random_engine(seed));
    shuffle(label.begin(), label.end(), std::default_random_engine(seed));
    std::cout<<"shuffle data\n";
    size_t test_size = data.size()*(1-train_ratio);
    test_data.reserve(test_size);
    test_label.reserve(test_size);
    std::cout<<"reserve vector\n";
    test_data.insert(test_data.end(),data.begin(),data.begin()+test_size);
    test_label.insert(test_label.end(),label.begin(),label.begin()+test_size);
    std::cout<<"insert vector\n";
    data.erase(data.begin(),data.begin()+test_size);
    label.erase(label.begin(),label.begin()+test_size);
    std::cout<<"train data shuffle: " << data.size()<<std::endl;
    std::cout<<"test data shuffle: " << test_data.size()<<std::endl;

    network<sequential> net;
    adagrad opt;
    net << conv(32,32,5,1,3) << relu()
        << max_pooling_layer(28,28,3,2) << relu()
        << conv(14,14,5,3,3) << relu()
        << max_pooling_layer(10,10,3,2) << relu()
        << fc(5*5*3, 120) << tanh_layer()
        << fc(120,2)
        << softmax()
            ;
    std::cout<<"network setup"<<std::endl;
    for (int i = 0; i < net.depth(); i++) {
        std::cout << "#layer:" << i << "\n";
        std::cout << "layer type:" << net[i]->layer_type() << "\n";
        std::cout << "input:" << net[i]->in_data_size() << "(" << net[i]->in_data_shape() << ")\n";
        std::cout << "output:" << net[i]->out_data_size() << "(" << net[i]->out_data_shape() << ")\n";
    }
    int epoch = 0;
    timer t;
    std::cout<< "init alpha:"<<opt.alpha;
    opt.alpha *= alpha;
    std::cout<< " current alpha:"<<opt.alpha<<std::endl;
    net.fit<mse>(opt,data,label,batch_size,epochs,
                 [&](){
//                     std::cout<<t.elapsed()<<std::endl;
                     t.restart();
                 },
                 [&](){
                result res = net.test(test_data,test_label);
                std::cout<< (double)res.num_success/(double)res.num_total<<std::endl;
//                std::ofstream ofs (("epoch_"+to_string(epoch++)).c_str());
//                ofs << net;
            });

    return 0;
}