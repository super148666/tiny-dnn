//
// Created by chaoz on 23/05/18.
//

#include <ros/ros.h>
#include "tiny_dnn/tiny_dnn.h"


using namespace tiny_dnn;
using namespace tiny_dnn::layers;
using namespace tiny_dnn::activation;


int main(int argc, char** argv) {
    ros::init(argc,argv,"trainer");
    ros::NodeHandle n;

    network<sequential> net;

    net << conv(32, 32, 5, 1, 6, padding::same) << tanh_layer()  // in:32x32x1, 5x5conv, 6fmaps
        << max_pool(32, 32, 6, 2) << tanh_layer()                // in:32x32x6, 2x2pooling
        << conv(16, 16, 5, 6, 16, padding::same) << tanh_layer() // in:16x16x6, 5x5conv, 16fmaps
        << max_pool(16, 16, 16, 2) << tanh_layer()               // in:16x16x16, 2x2pooling
        << fc(8*8*16, 100) << tanh_layer()                       // in:8x8x16, out:100
        << fc(100, 10) << softmax();                       // in:100 out:10

    adagrad opt;

    int epochs = 50;
    int batch = 20;
    //net.fit<cross_entropy>(opt, x_data, y_data, batch, epochs);

}