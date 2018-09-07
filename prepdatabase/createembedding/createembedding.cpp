/*
 *  Copyright (C) 2015-2018 Savoir-faire Linux Inc.
 *  
 *  Author: Timothée Menais <timothee.menais@savoirfairelinux.com>
 *  
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.
 */
#include "/usr/local/include/opencv2/objdetect.hpp"
#include "/usr/local/include/opencv2/highgui.hpp"
#include "/usr/local/include/opencv2/imgproc.hpp"
#include <iostream>
#include <fstream>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/client/client_session.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <experimental/filesystem>

using std::experimental::filesystem::recursive_directory_iterator;
 
using namespace std;
using namespace cv;
using namespace tensorflow;

string input_layer = "input";
string phase_train_layer = "phase_train";
string output_layer = "embeddings";



static tensorflow::Status ReadEntireFile(tensorflow::Env* env, const std::string& filename,
                             Tensor* output) {
    tensorflow::uint64 file_size = 0;
    TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

    std::string contents;
    contents.resize(file_size);

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

    tensorflow::StringPiece data;
    TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
    if (data.size() != file_size) {
    return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                        "' expected ", file_size, " got ",
                                        data.size());
    }
    output->scalar<std::string>()() = data.ToString();
    return Status::OK();
}


tensorflow::Status LoadGraph(const std::string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
        ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
            if (!load_graph_status.ok()) {
            return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
    }
    
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }

    return Status::OK();
}

std::unique_ptr<tensorflow::Session> initSession(std::string graphpath){
    std::unique_ptr<tensorflow::Session> session;
    Status load_graph_status = LoadGraph(graphpath, &session);
    
    if (!load_graph_status.ok()) {}   
  
    return session;
}

std::vector<std::string> ReadLabelsAndFile(const std::string file_name) {
    
    std::ifstream infile(file_name);
    std::string label_file;
    std::vector<std::string> labels_files;
    if(infile.is_open()){
        while(std::getline(infile, label_file)){
            labels_files.push_back(label_file);
        }
          infile.close();
    }
    return labels_files;
}



int main( int argc, char** argv )
{
    if( argc != 3)
    {
     cout <<" Usage: give a folder and label file" << endl;
     return -1;
    }
    std::vector<std::string> database = ReadLabelsAndFile(argv[2]);
    std::vector<std::string> label_database  = ReadLabelsAndFile(argv[2]); // segfault if not initialized
    std::vector<std::string> file_database = ReadLabelsAndFile(argv[2]); // segfault if not initialized


    std::unique_ptr<tensorflow::Session> session = initSession("../20180408-102900.pb");

    tensorflow::Tensor phase_tensor(tensorflow::DT_BOOL, tensorflow::TensorShape());
    phase_tensor.scalar<bool>()() = false;

    CascadeClassifier cascade;
    cascade.load("../haarcascade_frontalface_default.xml");

    ofstream myfile;
    myfile.open ("../face_embeddings_database.txt");

    for (int i = 0; i < database.size(); ++i)
        {
            std::string mystring = database[i];
            label_database[i] = mystring.substr(0, mystring.find_first_of(" "));
            file_database[i] = mystring.substr(mystring.find_first_of(" ")+1);


            //write string with correct path
            
            std::string imtoread = string(argv[1]);

            imtoread.append("/"); imtoread.append(label_database[i]); imtoread.append("/"); imtoread.append(file_database[i]);
            
            std::cout << "processing :" << imtoread << '\n';   
        
            Mat image = imread(imtoread, CV_LOAD_IMAGE_COLOR);   
            if(! image.data )                // Check for invalid input
            {
                cout <<  "Could not open or find the image" << std::endl ;
                return -1;
            }

            vector<Rect> faces;
            double scale=1; //modify if needed
            
            double fx = 1 / scale;
            Mat smallImg;
            
            resize( image, smallImg, Size(), fx, fx, CV_INTER_LINEAR );

            
            cascade.detectMultiScale( smallImg, faces, 1.1, 
                                    7, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

            //std::cout << "after cascade"  << '\n';

            if (faces.size() !=1)
            {
                cout <<  "There should be exactly 1 face" << std::endl ;
               
            }else{               

                if(! smallImg.data )                // Check for invalid input
                {
                    cout <<  "Could not open or find the image" << std::endl ;
                    return -1;
                }

                Rect r = faces[0];

                Mat smallImgROI = smallImg(r);

                resize( smallImgROI, smallImgROI, cv::Size(160, 160), CV_INTER_LINEAR); // model needs image 160*160 pix

                if(! smallImgROI.data )                // Check for invalid input
                {
                    cout <<  "Could not open or find the image" << std::endl ;
                    return -1;
                }

                auto data = smallImgROI.data;
                Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,smallImgROI.rows,smallImgROI.cols,3}));
                auto input_tensor_mapped = input_tensor.tensor<float, 4>(); 

                for (int x = 0; x < smallImgROI.cols; ++x) {
                    for (int y = 0; y < smallImgROI.rows; ++y) {
                        for (int c = 0; c < 3; ++c) {
                            int offset = y * smallImgROI.cols + x * 3 + c;
                            input_tensor_mapped(0, y, x, c) = tensorflow::uint8(data[offset]);
                        }
                    }
                }

                std::vector<Tensor> outputs ;
                Status run_status = session->Run({{input_layer, input_tensor},
                                               {phase_train_layer, phase_tensor}}, 
                                               {output_layer}, {}, &outputs);

                auto output_c = outputs[0].tensor<float, 2>();
                myfile << label_database[i] << " ";
                
                for (int i = 0; i <  outputs[0].shape().dim_size(1); ++i)
                {
                    myfile << output_c(0,i) << " ";
                }
                
                myfile << "\n" ;
            }
        }

    myfile.close();

    return 0;
}