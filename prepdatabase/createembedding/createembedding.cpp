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
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>

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

//using std::experimental::filesystem::recursive_directory_iterator;
 

using namespace tensorflow;

using namespace std;
using namespace cv;
using namespace cv::face;

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

cv::Mat faceCenterRotateCrop(Mat &im, vector<Point2f> landmarks, Rect face , int i){


    //description of the landmarks in case someone wants to do something custom
    // landmarks 0, 16           // Jaw line
    // landmarks 17, 21          // Left eyebrow
    // landmarks 22, 26          // Right eyebrow
    // landmarks 27, 30          // Nose bridge
    // landmarks 30, 35          // Lower nose
    // landmarks 36, 41          // Left eye
    // landmarks 42, 47          // Right Eye
    // landmarks 48, 59          // Outer lip
    // landmarks 60, 67          // Inner lip


    // 2D image points. If you change the image, you need to change vector
    std::vector<cv::Point2d> image_points;
    image_points.push_back(landmarks[30]);    // Nose tip
    image_points.push_back(landmarks[8]);    // Chin
    image_points.push_back(landmarks[45]);     // Left eye left corner
    image_points.push_back(landmarks[36]);    // Right eye right corner
    image_points.push_back(landmarks[54]);    // Left Mouth corner
    image_points.push_back(landmarks[48]);    // Right mouth corner

    // 3D model points.
    std::vector<cv::Point3d> model_points;
    model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));               // Nose tip
    model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));          // Chin
    model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));       // Left eye left corner
    model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));        // Right eye right corner
    model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));      // Left Mouth corner
    model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f)); // Right mouth corner

    // Camera internals
    double focal_length = im.cols; // Approximate focal length. //3 nb channels
    Point2d center = cv::Point2d(im.cols/2,im.rows/2);
    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
    cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type); // Assuming no lens distortion

    cv::Mat rotation_vector; // Rotation in axis-angle form
    cv::Mat translation_vector;
         
    // Solve for pose
    cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

    // Access the last element in the Rotation Vector
    double rot = rotation_vector.at<double>(0,2);
    double theta_deg = rot/M_PI*180;
    Mat dst;

    // Rotate around the center
    
    Point2f pt = landmarks[30]; //center is nose tip
    Mat r = getRotationMatrix2D(pt, theta_deg, 1.0);
    // determine bounding rectangle
    cv::Rect bbox = cv::RotatedRect(pt,im.size(), theta_deg).boundingRect();

    // Apply affine transform
    warpAffine(im, dst, r, bbox.size());

    // Now crop the face
    Mat Cropped_Face = dst(face);

    resize( Cropped_Face, Cropped_Face, cv::Size(160, 160), CV_INTER_LINEAR);

    return Cropped_Face ;
    //std::string text = "Cropped Face ";
    //text += std::to_string(i);

    //imshow(text,Cropped_Face);
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

    //if [ ! -d ../data/../plop ]; then mkdir ../plop ; fi ;

    string folder = "/../aligned_data_base";

    stringstream call_line;

    call_line << "if [ ! -d " << argv[1] << folder << " ]; then mkdir " << argv[1] << folder << " ; fi ;";
    //call_line << "if [ ! -d " << argv[1] << folder << " ]; then echo plop ; fi ;";

    system(call_line.str().c_str());


  
    for (int i = 0; i < label_database.size(); ++i)
    {
        stringstream second_call_line;
        //second_call_line << "if [ ! -d " << argv[1] << folder  << " ]; then echo plop ; fi ;";
        second_call_line << "if [ ! -d " << argv[1] << folder << "/" << database[i].substr(0, database[i].find_first_of(" ")) << " ]; then mkdir " << argv[1] << folder << "/" << database[i].substr(0, database[i].find_first_of(" ")) << " ; fi ;";


        system(second_call_line.str().c_str());
    }


    std::unique_ptr<tensorflow::Session> session = initSession("../20170512-110547.pb");

    tensorflow::Tensor phase_tensor(tensorflow::DT_BOOL, tensorflow::TensorShape());
    phase_tensor.scalar<bool>()() = false;

    CascadeClassifier cascade;
    cascade.load("../haarcascade_frontalface_alt2.xml");

    Ptr<Facemark> facemark = FacemarkLBF::create();
    facemark->loadModel("../lbfmodel.yaml");

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

                //Mat smallImgROI = smallImg(r);

                Mat detectlandmarks;

                vector<vector<Point2f>> landmarks;

                //cvtColor(image, detectlandmarks, COLOR_BGR2GRAY);
                
                // Run landmark detector
                bool success = facemark->fit(smallImg,faces,landmarks);

                Mat smallImgROI ;

                if(success)
                {
                // If successful, render the landmarks on the face
                    for(int i = 0; i < landmarks.size(); i++)
                    {
                        if (landmarks[i].size()==68)
                        {
                            smallImgROI = faceCenterRotateCrop(smallImg,landmarks[i],faces[i],i);
                        }

                    }
                }

                //resize( smallImgROI, smallImgROI, cv::Size(160, 160), CV_INTER_LINEAR); // model needs image 160*160 pix

                if(! smallImgROI.data )                // Check for invalid input
                {
                    cout <<  "Could not open or find the image" << std::endl ;
                    return -1;
                }

                std::string aligned_data_base = string(argv[1]);
                aligned_data_base.append("/../aligned_data_base/"); aligned_data_base.append(label_database[i]); aligned_data_base.append("/"); aligned_data_base.append(file_database[i]);
                imwrite(aligned_data_base, smallImgROI);

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