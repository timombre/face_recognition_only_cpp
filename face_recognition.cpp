// CPP program to detects face in a video
 
// Include required header files from OpenCV directory
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
    
    if (!load_graph_status.ok()) {
    }   
  
    return session;
}

std::vector<std::string> ReadLabelsAndEmbeddings(const std::string file_name) {
    
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

std::vector<float> ConvStringToFloats(std::string str){

    std::vector<float> vect;
    std::istringstream stm(str) ;
    
    float number;

    while(stm >> number){
       
                vect.push_back(number);
                            
    }

    return vect;
}

 
// Function for Face Detection
void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    double scale, std::unique_ptr<tensorflow::Session>* session,
                    std::vector<std::string> label_database,
                    std::vector<std::string> embeddings_database)
{
    vector<Rect> faces;
    Mat smallImg;
    double fx = 1 / scale; 
    resize( img, smallImg, Size(), fx, fx, CV_INTER_LINEAR ); 
   
    cascade.detectMultiScale( smallImg, faces, 1.1, 
                            7, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    tensorflow::Tensor phase_tensor(tensorflow::DT_BOOL, tensorflow::TensorShape());
    phase_tensor.scalar<bool>()() = false;

    int linewidth = std::max(1, int(img.rows * .005));
 
    // Draw boxes around the faces
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i];
        
        Point center;
        Scalar color = Scalar(255, 0, 0); // Color for Drawing tool

        rectangle( img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
                cvPoint(cvRound((r.x + r.width-1)*scale), 
                cvRound((r.y + r.height-1)*scale)), color, linewidth, 8, 0);

        Mat smallImgROI = smallImg(r);
        

        cv::resize(smallImgROI, smallImgROI, cv::Size(160, 160), CV_INTER_LINEAR);//network needs pictures of size 160X160 pixels

          
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
        Status run_status = session->get()->Run({{input_layer, input_tensor},
                                       {phase_train_layer, phase_tensor}}, 
                                       {output_layer}, {}, &outputs);

        if (!run_status.ok()) {
            LOG(ERROR) << "\tRunning model failed: " << run_status << "\n";
            return ;
        }

        auto output_c = outputs[0].tensor<float, 2>();

        float min_emb_diff =10000;
        
        int posofmin=0;


        for (int i = 0; i < label_database.size(); ++i){

            float diff=0;

            auto vect = ConvStringToFloats(embeddings_database[i]);


            for (int j = 0; j < outputs[0].shape().dim_size(1); ++j){
                diff += (output_c(0,j) - vect[j]) * (output_c(0,j) - vect[j]) ;
            }

            diff= diff; //no need to sqrt

            if (diff < min_emb_diff)
            {
                min_emb_diff = diff ;
                posofmin = i ;
            }
                
        }

        cv::Point txt_up = cvPoint(cvRound(r.x*scale + linewidth ), cvRound(r.y*scale - 4 * linewidth));      
        cv::Point txt_in = cvPoint(cvRound(r.x*scale + linewidth ), cvRound(r.y*scale + 12 * linewidth));

        if(min_emb_diff < 0.0009) {
            cout <<"Hello " << label_database[posofmin] << " confidence: " << min_emb_diff << endl;
            if ( cvRound(r.y*scale -12 * linewidth) > 0 )
            {
                cv::putText(img, label_database[posofmin], txt_up, 1, linewidth , color );
            }else{ // write in box if box is too high
                cv::putText(img, label_database[posofmin], txt_in, 1, linewidth , color );
            }
        }else{
            cout <<"WHO ARE YOU ?"<< " confidence: " << min_emb_diff << endl;
            if ( cvRound(r.y*scale - 12 * linewidth) > 0 )
            {
                cv::putText(img, "404", txt_up, 1, linewidth , color );
            }else{
                cv::putText(img, "404", txt_in, 1, linewidth , color );
            }
        }

    }
 
    // Show Processed Image with detected faces
    imshow( "Face Detection", img );
}


int main( int argc, const char** argv )
{

    if( argc != 2)
    {
        cout <<" Provide your embeddings database" << endl;
        return -1;
    }

    std::vector<std::string> database = ReadLabelsAndEmbeddings(argv[1]);
    std::vector<std::string> label_database  = database; // segfault if not initialized
    std::vector<std::string> embeddings_database = database; // segfault if not initialized
    std::vector<std::vector<float>> embeddings_float;
    embeddings_float.reserve(database.size() * 512); //512 embeddings


    for (int i = 0; i < label_database.size(); ++i){

            std::string mystring = database[i];
            label_database[i] = mystring.substr(0, mystring.find_first_of(" "));
            embeddings_database[i] = mystring.substr(mystring.find_first_of(" ")+1);
            //std::cout << embeddings_database[i]  << '\n';
            //embeddings_float[i]=ConvStringToFloats(embeddings_database[i]);
            //std::cout <<label_database[i] << " " << embeddings_database[i]  << '\n';
    }

    



    
    // VideoCapture class for playing video for which faces to be detected
    VideoCapture capture; 
    Mat frame;
 
    // PreDefined trained XML classifiers with facial features
    CascadeClassifier cascade; //, nestedCascade; 
    double scale=1;
 
    // Load classifiers from "opencv/data/haarcascades" directory 
    //nestedCascade.load( "/home/tmenais/opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml" ) ;
 
    // Change path before execution 
    cascade.load("/home/tmenais/opencv/data/haarcascades/haarcascade_frontalface_default.xml") ; 

    std::unique_ptr<tensorflow::Session> session = initSession("/home/tmenais/Face_Recognition/lib/src/ckpt/20180408-102900/20180408-102900.pb");

 
    //Start Video..1) 0 for WebCam 2) "Path to Video" for a Local Video
    capture.open(0); 
    if( capture.isOpened() )
    {
        // Capture frames from video and detect faces
        cout << "Face Detection Started...." << endl;
        while(1)
        {
            capture >> frame;
            if( frame.empty() )
                break;
            Mat frame1 = frame.clone();
            detectAndDraw(frame1, cascade,
                          // nestedCascade,
                           scale,
                           &session,
                           label_database,
                           embeddings_database); //detectAndDraw( frame1, cascade, nestedCascade, scale );
            char c = (char)waitKey(10);
         
            // Press q to exit from window
            if( c == 27 || c == 'q' || c == 'Q' ) 
                break;
        }
    }
    else
        cout<<"Could not Open Camera";
    return 0;
}
 
