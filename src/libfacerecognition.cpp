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
#include "libfacerecognition.hpp"

std::vector<float> ConvStringToFloats(std::string str){

    std::vector<float> vect;
    vect.reserve(512); // embedding is a 512 components vector
    std::istringstream stm(str) ;
    
    float number;

    while(stm >> number){
        vect.push_back(number);              
    }

    return vect;
}

dataSet CreateDataSet(std::string file){
    dataSet data;

    std::vector<std::string> database = ReadLabelsAndEmbeddings(file);
    std::vector<std::string> label_database;
    std::vector<std::vector<float>> embeddings_float;

    label_database.reserve(database.size());
    embeddings_float.reserve(database.size());

    for (int i = 0; i < database.size(); ++i){
        if ( database[i].length() >0 )
        {
            label_database.push_back(database[i].substr(0, database[i].find_first_of(" ")));
            embeddings_float.push_back(ConvStringToFloats(database[i].substr(database[i].find_first_of(" ")+1)));
        }
        
    }

    std::set<std::string> unique_from_label_database( label_database.begin(), label_database.end() );

    data.labels = label_database;
    data.embeddings = embeddings_float;
    data.unique_labels = unique_from_label_database;

    return data;

}


float SquaredDistance(std::vector<float> vect1, std::vector<float> vect2){

    float diff = 0.0;

    if (vect1.size() == vect2.size())
    {
        for (int i = 0; i < vect1.size(); ++i){
            diff += (vect1[i] - vect2[i]) * (vect1[i] - vect2[i]) ;
        }

    } else {
        std::cout <<" WARNING: vectors have different sizes" << std::endl;

        for (int i = 0; i < std::min(vect1.size(),vect2.size()); ++i){
            diff += (vect1[i] - vect2[i]) * (vect1[i] - vect2[i]) ;
        }
    }

    return diff;
}

dataHandler CreateDataHandler(std::string label, dataSet ref, dataSet comp,  float thresh){

    dataHandler labelstats;
    float sdist;

    labelstats.right_predictions = 0 ;
    labelstats.wrong_predictions = 0 ;
    labelstats.outofrange_predictions = 0 ;
    labelstats.right_final_predictions = 0 ;
    labelstats.wrong_final_predictions = 0 ;
    labelstats.outofrange_final_predictions = 0 ;

    size_t i =0;
    for (const auto& lab : ref.labels)
    {
        if (label.compare(lab))
        {
            size_t j =0;
            int posofmin =0;
            float min_diff =SquaredDistance(ref.embeddings[i],comp.embeddings[0]);
            for (const auto& lab2 : comp.labels)
            {
                sdist = SquaredDistance(ref.embeddings[i],comp.embeddings[j]);

                //std::cout << i << " " << j << std::endl;
                //std::cout << sdist << std::endl;

                if (sdist > thresh)
                {
                    labelstats.outofrange_predictions++ ;
                } else {
                    if (comp.labels[j].compare(lab))
                    {
                        labelstats.right_predictions++;
                    } else {
                        labelstats.wrong_predictions++;
                    }
                }

                if (sdist < min_diff)
                {
                    posofmin = j;
                    min_diff = sdist;
                }

                j++;
            }
            if (min_diff > thresh)
            {
                labelstats.outofrange_final_predictions++ ;
            } else {
                if (comp.labels[posofmin].compare(lab))
                {
                    labelstats.right_final_predictions++ ;
                } else {
                    labelstats.wrong_final_predictions++ ;
                }
            }
        }
        i++;
    }

    std::cout <<  label << std::endl;

    std::cout << "tot right: " <<  labelstats.right_predictions << " tot wrong: " <<  labelstats.wrong_predictions << " tot outofrange: " << labelstats.outofrange_predictions << std::endl;
    std::cout << "tot right_final: " <<  labelstats.right_final_predictions << " tot wrong_final: " <<  labelstats.wrong_final_predictions << " tot outofrange_final: " << labelstats.outofrange_final_predictions << std::endl;

    return labelstats ;

}


std::vector<std::string> getAllFilesInDir(const std::string &dirPath, bool dir){

    // Create a vector of string
    std::vector<std::string> list;
    try {
        // Check if given path exists and points to a directory
        if (filesys::exists(dirPath) && filesys::is_directory(dirPath))
        {
            // Create a Recursive Directory Iterator object and points to the starting of directory
            filesys::recursive_directory_iterator iter(dirPath);

            // Create a Recursive Directory Iterator object pointing to end.
            filesys::recursive_directory_iterator end;

            // Iterate till end
            while (iter != end)
            {       
                // Add the name in vector
                if (dir == true)
                {
                    if (filesys::is_directory(iter->path().string())){
                        list.push_back(iter->path().string());
                    }
                } else {
                    if (!filesys::is_directory(iter->path().string())){
                        list.push_back(iter->path().string());
                    }
                }
                

                error_code ec;
                // Increment the iterator to point to next entry in recursive iteration
                iter.increment(ec);
                if (ec) {
                    std::cerr << "Error While Accessing : " << iter->path().string() << " :: " << ec.message() << '\n';
                }
            }
        }
    }
    catch (std::system_error & e)
    {
        std::cerr << "Exception :: " << e.what();
    }
    return list;
}

bool isFloat(string s){
    istringstream iss(s);
    float dummy;
    iss >> noskipws >> dummy;
    return iss && iss.eof();     // Result converted to bool
}




static tensorflow::Status ReadEntireFile(tensorflow::Env* env, const std::string& filename, Tensor* output){
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


tensorflow::Status LoadGraph(const std::string& graph_file_name, std::unique_ptr<tensorflow::Session>* session) {
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

std::vector<std::string> ReadLabelsAndEmbeddings(const std::string file_name){
    
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


cv::Mat faceCenterRotateCrop(Mat &im, vector<Point2f> landmarks, Rect face, int i, bool show_crop){

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
    image_points.push_back(landmarks[8]);     // Chin
    image_points.push_back(landmarks[45]);    // Left eye left corner
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
    model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));       // Right mouth corner

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
    
    // Rotate around the center    
    Point2f pt = landmarks[30]; //center is nose tip
    Mat r = getRotationMatrix2D(pt, theta_deg, 1.0);
    // determine bounding rectangle
    cv::Rect bbox = cv::RotatedRect(pt,im.size(), theta_deg).boundingRect();

    // Apply affine transform
    Mat dst;
    warpAffine(im, dst, r, bbox.size());

    // Now crop the face
    Mat Cropped_Face = dst(face);

    resize( Cropped_Face, Cropped_Face, cv::Size(160, 160), CV_INTER_LINEAR);

    

    if (show_crop == true)
    {            
        std::string text = "Cropped Face ";
        text += std::to_string(i);
        imshow(text,Cropped_Face);
    }

    return Cropped_Face ;
}

 
// Function for Face Detection
void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    Ptr<Facemark> facemark,
                    double scale, std::unique_ptr<tensorflow::Session>* session,
                    std::vector<std::string> label_database,
                    std::vector<std::string> embeddings_database,
                    bool show_crop, float thresh){

    vector<Rect> faces;
    Mat smallImg;
    double fx = 1 / scale; 
    resize( img, smallImg, Size(), fx, fx, CV_INTER_LINEAR ); 
   
    cascade.detectMultiScale( smallImg, faces, 1.1, 
                            7, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    
    int linewidth = std::max(1, int(img.rows * .005));

    vector<vector<Point2f>> landmarks;

    bool success = facemark->fit(smallImg,faces,landmarks);
 
    // Draw boxes around the faces
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i];
        
        Scalar color = Scalar(255, 0, 0); // Color for Drawing tool

        rectangle( img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
                cvPoint(cvRound((r.x + r.width-1)*scale), 
                cvRound((r.y + r.height-1)*scale)), color, linewidth, 8, 0);

        Mat smallImgROI ;

        if(success)
        {
        // If success, align face
            if (landmarks[i].size()==68)
            {
                 smallImgROI = faceCenterRotateCrop(smallImg,landmarks[i],faces[i],i,show_crop);
            }

        }

        if(! smallImgROI.data )                // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
            return ;
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

        //Tensor Flow graph specifics :
        string input_layer = "input";
        string phase_train_layer = "phase_train";
        string output_layer = "embeddings";
        tensorflow::Tensor phase_tensor(tensorflow::DT_BOOL, tensorflow::TensorShape());
        phase_tensor.scalar<bool>()() = false;

        //Run session
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

            std::vector<float> vect = ConvStringToFloats(embeddings_database[i]);


            for (int j = 0; j < outputs[0].shape().dim_size(1); ++j){
                diff += (output_c(0,j) - vect[j]) * (output_c(0,j) - vect[j]) ;
            }
            //diff= diff; //no need to sqrt
            if (diff < min_emb_diff)
            {
                min_emb_diff = diff ;
                posofmin = i ;
            }                
        }

        cv::Point txt_up = cvPoint(cvRound(r.x*scale + linewidth ), cvRound(r.y*scale - 4 * linewidth));      
        cv::Point txt_in = cvPoint(cvRound(r.x*scale + linewidth ), cvRound(r.y*scale + 12 * linewidth));

        if(min_emb_diff < thresh) {
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

std::string genEmbeddings(CascadeClassifier cascade, Ptr<Facemark> facemark, tensorflow::Session& session, std::string filename,
                   std::string label,  bool gen_dt, std::string data_root ){

    std::string embedding;

    std::cout << "processing :" << filename << '\n'; 
    
    Mat image = imread(filename, CV_LOAD_IMAGE_COLOR);   
    if(! image.data )                // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return embedding;
    }

    vector<Rect> faces;
    double scale=1; //modify if needed
    
    double fx = 1 / scale;
    Mat smallImg;
    
    resize( image, smallImg, Size(), fx, fx, CV_INTER_LINEAR );

    
    cascade.detectMultiScale( smallImg, faces, 1.1, 
                            7, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );


    if (faces.size() !=1)
    {
        cout <<  "There should be exactly 1 face" << std::endl ;
        return embedding;
       
    }else{               

        if(! smallImg.data )                // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
            return embedding;
        }

        Rect r = faces[0];

        Mat detectlandmarks;

        vector<vector<Point2f>> landmarks;
        
        // Run landmark detector
        bool success = facemark->fit(smallImg,faces,landmarks);

        Mat smallImgROI ;

        if(success)
        {
        // If success, align face
            for(int i = 0; i < landmarks.size(); i++)
            {
                if (landmarks[i].size()==68)
                {
                    smallImgROI = faceCenterRotateCrop(smallImg,landmarks[i],faces[i],i,false);
                }

            }
        }

        if(! smallImgROI.data )                // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
            return embedding;
        }

        if (gen_dt)
        {
            std::string aligned_data_base = data_root;
            aligned_data_base.append("/../aligned_data_base/"); aligned_data_base.append(label); aligned_data_base.append("/"); aligned_data_base.append(filename.substr(filename.find_last_of("/") +1));
            imwrite(aligned_data_base, smallImgROI);
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

        //Tensor Flow graph specifics :
        string input_layer = "input";
        string phase_train_layer = "phase_train";
        string output_layer = "embeddings";
        tensorflow::Tensor phase_tensor(tensorflow::DT_BOOL, tensorflow::TensorShape());
        phase_tensor.scalar<bool>()() = false;

        //Run session
        
        std::vector<Tensor> outputs ;
        Status run_status = session.Run({{input_layer, input_tensor},
                                       {phase_train_layer, phase_tensor}}, 
                                       {output_layer}, {}, &outputs);
        
        auto output_c = outputs[0].tensor<float, 2>();

        embedding.append(label);

        
        
        for (int i = 0; i <  outputs[0].shape().dim_size(1); ++i)
        {
            embedding.append(" ");
            std::ostringstream ss;
            ss << output_c(0,i);
            embedding.append(ss.str());
        }
        
       return embedding;
    }
}
