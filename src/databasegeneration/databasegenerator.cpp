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
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

#include <experimental/filesystem>
#include <random>
#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;

namespace filesys = std::experimental::filesystem;

void genDatabase( Mat& im, float period, std::clock_t &timestamp, std::string filename, int &i){

    imshow( "Database generation", im );

    if (std::clock() - timestamp > period * 1000000) //clock is in microseconds
        {
            timestamp = std::clock() ;
            imshow( "Snapshot", im );
            imwrite(filename, im);
            i++ ;
        }    

}

bool isFloat(std::string s){
    std::istringstream iss(s);
    float dummy;
    iss >> std::noskipws >> dummy;
    return iss && iss.eof();     // Result converted to bool
}

int main( int argc, const char** argv )
{

    if( argc < 3)
    {
        std::cout <<" Provide a database folder and a label" << std::endl;
        return -1;
    }

    bool show_crop = false;
    float freq = 1;
    std::string dirPath = argv[1];
    std::string label = argv[2];

    filesys::create_directory(dirPath + "/" + label );


    if (argc > 3)
    {
        
        for (int i = 0; i < argc; ++i)
        {
           
           if (strcmp( argv[i], "-freq") == 0 )
           {
               if (i != argc-1 && isFloat(argv[i+1]))
               {
                   freq = atof(argv[i+1]);
                   
               } else {
                std::cout <<" Invalid frequency given, keeping " << freq << " as default " << std::endl;
               }
           }

        }
    }

    VideoCapture capture;
    Mat frame;

    std::clock_t timestamp ;
    timestamp = std::clock();

    double period = double(1/freq);
    int i = 0 ; 
 
    //Start Video..1) 0 for WebCam 2) "Path to Video" for a Local Video
    capture.open(0); 
    if( capture.isOpened() )
    {
        // Capture frames from video and detect faces
        std::cout << "Start database generation...." << std::endl;
        while(1)
        {
            capture >> frame;
            if( frame.empty() ){
                break;
            }

            std::ostringstream oss;
            oss << dirPath << "/" << label << "/" << label << "_" << i << ".jpg";
            std::string filename = oss.str();

            Mat frame1 = frame.clone();

            genDatabase(frame1, period , timestamp, filename, i);
            char c = (char)waitKey(10);
         
            // Press q to exit from window
            if( c == 27 || c == 'q' || c == 'Q' ){
                break;
            }
        }
    }
    else
        std::cout<<"Could not Open Camera";
    return 0;
}