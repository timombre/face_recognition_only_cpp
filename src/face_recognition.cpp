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



int main( int argc, const char** argv )
{

    if( argc < 2)
    {
        cout <<" Provide your embeddings database" << endl;
        return -1;
    }

    bool show_crop = false;
    bool volume_points = true;
    float thresh = 0.8;
    float relative_probability = 1.4;
    bool verbose =false;

    std::string pbfile = "20170512-110547.pb";
    std::string cascfile = "haarcascade_frontalface_alt2.xml";
    std::string lmfile = "lbfmodel.yaml";

    if (argc > 2)
    {
        
        for (int i = 0; i < argc; ++i)
        {
           
            if (strcmp( argv[i], "-show_crop") == 0)
            {
               show_crop = true ;
            }

            if (strcmp( argv[i], "-verbose") == 0)
            {
               verbose = true ;
            }

            if (strcmp( argv[i], "-mindist") == 0)
            {
               volume_points = false ;
            }

            if (strcmp( argv[i], "-thresh") == 0 )
            {
                if (i != argc-1 && isFloat(argv[i+1]))
                {
                    thresh = atof(argv[i+1]);
                   
                } else {
                    cout <<" No threshold given, keeping " << thresh << " as default " << endl;
                }
            }

            if (strcmp( argv[i], "-rp") == 0 )
            {
                if (i != argc-1 && isFloat(argv[i+1]))
                {
                    relative_probability = atof(argv[i+1]);
                   
                } else {
                    cout <<" No relative probability given, keeping " << relative_probability << " as default " << endl;
                }
            }

            if (strcmp( argv[i], "-choose_pb") == 0 && i != argc-1 && argv[i+1])
            {
                pbfile = argv[i+1];
            }
            if (strcmp( argv[i], "-choose_casc") == 0 && i != argc-1 && argv[i+1])
            {
                cascfile = argv[i+1];
            }
            if (strcmp( argv[i], "-choose_lm") == 0 && i != argc-1 && argv[i+1])
            {
                lmfile = argv[i+1];           
            }

        }
    }

    dataSet database = CreateDataSet(argv[1]);

    // VideoCapture class for playing video for which faces to be detected
    VideoCapture capture; 
    Mat frame;
  
    // Load everything needed    
    Ptr<Facemark> facemark = FacemarkLBF::create();
    facemark->loadModel(lmfile);

    CascadeClassifier cascade;
    double scale=1; 
    cascade.load(cascfile) ;

    std::unique_ptr<tensorflow::Session> session = initSession(pbfile);

 
    //Start Video..1) 0 for WebCam 2) "Path to Video" for a Local Video
    capture.open(0); 
    if( capture.isOpened() )
    {
        // Capture frames from video and detect faces
        cout << "Face Detection Started...." << endl;
        while(1)
        {
            capture >> frame;
            if( frame.empty() ){
                break;
            }
            Mat frame1 = frame.clone();
            detectAndDraw(frame1, cascade,
                           facemark,
                           scale,
                           &session,
                           database,
                           show_crop,
                           thresh,
                           volume_points,
                           relative_probability,
                           verbose);
            char c = (char)waitKey(10);
         
            // Press q to exit from window
            if( c == 27 || c == 'q' || c == 'Q' ){
                break;
            }
        }
    }
    else
        cout<<"Could not Open Camera";
    return 0;
}
 
