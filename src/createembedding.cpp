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


int main( int argc, char** argv )
{
    if( argc < 3)
    {
     cout <<" Usage: give a folder and label file" << endl;
     return -1;
    }
    std::vector<std::string> database = ReadLabelsAndEmbeddings(argv[2]);
    
    std::string testset;
    float percent_frac;    


    bool gen_dt = false;
    bool show_crop = false;
    bool splitdb = false;
    bool process_test = false;

    for (int i = 0; i < argc; ++i)
    {
        if (strcmp( argv[i], "-gen_aligned_db") == 0)
        {
           gen_dt = true ;
        }
        if (strcmp( argv[i], "-splitdb") == 0 )
        {
            if (i != argc-1 && isFloat(argv[i+1]))
            {
                splitdb = true ;
                percent_frac = atof(argv[i+1]);
            } else {
                cout <<" No percentage given, database not split " << endl;
            }
        }
        if (strcmp( argv[i], "-testset") == 0 && i != argc-1 && argv[i+1])
        {
           testset = argv[i+1] ;
           process_test = true;
        }
    }


    if (gen_dt)
    {
        string folder = "/../aligned_data_base";

        stringstream call_line;

        call_line << "if [ ! -d " << argv[1] << folder << " ]; then mkdir " << argv[1] << folder << " ; fi ;";
        //call_line << "if [ ! -d " << argv[1] << folder << " ]; then echo plop ; fi ;";

        system(call_line.str().c_str());


      
        for (int i = 0; i < database.size(); ++i)
        {
            stringstream second_call_line;
            second_call_line << "if [ ! -d " << argv[1] << folder << "/" << database[i].substr(0, database[i].find_first_of(" ")) << " ]; then mkdir " << argv[1] << folder << "/" << database[i].substr(0, database[i].find_first_of(" ")) << " ; fi ;";
            system(second_call_line.str().c_str());
        }

    }   
   

    // Load everything needed
    CascadeClassifier cascade;
    cascade.load("haarcascade_frontalface_alt2.xml");

    Ptr<Facemark> facemark = FacemarkLBF::create();
    facemark->loadModel("lbfmodel.yaml");

    std::unique_ptr<tensorflow::Session> session = initSession("20170512-110547.pb");

    genEmbeddings(cascade, facemark, *session, "face_embeddings_database.txt", database, argv[1], show_crop, gen_dt);

    if (splitdb && process_test)
    {
        std::vector<std::string> test_database = ReadLabelsAndEmbeddings(testset);
        genEmbeddings(cascade, facemark, *session, "face_embeddings_test_database.txt", test_database, argv[1], show_crop, gen_dt);
    }  

    

    return 0;
}