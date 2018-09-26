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

int main( int argc, char** argv ){

	if( argc < 1)
	{
	    cout <<" Usage: give a folder " << endl;
	    return -1;
	}

	std::string dirPath = argv[1];
	bool splitdb = false;
	float percent_frac = 100.0;
	bool keepratio = false;
    bool gen_aligned_db = false;
    bool show_crop = false;
    bool process_test = false;
    bool filesnames= false;

    std::vector<std::string> listOfFilesDataset;
	std::vector<std::string> listOfFilesDatasetTest;

	std::string pbfile = "20170512-110547.pb";
	std::string cascfile = "haarcascade_frontalface_alt2.xml";
	std::string lmfile = "lbfmodel.yaml";


	for (int i = 0; i < argc; ++i)
	{
	    if (strcmp( argv[i], "-splitdb") == 0 )
	    {
	        if (i != argc-1 && isFloat(argv[i+1]))
	        {
	        	splitdb = true ;
	            percent_frac = std::min(atof(argv[i+1]),100.0);
	            if (atof(argv[i+1])>100.0)
	            {
	            	splitdb = false ;
	            	cout <<" More than 100% , database not split " << endl;
	            }
	        } else {
	        	cout <<" No percentage given, database not split " << endl;
	        }
	        
	    }
	    if (strcmp( argv[i], "-kr") == 0 )
	    {
	    	keepratio = true ;
	    }

	    if (strcmp( argv[i], "-filesnames") == 0 )
	    {
	    	filesnames = true ;
	    }

	    if (strcmp( argv[i], "-gen_aligned_db") == 0)
	    {
	       gen_aligned_db = true ;
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

	std::cout << pbfile << std::endl;



	
	
	std::vector<std::string> listOfDirs = getAllFilesInDir(dirPath, true);

	if (gen_aligned_db)
	{
	    std::string root_folder = dirPath + "/../aligned_data_base";
	    filesys::create_directory(root_folder);

	    //stringstream call_line;

	    //call_line << "if [ ! -d " << argv[1] << root_folder << " ]; then mkdir " << argv[1] << root_folder << " ; fi ;";
	    //call_line << "if [ ! -d " << argv[1] << folder << " ]; then echo plop ; fi ;";

	    //system(call_line.str().c_str());


	  
	    for (const auto& folder : listOfDirs)
	    {
	        //stringstream second_call_line;
	        //second_call_line << "if [ ! -d " << argv[1] << root_folder << "/" << folder.substr(folder.find_last_of("/") +1) << " ]; then mkdir " << argv[1] << root_folder << "/" << folder.substr(folder.find_last_of("/") +1) << " ; fi ;";
	        //system(second_call_line.str().c_str());
	        filesys::create_directory(root_folder + "/" + folder.substr(folder.find_last_of("/") +1) );
	    }

	}

	// Load everything needed
    CascadeClassifier cascade;
    cascade.load(cascfile);

    Ptr<Facemark> facemark = FacemarkLBF::create();
    facemark->loadModel(lmfile);

    std::unique_ptr<tensorflow::Session> session = initSession(pbfile);


	std::vector< std::vector<std::string> > filesInDataset;
	std::vector< std::vector<std::string> > filesInDatasetTest;
	
	std::random_device                  rand_dev;
	std::mt19937                        generator(rand_dev()); 

	ofstream myembeddings, mytestembeddings;

	myembeddings.open("face_embeddings_database.txt");
	mytestembeddings.open("face_embeddings_database_testset.txt");

    for (const auto& folder : listOfDirs)
    {
    	std::vector<std::string> listOfFiles = getAllFilesInDir(folder, false);

    	std::string label = folder.substr(folder.find_last_of("/") +1);


    	std::vector<unsigned int> indices(listOfFiles.size());
    	std::iota(indices.begin(), indices.end(), 0);
    	std::shuffle(indices.begin(), indices.end(), generator);

    	if (splitdb == false)
    	{
    		for (int i = 0; i < listOfFiles.size() ; ++i)
    		    	{
    		    		listOfFilesDataset.push_back(listOfFiles[indices[i]]);
    		    		myembeddings <<  genEmbeddings(cascade, facemark, *session, listOfFiles[indices[i]], label, gen_aligned_db, argv[1]) << "\n";
    		    	}
    		
    	} else {

    		

    		for (int i = 0; i < int(listOfFiles.size() * percent_frac / 100.0); ++i)
    		{
    			listOfFilesDataset.push_back(listOfFiles[indices[i]]);
    			myembeddings <<  genEmbeddings(cascade, facemark, *session, listOfFiles[indices[i]], label, gen_aligned_db, argv[1]) << "\n";
    		}
    		
    		for (int i = int(listOfFiles.size() * percent_frac / 100.0); i < listOfFiles.size(); ++i)
    		{
    			listOfFilesDatasetTest.push_back(listOfFiles[indices[i]]);
    			mytestembeddings <<  genEmbeddings(cascade, facemark, *session, listOfFiles[indices[i]], label, gen_aligned_db, argv[1]) << "\n";
    		}
    		
    	}    	
    }

    myembeddings.close();
    mytestembeddings.close();			


	if (filesnames == true)
	{
		ofstream mygenset;

		mygenset.open ("labels_and_files_of_database.txt");
		for (const auto& file : listOfFilesDataset)
		{
			mygenset <<  file << std::endl;			
		}
		mygenset.close();

		if (splitdb == true)
		{			
			mygenset.open ("labels_and_files_of_database_testset.txt");
			for (const auto& file : listOfFilesDatasetTest)
			{
				mygenset <<  file << std::endl;
			}
			mygenset.close();
		}

	}

	return 0;


}