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
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <experimental/filesystem>
#include <random>

using namespace std;
namespace filesys = std::experimental::filesystem;




/*
 * Get the list of all files in given directory and its sub directories.
 *
 * Arguments
 * 	dirPath : Path of directory to be traversed
 * 	dirSkipList : List of folder names to be skipped
 *
 * Returns:
 * 	vector containing paths of all the files in given directory and its sub directories
 *
 */
std::vector<std::string> getAllFilesInDir(const std::string &dirPath, bool dir)
{

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

bool isFloat(string s) {
    istringstream iss(s);
    float dummy;
    iss >> noskipws >> dummy;
    return iss && iss.eof();     // Result converted to bool
}

int main(int argc, char *argv[])
{

	if( argc < 1)
	{
	    cout <<" Usage: give a folder " << endl;
	    return -1;
	}

	std::string dirPath = argv[1];
	bool splitdb = false;
	float percent_frac = 100.0;
	bool keepratio = false;

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
	}

	// Get recursive list of files in given directory and its sub directories

	if (splitdb == false)
	{

		std::vector<std::string> listOfFiles = getAllFilesInDir(dirPath, false);

		ofstream myfile;
	    myfile.open ("labels_and_files_of_database.txt");

		for (const auto& str : listOfFiles){
			myfile <<  str << std::endl;
		}
		myfile.close();
		
	} else {
		std::vector<std::string> listOfDirs = getAllFilesInDir(dirPath, true);
		std::vector< std::vector<std::string> > filesInDataset;
		
		std::random_device                  rand_dev;
		std::mt19937                        generator(rand_dev());

		ofstream mygenset;
		ofstream mytestset;
	    
	    mytestset.open ("labels_and_files_of_database_testset.txt");
	    mygenset.open ("labels_and_files_of_database.txt");

		if (keepratio == false)
		{
			std::vector<std::string> listOfFiles = getAllFilesInDir(dirPath, false);
			std::vector<unsigned int> indices(listOfFiles.size());
		    std::iota(indices.begin(), indices.end(), 0);
		    std::shuffle(indices.begin(), indices.end(), generator);

		    for (int i = 0; i < int(listOfFiles.size() * percent_frac / 100.0); ++i)
		    {
		    	mygenset <<  listOfFiles[indices[i]] << std::endl;
		    }

		    for (int i = int(listOfFiles.size() * percent_frac / 100.0); i < listOfFiles.size(); ++i)
		    {
		    	mytestset <<  listOfFiles[indices[i]] << std::endl;
		    }


		} else {


		    for (const auto& folder : listOfDirs)
		    {
		    	std::vector<std::string> listOfFiles = getAllFilesInDir(folder, false);


		    	std::vector<unsigned int> indices(listOfFiles.size());
		    	std::iota(indices.begin(), indices.end(), 0);
		    	std::shuffle(indices.begin(), indices.end(), generator);

		    	for (int i = 0; i < int(listOfFiles.size() * percent_frac / 100.0); ++i)
		    	{
		    		mygenset <<  listOfFiles[indices[i]] << std::endl;
		    	}

		    	for (int i = int(listOfFiles.size() * percent_frac / 100.0); i < listOfFiles.size(); ++i)
		    	{
		    		mytestset <<  listOfFiles[indices[i]] << std::endl;
		    	}
		    }
		}		

		mygenset.close();
		mytestset.close();
	}
	

}