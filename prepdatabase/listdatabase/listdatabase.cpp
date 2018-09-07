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
#include <vector>
#include <algorithm>
#include <string>
#include <experimental/filesystem>

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
std::vector<std::string> getAllFilesInDir(const std::string &dirPath)
{

	// Create a vector of string
	std::vector<std::string> listOfFiles;
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
				listOfFiles.push_back(iter->path().string());
				

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
	return listOfFiles;
}

int main(int argc, char *argv[])
{
	std::string dirPath = argv[1];

	// Get recursive list of files in given directory and its sub directories
	std::vector<std::string> listOfFiles = getAllFilesInDir(dirPath);

	// Iterate over the vector and print all files
	for (auto str : listOfFiles)
		std::cout << str << std::endl;

}