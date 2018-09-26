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

int main(int argc, char *argv[])
{

	if( argc < 3)
    {
        std::cout <<" Usage: give a label_embeddings database file and a test database file" << std::endl;
        return -1;
    }

    float thresh = 0.045;


    for (int i = 0; i < argc; ++i)
    {
        if (strcmp( argv[i], "-thresh") == 0 )
        {
            if (i != argc-1 && isFloat(argv[i+1]))
            {
                thresh = atof(argv[i+1]);
                
            } else {
            	cout <<" No threshold given, keeps " << thresh << " default " << endl;
            }
            
        }
        
    }

    dataSet ref_database = CreateDataSet(argv[1]);
    dataSet test_database = CreateDataSet(argv[2]);
    
    std::vector<dataHandler> fullstat;
    fullstat.reserve(ref_database.unique_labels.size());

    for (const auto& substring : ref_database.unique_labels)
    {
    	fullstat.push_back(CreateDataHandler(substring, ref_database, test_database, thresh));
    }
    std::cout << std::endl;

}

