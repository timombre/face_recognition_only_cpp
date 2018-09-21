#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <set>

using namespace std;


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



int main(int argc, char *argv[])
{

	if( argc < 3)
    {
        std::cout <<" Usage: give a label_embeddings database file and a test database file" << std::endl;
        return -1;
    }
    
    std::vector<std::string> database = ReadLabelsAndFile(argv[1]);
    std::vector<std::string> test_database = ReadLabelsAndFile(argv[2]);

    std::vector<std::string> label_database, test_label_database;
    label_database.reserve(database.size());
    test_label_database.reserve(database.size());

    std::vector<std::vector<float>> embeddings_float, test_embeddings_float;
    embeddings_float.reserve(database.size()); 
    test_embeddings_float.reserve(database.size());

    for (int i = 0; i < database.size(); ++i){
        label_database.push_back(database[i].substr(0, database[i].find_first_of(" ")));
        embeddings_float.push_back(ConvStringToFloats(database[i].substr(database[i].find_first_of(" ")+1)));
    }

    for (int i = 0; i < test_database.size(); ++i){
    	test_label_database.push_back(test_database[i].substr(0, test_database[i].find_first_of(" ")));
        test_embeddings_float.push_back(ConvStringToFloats(test_database[i].substr(test_database[i].find_first_of(" ")+1)));
    }

    std::set<std::string> uniquefrom_label_database( label_database.begin(), label_database.end() );
    std::set<std::string> uniquefrom_test_label_database( test_label_database.begin(), test_label_database.end() );

    for (auto substring : uniquefrom_label_database)
    {
    	for (int j = 0; j < database.size(); ++j)
    	{
    		for (int i = 0; i < test_database.size(); ++i)
    		{
    			if (substring == label_database[i])
    			{
    				/* code */
    			}
    		}
    	}
    	std::cout << substring << std::endl;
    }

    std::cout << uniquefrom_label_database.size() << std::endl;

    for (auto substring : uniquefrom_test_label_database)
    {
    	std::cout << substring << std::endl;
    }


    //std::cout << uniquefromlabel_database << std::endl;
    //std::cout << uniquefromtest_label_database << std::endl;

}

