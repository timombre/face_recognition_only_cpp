#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <cstring>
#include <experimental/filesystem>
#include <random>

using namespace std;
namespace filesys = std::experimental::filesystem;


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
    std::istringstream stm(str) ;
    
    float number;

    while(stm >> number){
       
                vect.push_back(number);
                            
    }

    return vect;
}

int main(int argc, char *argv[])
{

	if( argc != 3)
    {
     cout <<" Usage: give a label_embeddings database file and a test database file" << endl;
     return -1;
    }
    
    std::vector<std::string> database = ReadLabelsAndFile(argv[1]);
    std::vector<std::string> test_database = ReadLabelsAndFile(argv[2]);

    std::vector<std::vector<float>> embeddings_float;
    embeddings_float.reserve(database.size() * 512);

    std::vector<std::vector<float>> test_embeddings_float;
    test_embeddings_float.reserve(database.size() * 512);

	

}

