#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

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

    std::vector<std::string> label_database, test_label_database;
    label_database.reserve(database.size());
    test_label_database.reserve(database.size());

    std::vector<std::vector<float>> embeddings_float, test_embeddings_float;
    embeddings_float.reserve(database.size() * 512); // embeddings is a vector with 512 components
    test_embeddings_float.reserve(database.size() * 512);

    for (int i = 0; i < database.size(); ++i){
        label_database.push_back(database[i].substr(0, database[i].find_first_of(" ")));
        embeddings_float.push_back(ConvStringToFloats(database[i].substr(database[i].find_first_of(" ")+1)));
    }

    for (int i = 0; i < test_database.size(); ++i){
    	test_label_database.push_back(test_database[i].substr(0, test_database[i].find_first_of(" ")));
        test_embeddings_float.push_back(ConvStringToFloats(test_database[i].substr(test_database[i].find_first_of(" ")+1)));
    }

}

