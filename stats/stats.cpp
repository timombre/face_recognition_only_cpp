#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <set>
#include <cstring>

using namespace std;

struct dataSet
{
	std::vector<string> labels;
	std::vector<std::vector<float>> embeddings;
	std::set<std::string> unique_labels;

};

struct dataHandler
{
	int right_predictions;
	int wrong_predictions;
	int outofrange_predictions;
	int right_final_predictions;
	int wrong_final_predictions;
	int outofrange_final_predictions;
	
};

bool isFloat(string s) {
    istringstream iss(s);
    float dummy;
    iss >> noskipws >> dummy;
    return iss && iss.eof();     // Result converted to bool
}




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

dataSet CreateDataSet(std::string file)
{
	dataSet data;

	std::vector<std::string> database = ReadLabelsAndFile(file);
	std::vector<std::string> label_database;
	std::vector<std::vector<float>> embeddings_float;

	label_database.reserve(database.size());
	embeddings_float.reserve(database.size());

	for (int i = 0; i < database.size(); ++i){
        label_database.push_back(database[i].substr(0, database[i].find_first_of(" ")));
        embeddings_float.push_back(ConvStringToFloats(database[i].substr(database[i].find_first_of(" ")+1)));
    }

    std::set<std::string> unique_from_label_database( label_database.begin(), label_database.end() );

    data.labels = label_database;
    data.embeddings = embeddings_float;
    data.unique_labels = unique_from_label_database;

    return data;

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

dataHandler CreateDataHandler(std::string label, dataSet ref, dataSet comp,  float thresh)
{

	dataHandler labelstats;
	float sdist;

	labelstats.right_predictions = 0 ;
	labelstats.wrong_predictions = 0 ;
	labelstats.outofrange_predictions = 0 ;
	labelstats.right_final_predictions = 0 ;
	labelstats.wrong_final_predictions = 0 ;
	labelstats.outofrange_final_predictions = 0 ;

	size_t i =0;
	for (const auto& lab : ref.labels)
	{
		if (label.compare(lab))
		{
			size_t j =0;
			int posofmin =0;
			float min_diff =SquaredDistance(ref.embeddings[i],comp.embeddings[0]);
			for (const auto& lab2 : comp.labels)
			{
				sdist = SquaredDistance(ref.embeddings[i],comp.embeddings[j]);

				//std::cout << i << " " << j << std::endl;
				//std::cout << sdist << std::endl;

				if (sdist > thresh)
				{
					labelstats.outofrange_predictions++ ;
				} else {
					if (comp.labels[j].compare(lab))
					{
						labelstats.right_predictions++;
					} else {
						labelstats.wrong_predictions++;
					}
				}

				if (sdist < min_diff)
				{
					posofmin = j;
					min_diff = sdist;
				}

				j++;
			}
			if (min_diff > thresh)
			{
				labelstats.outofrange_final_predictions++ ;
			} else {
				if (comp.labels[posofmin].compare(lab))
				{
					labelstats.right_final_predictions++ ;
				} else {
					labelstats.wrong_final_predictions++ ;
				}
			}
		}
		i++;
	}

	std::cout <<  label << std::endl;

	std::cout << "tot right: " <<  labelstats.right_predictions << " tot wrong: " <<  labelstats.wrong_predictions << " tot outofrange: " << labelstats.outofrange_predictions << std::endl;
	std::cout << "tot right_final: " <<  labelstats.right_final_predictions << " tot wrong_final: " <<  labelstats.wrong_final_predictions << " tot outofrange_final: " << labelstats.outofrange_final_predictions << std::endl;

	return labelstats ;

}



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
    	//std::cout << substring << std::endl;
    }


    for (const auto& substring : test_database.unique_labels)
    {
    	//std::cout << substring << std::endl;
    }


    //std::cout << uniquefromlabel_database << std::endl;
    //std::cout << uniquefromtest_label_database << std::endl;

}

