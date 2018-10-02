
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
#include "libstats.hpp"

dataSet CreateDataSet(std::string file){
    dataSet data;

    std::vector<std::string> database = ReadLabelsAndEmbeddings(file);
    std::vector<std::string> label_database;
    std::vector<std::vector<float>> embeddings_float;

    label_database.reserve(database.size());
    embeddings_float.reserve(database.size());

    for (int i = 0; i < database.size(); ++i){
        if ( database[i].length() >0 )
        {
            label_database.push_back(database[i].substr(0, database[i].find_first_of(" ")));
            embeddings_float.push_back(ConvStringToFloats(database[i].substr(database[i].find_first_of(" ")+1)));
        }
        
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

dataHandler CreateDataHandler(std::string label, dataSet ref, dataSet comp,  float thresh){

    dataHandler labelstats;
    float sdist;

    labelstats.right_predictions = 0 ;
    labelstats.wrong_predictions = 0 ;
    labelstats.outofrange_predictions = 0 ;
    labelstats.right_final_predictions = 0 ;
    labelstats.wrong_final_predictions = 0 ;
    labelstats.outofrange_final_predictions = 0 ;

    std::cout << "\n" << "\033[1;35m" <<  label << "\033[0m" << std::endl;

    int posofmin =0;

    size_t i =0;
    for (const auto& lab : comp.labels)
    {
        if (label == lab)
        {
            size_t j =0;
            float min_diff =SquaredDistance(comp.embeddings[i],ref.embeddings[j]);
            for (const auto& lab2 : ref.labels)
            {
                sdist = SquaredDistance(ref.embeddings[j],comp.embeddings[i]);

                if (sdist > thresh)
                {
                    labelstats.outofrange_predictions++ ;
                    //std::cout << "\033[1;33m"<< "out of range: " << lab << " " << lab2 << "\033[0m" << std::endl;

                } else {
                    if (lab2 == lab)
                    {
                        labelstats.right_predictions++;
                        //std::cout << "\033[1;32m"<< "good: " << lab << " " << lab2 << "\033[0m" << std::endl;
                       
                    } else {
                        labelstats.wrong_predictions++;
                        //std::cout << "\033[1;31m"<< "bad: " << lab << " " << lab2 << "\033[0m" << std::endl;
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
                if (ref.labels[posofmin] == lab)
                {
                    labelstats.right_final_predictions++ ;
                } else {
                    labelstats.wrong_final_predictions++ ;
                    std::cout << "\033[1;34m" << "confused " <<  ref.labels[posofmin] << " with " << lab << "\033[0m" << std::endl;

                }
            }
        }
        i++;
    }

    std::cout << "\033[1;32m" << "tot right: " <<  labelstats.right_predictions << "\033[0m" << "\033[1;31m" << " tot wrong: " <<  labelstats.wrong_predictions << "\033[0m" << "\033[1;33m" << " tot outofrange: " << labelstats.outofrange_predictions << "\033[0m" << std::endl;
    std::cout << "\033[1;32m" << "tot right_final: " <<  labelstats.right_final_predictions << "\033[0m" << "\033[1;31m" << " tot wrong_final: " <<  labelstats.wrong_final_predictions << "\033[0m" << "\033[1;33m" <<" tot outofrange_final: " << labelstats.outofrange_final_predictions << "\033[0m" << std::endl;

    return labelstats ;

}


std::vector<float> ConvStringToFloats(std::string str){

    std::vector<float> vect;
    //vect.reserve(512); // embedding is a 512 components vector in 20180408-102900.pb
    vect.reserve(128); // embedding is a 128 components vector in 20170512-110547.pb
    std::istringstream stm(str) ;
    
    float number;

    while(stm >> number){
        vect.push_back(number);              
    }

    return vect;
}

bool isFloat(std::string s){
    std::istringstream iss(s);
    float dummy;
    iss >> std::noskipws >> dummy;
    return iss && iss.eof();     // Result converted to bool
}

std::vector<std::string> ReadLabelsAndEmbeddings(const std::string file_name){
    
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

std::vector<float>
vectmean(const std::vector<std::vector<float>>& input) {
    std::vector<float> ret;

    if (not input.empty()) {
        ret.resize(input[0].size());

        for (const auto& i : input) {
            for (size_t n=0; n<i.size(); n++)
                ret[n] += i[n];
        }

        float ni = input.size();
        for (auto& v : ret)
            v /= ni;
    }

    return ret;
}

std::vector<datasetPoint> CreateDataPoints(dataSet database){
    std::vector<datasetPoint> points;

    for (const auto& label : database.unique_labels)
    {
        datasetPoint point;
        std::vector<float> meanposition;
        meanposition.reserve(database.embeddings[0].size());

        std::vector<std::vector<float>> embgoodlabel;
        float variance = 0.;

        for (int i = 0; i < database.labels.size(); ++i)
        {
            
            if (database.labels[i] == label)
            {
                embgoodlabel.push_back(database.embeddings[i]);
            }
            
        }

        meanposition = vectmean(embgoodlabel);       

        for (int i = 0; i < embgoodlabel.size(); ++i)
        {
           variance += SquaredDistance(meanposition,embgoodlabel[i]);
        }

        variance = sqrt(variance / embgoodlabel.size() /embgoodlabel.size()) ;

        point.meanposition = meanposition;
        point.label = label;
        point.dataradius = variance;

        std::cout << "point : " << label << " variance : " << variance <<std::endl;
        std::cout << "meanpos size " << meanposition.size() << " vector size : " << database.embeddings[0].size() <<std::endl;
        points.push_back(point);

    }
    return points;

}