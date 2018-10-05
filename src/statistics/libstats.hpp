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
#pragma once

#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include <random>
#include <ctime>

#include <cstring>
#include <string>
#include <set>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;

namespace filesys = std::experimental::filesystem;

struct datasetPoint
{
	std::string label;
	std::vector<float> meanposition;
	float sigma;
};


struct dataSet
{
	std::vector<string> labels;
	std::vector<std::vector<float>> embeddings;
	std::set<std::string> unique_labels;
	std::vector<datasetPoint> datasetFramework;
	float maxsigma;

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


std::vector<float> vectmean(const std::vector<std::vector<float>>& input);

std::vector<std::string> ReadLabelsAndEmbeddings(const std::string file_name);

std::vector<float> ConvStringToFloats(std::string str);

float MaxSigma (std::vector<datasetPoint> v);

bool isFloat(string s);

dataSet CreateDataSet(std::string file);

float SquaredDistance(std::vector<float> vect1, std::vector<float> vect2);

std::vector<datasetPoint> CreateDataMeanPoints(dataSet database);

dataHandler CreateDataHandler(std::string label, dataSet ref, dataSet comp,  float thresh);