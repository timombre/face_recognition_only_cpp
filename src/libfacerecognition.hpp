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
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

#include <iostream>
#include <fstream>


#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/client/client_session.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <experimental/filesystem>
#include <random>
#include <ctime>

#include "statistics/libstats.hpp"
 
using namespace std;
using namespace cv;
using namespace tensorflow;
using namespace cv::face;

namespace filesys = std::experimental::filesystem;


std::vector<std::string> getAllFilesInDir(const std::string &dirPath, bool dir);


static tensorflow::Status ReadEntireFile(tensorflow::Env* env, const std::string& filename, Tensor* output);

tensorflow::Status LoadGraph(const std::string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session);


std::unique_ptr<tensorflow::Session> initSession(std::string graphpath);

cv::Mat faceCenterRotateCrop(Mat &im, vector<Point2f> landmarks, Rect face, int i, bool show_crop);

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    Ptr<Facemark> facemark,
                    double scale, std::unique_ptr<tensorflow::Session>* session,
                    dataSet database,
                    bool show_crop, float thresh, bool volume_points);


std::string genEmbeddings(CascadeClassifier cascade, Ptr<Facemark> facemark, tensorflow::Session& session, std::string filename,
                   std::string label,  bool gen_dt, std::string data_root);

std::vector<float> ConvStringToFloats(std::string str);