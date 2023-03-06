//
//  main.cpp
//  3_Project
//
//  Created by Chandler Smith on 2/22/23.
//
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <vector>
#include <utility>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include "ObjectDetection.hpp"
#include <fstream>
#include <iomanip>
#include "readfiles.hpp"
#include "csv_util.h"
#include <fstream>
#include <sstream>
#include <cmath>



int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev;
    
    // open the video device
    capdev = new cv::VideoCapture(0);
    if( !capdev->isOpened() ) {
        printf("Unable to open video device\n");
        return(-1);
    }
    
    // get some properties of the image
    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                  (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);
    
    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame, testFrame;
    bool trainingMode = false;
    bool eucledean = false;
    bool knn = false;
    bool confusionMatrix = false;

    for(;;) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        
        //std::cout << color_mode << std::endl;
        if( frame.empty() ) {
            printf("frame is empty\n");
            break;
        }
        char key = cv::waitKey(10);
        // Quit
        if( key == 'q') {
            break;
        }
        
        // train data
        if( key == 'n') {
            trainingMode = true;
        }
        
        // calc Eucledean
        if( key == 'e') {
            eucledean = true;
        }
        
        // calc knn
        if( key == 'k') {
            knn = true;
        }
        
        // calc knn
        if( key == 'c') {
            confusionMatrix = true;
        }
        
        
        // Preprocess
        int MAX_KERNEL_LENGTH = 3;
        for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
        {
            cv::GaussianBlur( frame, frame, cv::Size( i, i ), 0, 0 );
        }
        //cv::medianBlur( frame, frame, 3);
        //cv::GaussianBlur(frame, frame, cv::Size);
        
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        
        // reference to other code
        // Thresholding
        // parameters: src, dst, value assigned, method, threshold type, block size, constant
        //cv::adaptiveThreshold(frame, frame, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 5, 4);
        cv::threshold(frame, frame, 60, 255, cv::THRESH_BINARY);
        // result is a image identifier
        
        // morphological
        cv::Mat kernel = cv::Mat::ones(1, 1, CV_8U);
        // erode and then dialate
        //cv::morphologyEx(frame, frame, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(frame, frame, cv::MORPH_OPEN, kernel);
        
        // Segmentation
        cv::Mat labels, stats, centroids;
        // Label is an int for a connected component, or 0 if not.
        // Stats contains
        int vid_statistics = cv::connectedComponentsWithStats(frame, labels, stats, centroids);
        // compute features
        // to calculate region id, basically do a depth first traversal of the region map and label an id
        cv::Mat regionMap = frame;
        cv::Mat area_interest;
        cv::Point2f axisLCM;
        cv::RotatedRect orientedBB;
        float target1, target2, target3, target4;
        std::vector<float> target;
        float Alpha = 0;
        cv::Point centerPoint;
        //std::cout << "LOOK HERE" << "\n" << labels;
        for (int i = 1; i < vid_statistics; i++){
            area_interest = (labels == i);
            // Compute features
            cv::Mat mask = (regionMap == area_interest);
            cv::Moments moments = cv::moments(mask, true);
            Alpha = 0.5*atan2((2 * moments.mu11), (moments.mu20 - moments.mu02));
            //cv::Point centerPoint;
            centerPoint.x = moments.m10 / moments.m00;
            centerPoint.y = moments.m01 / moments.m00;
            cv::Point endPoint;
            endPoint.x = centerPoint.x + 250 * cos(Alpha);
            endPoint.y = centerPoint.y + 250 * sin(Alpha);
            cv::line(regionMap, centerPoint, endPoint, cv::Scalar(160, 174, 240), 3);
            
            // Bounding box computation
            int maxX = -1000000;
            int minX = 1000000;
            int maxY = -1000000;
            int minY = 1000000;
            
            for (int y = 0; y < mask.rows; y++) {
                for (int x = 0; x < mask.cols; x++) {
                    if (mask.at<uchar>(y, x)) {
                        int x_prime = (x - centerPoint.x) * cos(Alpha) + (y - centerPoint.y) * sin(Alpha);
                        int y_prime = (x - centerPoint.x) * -sin(Alpha) +
                        (y - centerPoint.y) * cos(Alpha);
                        if (x_prime > maxX) {maxX = x_prime;}
                        if (x_prime < minX) {minX = x_prime; }
                        if (y_prime > maxY) {maxY = y_prime;}
                        if (y_prime < minY) {minY = y_prime;}
                    }
                }
            }
            cv::Point A, B, C, D;
            A.x = (maxX * cos(Alpha) - maxY * sin(Alpha)) + centerPoint.x;
            A.y = (maxX * sin(Alpha) + maxY * cos(Alpha)) + centerPoint.y;
            B.x = (maxX * cos(Alpha) - minY * sin(Alpha)) + centerPoint.x;
            B.y = (maxX * sin(Alpha) + minY * cos(Alpha)) + centerPoint.y;
            C.x = (minX * cos(Alpha) - minY * sin(Alpha)) + centerPoint.x;
            C.y = (minX * sin(Alpha) + minY * cos(Alpha)) + centerPoint.y;
            D.x = (minX * cos(Alpha) - maxY * sin(Alpha)) + centerPoint.x;
            D.y = (minX * sin(Alpha) + maxY * cos(Alpha)) + centerPoint.y;
            if (stats.at<int>(i, cv::CC_STAT_AREA) > 1000){
                cv::line(regionMap, A, B, cv::Scalar(100, 174, 240), 2);
                cv::line(regionMap, B, C, cv::Scalar(100, 174, 240), 2);
                cv::line(regionMap, C, D, cv::Scalar(100, 174, 240), 2);
                cv::line(regionMap, D, A, cv::Scalar(100, 174, 240), 2);
            }
            
            float width = sqrt((A.x - B.x)*(A.x - B.x) + (A.y - B.y)*(A.y - B.y));
            float height = sqrt((B.x - C.x)*(B.x - C.x) + (B.y - C.y)*(B.y - C.y));
            
            float heightDivWidth = height / width;
            
            target1 = moments.nu20 + moments.nu02;
            target2 = (moments.nu20 - moments.nu02)*(moments.nu20 - moments.nu02) + (4*moments.nu11)*(4*moments.nu11);
            target3 = (moments.nu30 + moments.nu12)*(moments.nu30 + moments.nu12) + (moments.nu21 + moments.nu03)*(moments.nu21 + moments.nu03);
            target4 = heightDivWidth;
            target.push_back(target1);
            target.push_back(target2);
            target.push_back(target3);
            target.push_back(target4);
            
            //target = [target1, target2, ]
            
            
            
            // Collect training data
            while(trainingMode == true && key != 'q'){
                std::cout << "\n" << "Enter training label: ";
                std::string label;
                getline(std::cin, label);
                std::string filepath = "trainingData.csv";
                std::ofstream file;
                
                //write to csv
                std::ofstream myFile;
                myFile.open("/Users/chandlersmith/Desktop/CS5330/3_Project/3_Project/foo.csv", std::ios_base::app);
                
                // Send data to the stream
                myFile << label << ",";
                myFile << moments.nu20 + moments.nu02 << ",";
                myFile << (moments.nu20 - moments.nu02)*(moments.nu20 - moments.nu02) + (4*moments.nu11)*(4*moments.nu11) << ",";
                myFile << (moments.nu30 + moments.nu12)*(moments.nu30 + moments.nu12) + (moments.nu21 + moments.nu03)*(moments.nu21 + moments.nu03) << ",";
                myFile << heightDivWidth << "\n";
                
                // Close the file
                myFile.close();
                return 0;
            }
        }
        
        // Classify new images
        // euclidean
        // calc SD of every feature by reading csv, cite: GFG
        
        std::vector< char*> training_file_names;
        std::vector<std::vector<float>> fileFeatures;
        const char* csvFilename = "/Users/chandlersmith/Desktop/CS5330/3_Project/3_Project/foo.csv";
        read_image_data_csv(csvFilename, training_file_names, fileFeatures);
        //std::cout<<"Loaded feature set size: " << fileFeatures.size() <<std::endl;
        /*
         [[1test, 1, 2, 3][2test, 1, 2, 3, 4]
         */
        //std::cout << fileFeatures[0].size();
        // Standard Deviation
        std::vector<pair<float, string>> EucledeanSortVec;
        std::vector<double> featureSD;
        for(int i=0; i<fileFeatures[0].size(); i++){
            double sum = 0.0;
            double mean = 0.0;
            double SD = 0.0;
            for( int j = 0; j < fileFeatures.size(); j++){
                sum += fileFeatures[j][i];
            }
            mean = sum/fileFeatures.size();
            for( int k = 0; k < fileFeatures.size(); k++){
                double difference = fileFeatures[k][i] - mean;
                SD += difference * difference;
            }
            SD = sqrt(SD/fileFeatures.size());
            featureSD.push_back(SD);
        }
        for (int i = 0; i < featureSD.size(); i++) {
            std::cout << "Feature " << i + 1 << " standard deviation: " << featureSD[i] << std::endl;
        }
        
        if(eucledean == true){
            for(int i=0; i<fileFeatures.size(); i++){ // for each training set
                float distA = ((target[0] - fileFeatures[i][0]) / featureSD[0]) * ((target[0] - fileFeatures[i][0]) / featureSD[0]);
                float distB = ((target[1] - fileFeatures[i][1]) / featureSD[1]) * ((target[1] - fileFeatures[i][1]) / featureSD[1]);
                float distC = ((target[2] - fileFeatures[i][2]) / featureSD[2]) * ((target[2] - fileFeatures[i][2]) / featureSD[2]);
                float distD = ((target[3] - fileFeatures[i][3]) / featureSD[3]) * ((target[3] - fileFeatures[i][3]) / featureSD[3]);
                
                float distance = sqrt(distA + distB + distC + distD);
                EucledeanSortVec.push_back(make_pair( distance, training_file_names[i]));
                
            }
            sort(EucledeanSortVec.begin(),EucledeanSortVec.end());
            cout << "\n Eucledean Classification for this image: " << EucledeanSortVec[0].second;
        }
        
        if(knn == true){
            // put into a vector
            // target - vector of eucledeansortvec[0][1][2]
            if((EucledeanSortVec[0] == EucledeanSortVec[1]) || (EucledeanSortVec[0] == EucledeanSortVec[2])) {
                cout << "\n KNN Classification for this image: " << EucledeanSortVec[0].second;
            }
            else if ((EucledeanSortVec[1] == EucledeanSortVec[2])){
                cout << "\n KNN Classification for this image: " << EucledeanSortVec[1].second;
            }
            else {
                cout << "\n KNN Classification for this image: " << EucledeanSortVec[0].second;
            }
            
        }
        
        // Confusion matrix
        // take all the pen files
        // Run it though the data
        // then pass it through
        // Run a note it
        if( confusionMatrix == true){
            for(int k = 0; k < 6; k++){
                target = fileFeatures[k];
                for(int i=0; i<fileFeatures.size(); i++){ // for each training set
                    float distA = ((target[0] - fileFeatures[i][0]) / featureSD[0]) * ((target[0] - fileFeatures[i][0]) / featureSD[0]);
                    float distB = ((target[1] - fileFeatures[i][1]) / featureSD[1]) * ((target[1] - fileFeatures[i][1]) / featureSD[1]);
                    float distC = ((target[2] - fileFeatures[i][2]) / featureSD[2]) * ((target[2] - fileFeatures[i][2]) / featureSD[2]);
                    float distD = ((target[3] - fileFeatures[i][3]) / featureSD[3]) * ((target[3] - fileFeatures[i][3]) / featureSD[3]);
                    
                    float distance = sqrt(distA + distB + distC + distD);
                    EucledeanSortVec.push_back(make_pair( distance, training_file_names[i]));
                    
                }
                sort(EucledeanSortVec.begin(),EucledeanSortVec.end());
                cout << "\n Eucledean Classification for this image: " << EucledeanSortVec[0].second;
            }
        }
        cv::imshow("Video", regionMap);
    }
    delete capdev;
    return(0);
}


//std::cout << vid_statistics << "\n";
// Give your system the ability to display the regions it finds
//std::cout << "\n" << "object count: " << vid_statistics;
// TODO: go back and figure out the how to do this by N largest regions
// TODO: Abstract this function out
// Cite: https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/enum_cv_ConnectedComponentsTypes.html
// for each object
/*for (int i = 1; i < vid_statistics; i++){
    // limit by size
    if (stats.at<int>(i, cv::CC_STAT_AREA) > 1000){
        //cv::rectangle(frame, cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP), stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT)), cv::Scalar(0, 255, 0), 2);
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        cv::Rect objectBox = cv::Rect(x, y, width, height);
        cv::Point a, b, c, d;
      */
        
        //int x_temp = (x - center.x) * cos(alpha) + (y - center.y) * sin(alpha);
       // int y_temp = (x - center.x) * -sin(alpha) + (y - center.y) * cos(alpha);
        
        //cv::rectangle(frame, objectBox, cv::Scalar(160, 174, 240), 2);
        
// at the end I want the standard deviation for
// 1, 2, 3, 4 measurements
// I then can use that to iterate through comparing target1, to training1, / SD1


/* fstream fin;
fin.open("/Users/chandlersmith/Desktop/CS5330/3_Project/3_Project/foo.csv", ios::in);
vector<string> row;
string line, word, temp;
while( fin >> temp){
    row.clear();
    getline(fin, line);
    stringstream s(line);
    while(getline(s, word, ',')){
        row.push_back(word);
    }
    
}

std::cout << row[0];
    } */
// compute scaled Eucledean
// d =√[(x2 – x1)2 + (y2 – y1)2]
// Euclidean distance metric [ (x_1 - x_2) / stdev_x ]



// Target image and training data
// Whatever has the least distance, assign that label to the target image

/*
// Different classifier
//KNN
//

// Confusion matrix for evaluating performance
// opencv

*/
