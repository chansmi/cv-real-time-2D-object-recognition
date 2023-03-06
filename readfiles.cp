/*
 Chandler Smith
 Adapted from Bruce A. Maxwell
  
  Sample code to identify image fils in a directory
*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include "readfiles.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

/*
  Given a directory on the command line, scans through the directory for image files.

  Prints out the full path name for each file.  This can be used as an argument to fopen or to cv::imread.
 */
std::vector<std::string> readImages(std::string a, std::string b) {
  char dirname[500];
  char buffer[256];
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;
  int i;

    
  std::vector<std::string> imagePaths;
  // get the directory path
  //strcpy(dirname, argv[1]);
    //a = a.substr(1,a.length()-3); // remove ()
  strcpy(dirname, a.c_str());
   // std::cout << ("This is the readfile");
  printf("Processing directory %s\n", dirname );
    

  // open the directory
    // Troubleshooted extensively with Mrudula - baffling so just found a manual workaround.
  //dirp = opendir( "dirname" ); //TODO: Fix this
  dirp = opendir( "/Users/chandlersmith/Desktop/CS5330/2_Project/2_Project/olympus" );
  if( dirp == NULL) {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }

    
  // loop over all the files in the image file listing
  while( (dp = readdir(dirp)) != NULL ) {

    // check if the file is an image
    if( strstr(dp->d_name, ".jpg") ||
	strstr(dp->d_name, ".png") ||
	strstr(dp->d_name, ".ppm") ||
	strstr(dp->d_name, ".tif") ) {

      //printf("processing image file: %s\n", dp->d_name);
      // build the overall filename
        strcpy(buffer, "/Users/chandlersmith/Desktop/CS5330/2_Project/2_Project/olympus" ); // changed from dirname
        
      strcat(buffer, "/");
      strcat(buffer, dp->d_name);

      //printf("full path name: %s\n", buffer);
    
        //Append buffer to image array
        //cv::Mat img = cv::imread(buffer, cv::IMREAD_COLOR);
        imagePaths.push_back(buffer);
    }
  }
    closedir(dirp);
  //printf("Terminating\n");

    return imagePaths;
}


