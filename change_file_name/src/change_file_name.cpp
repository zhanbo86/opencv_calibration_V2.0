#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>
#include <iostream>
#include <vector>
#include <string.h>
#include <sstream>
#include <time.h>


using namespace std;

void int2str(const int &int_temp,string &string_temp)
{
        stringstream stream;
        stream<<int_temp;
        string_temp=stream.str();   //此处也可以用 stream>>string_temp
}

int main()
{
    DIR* pDir;
    struct dirent* ptr;
    vector<string> files;
    string myDir="/home/zb/BoZhan/OpenCV/Calibration/cal_image/";
    if( !(pDir = opendir("/home/zb/BoZhan/OpenCV/Calibration/cal_image")) )
        return -1;
    while( (ptr = readdir(pDir)) != 0 )
    {
      if(strcmp(ptr->d_name,".")!=0 && strcmp(ptr->d_name,"..")!=0)
      {
        files.push_back(ptr->d_name);
      }
    }
    closedir(pDir);

    string newName1 = "";
    string num1 = "";
    //rename files with random names
    for(int i=0;i<files.size();i++)
    {
      std::cout<<"files = "<<files[i]<<std::endl;
      int2str(i,num1);
      newName1 = myDir +"random" + num1 + ".bmp";
      if(!rename((myDir + files[i]).c_str(),newName1.c_str()))
      {
        std::cout << "rename success "<< std::endl;
      }
      else
      {
          std::cout << "rename error "<< std::endl;
      }
    }


    //rename files with uniform names
    string oldName = "";
    string newName2 = "";
    string num2 = "";
    for(int i=0;i<files.size();i++)
    {
      int2str(i,num2);
      oldName = myDir +"random" + num2 + ".bmp";
      newName2 = myDir + num2;
      if(!rename(oldName.c_str(),(newName2 + ".bmp").c_str()))
      {
        std::cout << "rename success "<< std::endl;
      }
      else
      {
          std::cout << "rename error "<< std::endl;
      }
    }
    return 0;
}
