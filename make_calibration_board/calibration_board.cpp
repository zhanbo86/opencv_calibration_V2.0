#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

int main(int argc, char*argv[])
{
    int width = 14;//棋盘格宽度
    int height = 14;//棋盘格高度

    Mat src(98,126,CV_8UC1,Scalar(0, 0, 0));
    uchar* p_src = src.ptr<uchar>(0);
    for (int i = 0; i < src.rows; i++)
    {
        p_src = src.ptr<uchar>(i);
        for (int j = 0; j < src.cols; j++)
        {
            if ((i / width + j / height) % 2 != 0)
            {
              p_src[j] = 255;
            }
        }
    }

    namedWindow("ChessBoard", 0);
    imshow("ChessBoard", src);
    waitKey(0);
    imwrite("ChessBoard.bmp", src);
    cvDestroyWindow("ChessBoard");

    return 0;


}
