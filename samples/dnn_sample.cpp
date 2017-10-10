#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>
#include <fstream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	const String keys = 
			"{@path        |.     | prototxt path     }"
			"{@path        |.     | binary model path }"
        ;
	
	CommandLineParser parser(argc, argv, keys);
	
	String protoSrc = parser.get<String>(0);
	String binSrc = parser.get<String>(1);
	
	dnn::Net net = dnn::readNetFromCaffe(protoSrc, binSrc);

	
	string imName = "";
	while(true) {
		cout << "Image path:\n";
		cin >> imName;

		if(imName == "") continue;
		
		Mat im = imread(imName);		
		Mat m = dnn::blobFromImage(im, 1.0, Size(224,224));
		
		net.setInput(m, "data");
		
		Mat res = net.forward("prob");
					
		double max;
		Point maxloc;
		
		minMaxLoc(res, nullptr, &max, nullptr, &maxloc);
		
		cout << "Max: " << max << " at " << maxloc.x << ", " << maxloc.y << "\n";
		
		ifstream infile("synset_words.txt");	
		
		int i=0;
		string classId, name;
		while(i++ < maxloc.x){
			infile >> classId;
			getline(infile, name);
		}
		
		cout << name << "\n";
		
		string win = "Scaled";
		
		Mat scaledIm;
		resize(im, scaledIm, Size(240, 240));
		putText(scaledIm, name, Point2f(20, 20), FONT_HERSHEY_PLAIN, 2, Scalar(0,0,200,200));
		namedWindow(win);
		imshow(win, scaledIm);
		
		waitKey();
		destroyAllWindows();
	}
	
	
    return 0;
}
