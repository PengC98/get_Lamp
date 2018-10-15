#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include<algorithm>
using namespace std;
using namespace cv;

const Scalar Redlow2(156,40,40);
const Scalar Redhigh2(180,255,255);
const Scalar Redlow1(0,40,40);
const Scalar Redhigh1(10,255,255);
const Scalar Bluelow(100,43,46);
const Scalar Bluehigh(124,255,255);
void match(Mat &img);
bool cmp(vector<RotatedRect> x,vector<RotatedRect> y)
{
   if(x.angle!=y.angle) return x.angle>y.angle;
   if(x.size.height!=y.size.height) return x.size.height>y.size.height;
   return x.size.width>y.size.width;
}
void threshold(Mat &img)
{
    Mat hsv;
    Mat mask;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarcy;
    VideoCapture cap;
    //cap.open("../res/1.mp4");
    Mat channel[3];
    GaussianBlur(img,img,Size(5,5),0,0);
    //match(img);
    split(img,channel);
    cvtColor(img,hsv,COLOR_BGR2HSV);
    inRange(hsv,Bluelow,Bluehigh,mask);
    imshow("hsv",mask);
    morphologyEx(channel[0]-channel[2], mask, MORPH_BLACKHAT, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*7+1, 2*7+1), cv::Point(-1, -1)));
    morphologyEx(mask, mask, MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5 ), cv::Point(-1, -1)));
    threshold(mask, mask, 0, 255, THRESH_BINARY+THRESH_OTSU);
    Canny(mask, mask, 3, 9, 3);
    //threshold(mask, mask, 5, 255, CV_THRESH_BINARY_INV);
    findContours(mask, contours, hierarcy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    vector<RotatedRect> box(contours.size());
    vector<RotatedRect> lamp(contours.size());
    Point2f rect[4];
    cout<<contours.size()<<endl;
    int j=0;
    for(int i=0;i<contours.size();i++){
        if(contours[i].size()>5)
        {
            box[i] = fitEllipse(Mat(contours[i]));

            float theta=box[i].angle;
            cout<<theta<<endl;
            if(abs(180-theta)<30||(theta-0)<30)
             {   
                if( MAX(box[i].size.width, box[i].size.height) < MIN(box[i].size.width, box[i].size.height)*2 )
			        continue;
                if(box[i].size.width>box[i].size.height)
                    continue;
                if(i%2==0)
                    continue;
            
                ellipse(img, box[i].center, Size(box[i].size.width/2, box[i].size.height/2), box[i].angle, 0, 360, Scalar(0, 255, 0), 1, 8);
                putText(img,to_string(i),box[i].center,FONT_HERSHEY_SIMPLEX,1,Scalar(255,23,0),2,8);
                ellipse(img, box[i], Scalar(255, 0, 0), 2, 8);
                lamp[j]=box[i];
                j++;
                //cv::rectangle(img,Rect(100,300,20,200),Scalar(0,0,255),1,1,0);
            }
        }
    }
    cout<<j<<endl;
    vector<RotatedRect> lamp_useful(j);
    for(int i=0;i<j;i++)
    {
        lamp_useful[i]=lamp[i];
    }
    cout<<lamp_useful.size()<<endl;
    sort(lamp_useful,lamp_useful+j,cmp);
    if(j%2==0&&j!=0)
    {

        float diff_last=1                    if(diff<diff_last)
                    {
                        
                    }00000;
        vector<RotatedRect> left_lamp[j/2],right_lamp[j/2];
        for(int i=0;i<j;i++)
        {
        
            for(int k=0;k<j;k++)
            {
                if(i==k)
                    continue;
                if(lamp_useful[i].angle!=0&&lamp_useful[k]!=0)
                {
                    diff=abs(lamp_useful[i].angle-lamp_useful[k].angle)+abs(lamp_useful[i].size.height-lamp_useful[k].size.height);
                    if(diff<diff_last)
                    {

                    }
                }
            }
        }
    }

    imshow("cv",mask);
    imshow("img",channel[0]-channel[2]);
    imshow("src",img);
    waitKey(1);
}
void match(Mat &img)
{
    /*SiftFeaturesFinder sift(20);
    
    
		
	vector<KeyPoint> keypoints_1, keypoints_2;
	sift.detect( image1, keypoints_1 );	//FAST(src_1, keypoints_1, 20); 
	sift.detect( img, keypoints_2 );	//FAST(src_2, keypoints_2, 20); 
    Mat imageDesc1, imageDesc2;
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    descriptor->compute( image1, keypoints_1, imageDesc1);
    descriptor->compute( img, keypoints_2, imageDesc2);
    //vector<DMatch> matches;
    //BFMatcher matcher ( NORM_HAMMING );
    //matcher->match ( imageDesc1, imageDesc2, matches );
    FlannBasedMatcher matcher;
    vector<vector<DMatch> > matchePoints;
    vector<DMatch> GoodMatchePoints;
    vector<Mat> train_desc(1, imageDesc1);
    matcher.add(train_desc);
    matcher.train();
    for (int i = 0; i < matchePoints.size(); i++)
    {
        if (matchePoints[i][0].distance < 0.6 * matchePoints[i][1].distance)
        {
            GoodMatchePoints.push_back(matchePoints[i][0]);
        }
    }

    Mat first_match;
    drawMatches(image02, keyPoint2, image01, keyPoint1, GoodMatchePoints, first_match);
    imshow("first_match ", first_match);
    waitKey();*/
    Mat image1=imread("../res/test4.jpg", CV_LOAD_IMAGE_COLOR);
    namedWindow("DMatch");

    vector<KeyPoint> keyPoint1,keyPoint2;
    Mat descriptor1,descriptor2;
    Ptr<ORB> orb = ORB::create(500);

    orb->detectAndCompute(image1,Mat(),keyPoint1,descriptor1);
    orb->detectAndCompute(img,Mat(),keyPoint2,descriptor2);

    vector<DMatch> match;
    //暴力匹配
    BFMatcher bfMatcher(NORM_HAMMING);
    //快速最近邻逼近搜索函数库
    //FlannBasedMatcher fbMatcher(NORM_HAMMING);
    Mat outImg;
    bfMatcher.match(descriptor1,descriptor2,match,Mat());

    double dist_min = 1000;
    double dist_max = 0;
    for(size_t t = 0;t<match.size();t++){
        double dist = match[t].distance;
        if(dist<dist_min) dist_min = dist;
        if(dist>dist_max) dist_max = dist;
    }

    vector<DMatch> goodMatch;
    for(size_t t = 0;t<match.size();t++){
        double dist = match[t].distance;
        if(dist <= max(2*dist_min,30.0))
            goodMatch.push_back(match[t]);
    }
    drawMatches(image1,keyPoint1,img,keyPoint2,goodMatch,outImg);
    imshow("DMatch",outImg);
    waitKey(0);
                                                       
}
int main()
{

    Mat img= imread("../res/my_photo-28.jpg", CV_LOAD_IMAGE_COLOR);
    threshold(img);
    /*while(1)
    {
        cap>>img;
        split(img,channel);
        resize(img,img,Size(504,672), (0, 0), (0, 0), INTER_LINEAR);
        GaussianBlur(img,img,Size(5,5),0,0);
        Mat dst = img.clone();
        cvtColor(img,hsv,COLOR_BGR2HSV);
        inRange(hsv,Bluelow,Bluehigh,mask);
        findContours(mask, contours, hierarcy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
        split(img,channel);
        //morphologyEx(channel[0]-channel[2], mask, MORPH_BLACKHAT, getStructuringElement(MORPH_RECT, Size(8,8)));
        //threshold(mask, mask, 80, 255, CV_THRESH_BINARY_INV);
        //cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*4+1, 2*4+1), cv::Point(4, 4));
        //dilate(mask, mask, element);
        //dilate(mask, mask, element);
        //等价于cap.read(frame);
        if(img.empty())//如果某帧为空则退出循环
            break;
        imshow("cv",mask);
        imshow("img",img);
        waitKey(30);//每帧延时20毫秒
    }
    cap.release();//释放资源
    */

    //morphologyEx(mask, mask, MORPH_TOPHAT, getStructuringElement(MORPH_RECT, Size(8,8)));
    //morphologyEx(mask, mask, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3,3)));



    waitKey(0);

}