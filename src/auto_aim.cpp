#include "auto_aim.h"
#include <string>
#include <iostream>
#include <algorithm>
#include "time.h"
const String fileName = "../res/pic2/my_photo-173.jpg";
const Scalar Bluelow(100,43,46);
const Scalar Bluehigh(124,255,255);
const float AutoAim::max_offset_angle = 30;

AutoAim::AutoAim(){}

AutoAim::~AutoAim(){}

//排序得到的椭圆，使得角度近似、高度近似的点邻近
bool cmp(RotatedRect &x, RotatedRect &y){
    float theta_x=x.angle;
    float theta_y=y.angle;
    if(x.angle>95)
        theta_x=abs(180-theta_x);
    if(y.angle>95)
        theta_y=abs(180-theta_y);
   if(theta_x!=theta_y) return theta_x>theta_y;
   if(x.size.height!=y.size.height) return x.size.height<y.size.height;
   return x.size.width>y.size.width;
}

//两点距离
float distPoint(Point2f center1, Point2f center2){
    return abs(center1.x-center2.x);//sqrt((center1.x-center2.x)*(center1.x-center2.x) + (center1.y-center2.y)*(center1.y-center2.y));
}

void AutoAim::setImage(Mat &img, Mat &mask){
    Mat hsv;
    Mat channel[3];
    //GaussianBlur(img,img,Size(5,5),0,0);
    split(img,channel); 
    //dilate(mask,mask,cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5),cv::Point(-1, -1)));
    threshold(channel[0]-channel[2], mask, 0, 255, THRESH_BINARY+THRESH_OTSU); //自适应阈值
    morphologyEx(mask, mask, MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7), cv::Point(-1, -1))); //开运算消除小物块，平滑物体的边界
    Canny(mask, mask, 3, 9, 3);
   // imshow("mask", mask);
}
void AutoAim::findLamp(Mat &mask, vector<RotatedRect> &lamps,Mat &src){
    
    lamps.clear();
    vector<vector<Point> > contours;
    vector<Vec4i> hierarcy;
    //寻找轮廓，将满足条件的轮廓放入待确定的数组中去
    findContours(mask, contours, hierarcy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
    RotatedRect temp;
    vector<RotatedRect> pre_lamps;
    
    cout<<"contours.size: "<<contours.size()<<endl;
    if(contours.size()<30){
    for(int i=0;i<contours.size();i++){
        if(i%2==0)
        {
            if(contours[i].size()>5){
                 temp =fitEllipseAMS(Mat(contours[i]));
            
                float theta=temp.angle;
                if(theta>95)
                    theta=abs(180-theta);

                if(theta<40){
                    
                    
                //判断是否放入的条件 
                    if(max(temp.size.width, temp.size.height) < min(temp.size.width, temp.size.height)*1.2)
			            continue;
                    if(temp.size.width<5||temp.size.height<5)
                        continue;                    
                    ellipse(src, temp.center, Size(temp.size.width/2, temp.size.height/2), temp.angle, 0, 360, Scalar(0, 255, 0), 1, 8);
                    
                    //ellipse(src, temp.center, Size(temp.size.width/2, temp.size.height/2), temp.angle, 0, 360, Scalar(0, 255, 0), 1, 8);
                    putText(src, to_string(i), temp.center, FONT_HERSHEY_SIMPLEX,1, Scalar(255,23,0), 2, 8);
                    cout<<i<<" : "<<temp.size.width<<" "<<temp.size.height<<" "<<temp.angle<<endl;
                    //ellipse(src, temp, Scalar(255, 0, 0), 2, 8);
                    pre_lamps.push_back(temp);

                
                }
            }
        } 
    }
    }

    //排序lamp找到满足比例条件的灯管
    sort(pre_lamps.begin(), pre_lamps.end(), cmp);
    
    //角度和高度的权重，角度更加重要，所以角度的偏差使得结果的值偏差更大
    int angle_diff_weight = 3;
    int height_diff_weight = 1;

    //按照比例　两灯管中心点之间的距离：两灯管长度的平均 近似等于 2:1
    //这里只跟右边三个进行比较
    int size = pre_lamps.size();
    vector<float> diff(size,0);
    //float *diff = new float(size);
    vector<float> best_match_index(size,0);
    //float *best_match_index = new float(size);
    for(int i=0; i<size; i++){
        diff[i] = 0x3f3f3f3f;
        best_match_index[i] = -1;

        //cout<<"angle: "<<pre_lamps[i].angle;
        //cout<<" length: "<<pre_lamps[i].size.height<<endl;
        //cout<<"x: "<<pre_lamps[i].center.x<<endl;
    }
    
    //中心点间距离，平均高度，角度差，高度差
    float dist, avg_height, diff_angle, diff_height, ratio, totalDiff;
    int i,j;
    for(i=0; i<size; i++){
        float currDiff = 0x3f3f3f3f;
        int currIndex = -1;
        const RotatedRect &current = pre_lamps[i];
        float theta_current=abs(current.angle);
        if(theta_current>95)
                    theta_current=abs(180-theta_current);
        for(j=1; j<=4 || (i+j)<size; j++){
            //计算比例，筛选灯管
            
            const RotatedRect &compare = pre_lamps[i+j];

            float theta_compare=abs(compare.angle);
            if(theta_compare>95)
                    theta_compare=abs(180-theta_compare);
            diff_angle = abs(theta_compare - theta_current);
            
            if(diff_angle > 15) continue;
            diff_height = abs(compare.size.height - current.size.height);
            if(diff_height>30) continue;
            dist = distPoint(compare.center, current.center);
           
            cout<<"distt: "<<dist<<endl;
            if(dist>10*current.size.height) continue;
            avg_height = (compare.size.height + current.size.height) / 2;
            ratio = dist / avg_height;
            cout<<"ratio: "<<ratio<<endl;
            if(ratio > 10) continue;
            
            totalDiff = angle_diff_weight*diff_angle + height_diff_weight*diff_height+0.5*dist;
            if(totalDiff < currDiff){
                currDiff = totalDiff;
                currIndex = i+j;
            }
        }
        //一对灯管肯定花费是最少的，所以如果当前花费比两个的花费都要少，就记录是最优
        if(currIndex==-1) continue;
        if(currDiff < diff[i] && currDiff < diff[currIndex]){
            diff[i] = currDiff;
            diff[currIndex] = currDiff;
            best_match_index[i] = currIndex;
            best_match_index[currIndex] = i;
        }
    }

    //遍历，将满足条件的灯管储存
    for(i=0; i<size; i++){
        int index = best_match_index[i];
        cout<<"best_match_index: "<<index;
        cout<<" diff: "<<diff[i]<<endl;
        if(index == -1 || index <= i) continue;
        //找到匹配对
        if(i == best_match_index[index]){
            lamps.push_back(pre_lamps[i]);
            lamps.push_back(pre_lamps[index]);
        }
    }
    cout<<"lamps counts: "<<lamps.size()/2<<endl;
}

int main(int argc, char const *argv[]){
    Mat src = imread(fileName);
    clock_t start,finish;
    double time_tol;
    
    AutoAim autoAim;
    Mat mask;
    start=clock();
    autoAim.setImage(src, mask);
     
    vector<RotatedRect> lamps;
    
    autoAim.findLamp(mask, lamps,src);
    cout<<lamps.size()<<endl;
    for(int i=0;i<lamps.size();i++)
    {
        cout<<i<<" "<<lamps[i].center.x<<endl;
        circle(src,lamps[i].center,20,(255,255,0),2);
    }
    imshow("src",src);
    finish=clock();
    time_tol=double(finish-start)/CLOCKS_PER_SEC;
    cout<<time_tol<<endl;
    waitKey(0);
    return 0;
}
