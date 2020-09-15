#include "dhe.hpp"


void build_is_hist(Mat img, Mat& hist_i, Mat &hist_s){
    int hei=img.rows, wid=img.cols, ch=img.channels();
    Mat Img = cv::Mat::zeros(hei+4, wid+4, img.type());
    
    Mat Imgs[ch],imgs[ch],hsvs[3];
    split(img,imgs);
    for(int i=0;i<ch;i++){
        Imgs[i] = cv::Mat::zeros(hei+4, wid+4, CV_8U);
        cv::copyMakeBorder(imgs[i], Imgs[i], 2, 2, 2, 2, BORDER_REPLICATE);
    }
    merge(Imgs,3,Img);
    Mat hsv;
    cv::cvtColor(Img, hsv, COLOR_RGB2HSV);
    split(hsv,hsvs);
    hsvs[0] = hsvs[0]*255;
    hsvs[1] = hsvs[1]*255;
    merge(hsvs,3,hsv);
    cv::threshold(hsv, hsv, 0, 255, THRESH_TOZERO);
    hsv.convertTo(hsv, CV_8U);
    hsv.convertTo(hsv, CV_32F);
    split(hsv,hsvs);
    Mat H=hsvs[0],S=hsvs[1],I=hsvs[2];//undo I 不一致
    Mat fh,fv;
    fh = (cv::Mat_<char>(3,3) << 1,0,-1,2,0,-2,1,0,-1);
    fv = (cv::Mat_<char>(3,3) << 1,2,1,0,0,0,-1,-2,-1);
    Mat dIh,dIv,dI,di;
    filter2D(I,dIh,I.depth(),fh);
    filter2D(I,dIv,I.depth(),fv);
    for(int i=0;i<dIh.rows;i++){
        for(int j=0;j<dIh.cols;j++){
            if(abs(float(dIh.at<float>(i,j))<0.000001)){
                dIh.at<float>(i,j)=0.00001;
            }
        }
    }
    
    for(int i=0;i<dIv.rows;i++){
        for(int j=0;j<dIv.cols;j++){
            if(abs(float(dIv.at<float>(i,j))<0.000001)){
                dIv.at<float>(i,j)=0.00001;
            }
        }
    }
    cv::sqrt(dIh.mul(dIh)+dIv.mul(dIv), dI);
    dI.convertTo(dI, CV_32S);
    Point p1=Point(2,2);
    Point p2=Point(2+wid,2+hei);
    di=dI(Rect(p1,p2));
    Mat dSh,dSv,dS,ds;
    filter2D(S,dSh,S.depth(),fh);
    filter2D(S,dSv,S.depth(),fv);
    for(int i=0;i<dSh.rows;i++){
        for(int j=0;j<dSh.cols;j++){
            if(abs(float(dSh.at<float>(i,j))<0.000001)){
                dSh.at<float>(i,j)=0.00001;
            }
        }
    }
    
    for(int i=0;i<dSv.rows;i++){
        for(int j=0;j<dSv.cols;j++){
            if(abs(float(dSv.at<float>(i,j))<0.000001)){
                dSv.at<float>(i,j)=0.00001;
            }
        }
    }
    cv::sqrt(dSh.mul(dSh)+dSv.mul(dSv), dS);
    dS.convertTo(dS, CV_32S);
    p1=Point(2,2);
    p2=Point(2+wid,2+hei);
    ds=dS(Rect(p1,p2));
    p1=Point(2,2);
    p2=Point(2+wid,2+hei);
    Mat i = I(Rect(p1,p2));
    
    Mat Rho = cv::Mat::ones(hei+4,wid+4,CV_32F);
    for(int p=2;p<hei+2;p++){
        for(int q=2;q<wid+2;q++){
            p1=Point(q-2,p-2);
            p2=Point(q+3,p+3);
            Mat tmpi =I(Rect(p1,p2)).clone(),tmps =S(Rect(p1,p2)).clone();
            tmpi = tmpi.reshape(0,1);
            tmps = tmps.reshape(0,1);
            Mat sample = cv::Mat::zeros(2, 25, CV_32F);
            tmpi.row(0).copyTo(sample.row(0));
            tmps.row(0).copyTo(sample.row(1));
            cv::Mat_<float> covar, mean;
            cv::calcCovarMatrix( sample, covar, mean, cv::COVAR_NORMAL|cv::COVAR_COLS, CV_32FC1);
            Rho.at<float>(p,q) = covar.at<float>(0,1)/(25-1);
        }
    }
    p1=Point(2,2);
    p2=Point(2+wid,2+hei);
    Mat rho = Rho(Rect(p1,p2));
    ds.convertTo(ds, CV_32F);
    Mat rd = rho.mul(ds);
    rd.convertTo(rd, CV_32S);
    hist_i = cv::Mat::zeros(256,1,CV_32S);
    hist_s = cv::Mat::zeros(256,1,CV_32S);
    
    i.convertTo(i, CV_8U);
    for(int ir=0; ir<i.rows; ir++){
        for(int ic=0; ic<i.cols; ic++){
            int value = int(i.at<uchar>(ir,ic));
            if(value>=255){
                value = 254;
                       }
            hist_i.at<int>(value+1,0) += di.at<int>(ir,ic);
            hist_s.at<int>(value+1,0) += rd.at<int>(ir,ic);
        }
    }
    return;
}

void dhe(cv::Mat img, cv::Mat & result,float alpha){
    cv::Mat hist_i ,hist_s;
    build_is_hist(img,hist_i,hist_s);
    Mat hist_c = alpha * hist_s + (1-alpha) * hist_i;
    Scalar hist_sum_s = cv::sum(hist_c);
    double hist_sum = hist_sum_s[0];
    cv::Mat hist_cum = cv::Mat::zeros(hist_c.rows, hist_c.cols, CV_32F);
    
    hist_cum.at<float>(0,0) = hist_c.at<int>(0,0);
    for(int i=1;i<hist_cum.rows;i++){
        hist_cum.at<float>(i,0) = hist_c.at<int>(i,0)+hist_cum.at<float>(i-1,0);
    }
    
    Mat hsv,hsvs[3];
    cv::cvtColor(img, hsv, COLOR_RGB2HSV);
    split(hsv,hsvs);
    Mat h = hsvs[0], s = hsvs[1],i = hsvs[2];
    Mat c = hist_cum/hist_sum;
    Mat s_r = c* 255;
    Mat i_s = cv::Mat::zeros(i.rows,i.cols,CV_32F);
    
    for(int ir=0;ir<i_s.rows;ir++){
        for(int ic=0;ic<i_s.cols;ic++)
        {
            int value = int(i.at<uchar>(ir,ic));
            if(value == 255){
                i_s.at<float>(ir,ic) =1;
            }
            else{
                i_s.at<float>(ir,ic) = s_r.at<float>(value+1,0)/255.0;
            }
        }
    }
    Mat hsi_o;//, result;
    i_s = i_s *255;
    i_s.convertTo(i_s, CV_8U);
    cv::threshold(i_s, i_s, 0, 255, THRESH_TOZERO);
    Mat hsv_new[3] = {h,s,i_s};
    merge(hsv_new,3,hsi_o);
    cv::cvtColor(hsi_o, result, COLOR_HSV2BGR);
    return;
}



