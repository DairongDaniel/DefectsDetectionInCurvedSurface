#include "algorithmmidea.h"
#include <QTime>
#include <QDebug>
#include <QDir>

#define IMAGESTORE true
#define DEBUGMESSAGE false
#define FEATURETYPE cv::AKAZE::DESCRIPTOR_MLDB
#define ADAPTIVETHRESHOLD true

AlgorithmMidea::AlgorithmMidea(QObject *parent) : QObject(parent)
{
    blackDotSets.blockSize=41;
    blackDotSets.binThred=100;
    blackDotSets.dotAreaThred=0;
    blackDotSets.logoMaskWidth=30;
    blackDotSets.foregroundMaskWidth=20;

    templateSets.trainRectExtendFringe=100;

    logoSets.ratioThredForTopTwo=0.7;
    logoSets.logoAreaThred=0;
}

int AlgorithmMidea::detectAllDefects(const cv::Mat &srcImage, cv::Mat &dstImage, ParamForBlackDot &paramSetsForBlackDot, FeatureOfTemplate &tempData, ParamForLogo &paramSetsForLogo)
{
    cv::Mat blackDotMaskImage;
    int eCode=0;
    int tempECode=detectLackofLogo(srcImage, dstImage, tempData, paramSetsForLogo, blackDotMaskImage);
    eCode=eCode|tempECode;
    cv::Mat tempImage=dstImage.clone();
    tempECode=detectBlackDot(tempImage, dstImage, paramSetsForBlackDot,tempData,blackDotMaskImage);
    eCode=eCode|tempECode;
    return eCode;
}

int AlgorithmMidea::detectAllDefects(const cv::Mat &srcImage, cv::Mat &dstImage)
{
    cv::Mat removedBPImage;
    removeBasePlane(srcImage, removedBPImage);
    return detectAllDefects(
                removedBPImage,
                dstImage,
                blackDotSets,
                templateFeature,
                logoSets
                );
}

int AlgorithmMidea::fuseAndDetectAllDefects(const cv::Mat &srcImage1, const cv::Mat &srcImage2, cv::Mat &dstImage)
{
    cv::Mat fusedImage;
    fuseImagesAndRomoveBG(srcImage1, srcImage2, fusedImage);
    return detectAllDefects(
                fusedImage,
                dstImage
                );
}

int AlgorithmMidea::detectBlackDot(const cv::Mat &srcImage, cv::Mat &dstImage, ParamForBlackDot &paramSets, FeatureOfTemplate &templateParamSets, cv::Mat &mask)
{
    if (DEBUGMESSAGE) {
        qDebug()<<"MD Algorithm: "<<"function--detectBlackDot()";
    }
    paramSets.blackDotRect.clear();
    //preprocessing
    dstImage=srcImage.clone();
    cv::Mat grayImage;
    if (srcImage.channels()>1) {
        cv::cvtColor(srcImage,grayImage,CV_BGR2GRAY);
    } else {
        grayImage=srcImage.clone();
        cv::cvtColor(dstImage,dstImage,CV_GRAY2BGR);
    }
    //    GaussianBlur( grayImage, grayImage, cv::Size(3,3), 1);
    cv::Mat binImage;
    if (ADAPTIVETHRESHOLD) {
        //adaptive binary threshold
        cv::adaptiveThreshold(
                    grayImage,
                    binImage,
                    255,
                    cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv::THRESH_BINARY_INV,
                    paramSets.blockSize/2*2+1,
                    paramSets.binThred
                    );
    } else {
        cv::threshold(grayImage,binImage,paramSets.binThred,255,cv::THRESH_BINARY_INV);
    }
    if (IMAGESTORE) {
        cv::imwrite	(QDir::currentPath().toStdString()+"/"+ QString::number(QDateTime::currentMSecsSinceEpoch()).toStdString()+"_blackDot.jpg", binImage);

    }
    //remove the area of logo
    if (mask.data) {
        cv::threshold(mask,mask,1,255,cv::THRESH_BINARY);
        cv::Mat dilKel=cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size(paramSets.logoMaskWidth,paramSets.logoMaskWidth));
        cv::dilate(mask,mask,dilKel);
        cv::bitwise_not(mask,mask);
        cv::bitwise_and(binImage,mask,binImage);
    }
    //remove the area of background
    for (size_t i=0;i<templateParamSets.backgroundAreaVec.size();i++) {
        binImage(templateParamSets.backgroundAreaVec[i])=0;
    }
    //remove the area of undetected area
    for (size_t i=0;i<templateParamSets.undetectAreaVec.size();i++) {
        binImage(templateParamSets.undetectAreaVec[i])=0;
    }

    //searching the demanding dots over the whole binary image.
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(binImage,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE );
    for (size_t l=0;l<contours.size();l++){
        cv::Rect tempAreaRect = cv::boundingRect( contours[l] );
        double curArea=cv::countNonZero(binImage(tempAreaRect));
        if(curArea>paramSets.dotAreaThred) {
            BlackDotData blackD;
            blackD.area=curArea;
            cv::Rect dotRect=boundingRect(contours[l]);
            blackD.loc=cv::Point(dotRect.x+dotRect.width/2,dotRect.y+dotRect.height/2);
            cv::drawContours( dstImage, contours, l, cv::Scalar( 0,0,255 ), CV_FILLED );
            paramSets.blackDotRect.push_back(blackD);
        }
    }
    //outputing the result
    if (paramSets.blackDotRect.size()>0) {
        return blackDot;
    }
    return correct;
}

void AlgorithmMidea::extractingTemplateFeature(const cv::Mat &srcImage,cv::Mat &dstImage, ParamForTemplate &extrtParam, FeatureOfTemplate &tempData)
{
    if (DEBUGMESSAGE) {
        qDebug()<<"MD Algorithm: "<<"function--extractingTemplateFeature()";
    }
    //preprocessing
    tempData.querySubBinImageVec.clear();
    tempData.querySubGrayImageVec.clear();
    tempData.queryDescriptorsVec.clear();
    tempData.queryKeyPointsVec.clear();
    tempData.undetectAreaVec.clear();
    tempData.backgroundAreaVec.clear();
    dstImage=srcImage.clone();

    cv::Mat grayImage;
    if (srcImage.channels()>1) {
        cv::cvtColor(srcImage,grayImage,CV_BGR2GRAY);
    } else {
        grayImage=srcImage.clone();
        cv::cvtColor(dstImage,dstImage,CV_GRAY2BGR);
    }
    //    GaussianBlur( grayImage, grayImage, cv::Size(3,3), 1);

    //remove the background
    for (size_t i=0;i<extrtParam.backgroundRoiRectVec.size();i++) {
        grayImage(extrtParam.backgroundRoiRectVec[i])=255;
    }
    tempData.backgroundAreaVec=extrtParam.backgroundRoiRectVec;

    //calculate the coefficience of the fitpoly algorithm
    if (tempData.baseImage.data==NULL) {
        int x_loc=grayImage.cols-1 , y_loc=grayImage.rows-1;
        if (extrtParam.roiRectVec.size() || extrtParam.undetectRoiRectVec.size()) {
            for (size_t i=0;i<extrtParam.roiRectVec.size();i++) {
                cv::Point tlP=extrtParam.roiRectVec[i].tl();
                if (x_loc>tlP.x) {
                    x_loc=tlP.x;
                }
                if (y_loc>tlP.y) {
                    y_loc=tlP.y;
                }
            }
            for (size_t i=0;i<extrtParam.undetectRoiRectVec.size();i++) {
                cv::Point tlP=extrtParam.undetectRoiRectVec[i].tl();
                if (x_loc>tlP.x) {
                    x_loc=tlP.x;
                }
                if (y_loc>tlP.y) {
                    y_loc=tlP.y;
                }
            }
            if (x_loc<5 || y_loc<5) {
                for (size_t i=0;i<extrtParam.roiRectVec.size();i++) {
                    cv::Point btP=extrtParam.roiRectVec[i].br();
                    if (x_loc<btP.x) {
                        x_loc=btP.x;
                    }
                    if (y_loc<btP.y) {
                        y_loc=btP.y;
                    }
                }
                for (size_t i=0;i<extrtParam.undetectRoiRectVec.size();i++) {
                    cv::Point btP=extrtParam.undetectRoiRectVec[i].br();
                    if (x_loc<btP.x) {
                        x_loc=btP.x;
                    }
                    if (y_loc<btP.y) {
                        y_loc=btP.y;
                    }
                }
            }
        } else {
            x_loc/=2;
            y_loc/=2;
        }
        tempData.undetectAreaVec=extrtParam.undetectRoiRectVec;
        //horizontal direction (i,y_loc)
        //Function(at(i,j)):i-row,j-col, Function(Point(x,y)):x-col,y-row
        std::vector<cv::Point> points_h;
        for (int i=0;i<grayImage.cols;i++) {
            if (grayImage.at<uchar>(y_loc,i)!=255) {
                points_h.push_back(cv::Point(i,grayImage.at<uchar>(y_loc,i)));
            }
        }
        std::random_shuffle(points_h.begin(),points_h.end());
        cv:: Mat coefficientPolyA_h;
        int power_h=6;
        fitPoly(points_h, power_h, coefficientPolyA_h);
        //vertical direction (x_loc,i)
        std::vector<cv::Point> points_v;
        for (int i=0;i<grayImage.rows;i++) {
            if (grayImage.at<uchar>(i,x_loc)!=255) {
                points_v.push_back(cv::Point(i,grayImage.at<uchar>(i,x_loc)));
            }
        }
        std::random_shuffle(points_v.begin(),points_v.end());
        cv:: Mat coefficientPolyA_v;
        int power_v=6;
        fitPoly(points_v, power_v, coefficientPolyA_v);

        cv::Mat tempBaseImage=cv::Mat::zeros(grayImage.size(),CV_32FC1);

        for (int y = 0; y < tempBaseImage.rows; y++) {
            float tempV= coefficientPolyA_v.at<double>(0, 0);
            for (int pI=0;pI<power_h;pI++) {
                tempV += coefficientPolyA_v.at<double>(pI+1, 0)*std::pow(y, pI+1);
            }
            for (int x = 0;x < tempBaseImage.cols; x++) {
                float tempH= coefficientPolyA_h.at<double>(0, 0);
                for (int pI=0;pI<power_v;pI++) {
                    tempH += coefficientPolyA_h.at<double>(pI+1, 0)*std::pow(x, pI+1);
                }
                tempBaseImage.at<float>(y,x) = tempH+tempV;
            }
        }
        double minV,maxV;
        cv::Point minL,maxL;
        cv::minMaxLoc(tempBaseImage, &minV, &maxV,&minL,&maxL);
        tempData.preBaseImage=cv::Mat::zeros(grayImage.size(),CV_32FC1);
        tempData.baseImage=cv::Mat::zeros(grayImage.size(),CV_8UC1);
        for (int y = 0; y < tempBaseImage.rows; y++) {
            for (int x = 0;x < tempBaseImage.cols; x++) {
                tempData.baseImage.at<uchar>(y,x) = maxV - tempBaseImage.at<float>(y,x);
                tempData.preBaseImage.at<float>(y,x) = tempBaseImage.at<float>(y,x)-maxV/2;
            }
        }
    }

    cv::Mat rmBPImage;
    removeBasePlane(grayImage, rmBPImage,tempData);

    //extracting features
    for (size_t i=0;i<extrtParam.roiRectVec.size();i++) {
        cv::Mat roiImage=rmBPImage(extrtParam.roiRectVec[i]);
        tempData.querySubGrayImageVec.push_back(roiImage.clone());
        cv::threshold(roiImage,roiImage,255,255,cv::THRESH_OTSU|cv::THRESH_BINARY_INV);
        //        inRange	( m_testImage, 80, 150, binImag );
        tempData.querySubBinImageVec.push_back(roiImage.clone());
        //used ORB operator
        cv::Ptr<cv::AKAZE> orb =cv::AKAZE::create(FEATURETYPE,
                                                  0,
                                                  3,
                                                  0.01f,
                                                  4,
                                                  4,
                                                  cv::KAZE::DIFF_PM_G2);
        //Ptr<ORB> orb =ORB::create(150,1.2f,8,101,0,2,ORB::HARRIS_SCORE,101,20);
        //computing keypoints and descriptors
        cv::Mat queryDescriptors;
        std::vector<cv::KeyPoint> queryKeyPoints;
        orb->detectAndCompute(roiImage, cv::Mat(),queryKeyPoints,queryDescriptors);
        tempData.queryDescriptorsVec.push_back(queryDescriptors);
        tempData.queryKeyPointsVec.push_back(queryKeyPoints);
        //drawing
        cv::Point ltP,rbp;
        ltP = cv::Point(extrtParam.roiRectVec[i].x,extrtParam.roiRectVec[i].y);
        rbp = cv::Point(extrtParam.roiRectVec[i].x+extrtParam.roiRectVec[i].width,
                        extrtParam.roiRectVec[i].y+extrtParam.roiRectVec[i].height);
        ltP = ltP-cv::Point(extrtParam.trainRectExtendFringe,3*extrtParam.trainRectExtendFringe);
        rbp = rbp+cv::Point(extrtParam.trainRectExtendFringe,3*extrtParam.trainRectExtendFringe);
        if (ltP.x<0) {
            ltP.x=0;
        }
        if (ltP.y<0) {
            ltP.y=0;
        }
        if (rbp.x>rmBPImage.cols-1) {
            rbp.x=rmBPImage.cols-1;
        }
        if (rbp.y>rmBPImage.rows-1) {
            rbp.y=rmBPImage.rows-1;
        }
        tempData.trainSubImageRectVec.push_back(cv::Rect(ltP,rbp));
        cv::Mat markImage=dstImage(extrtParam.roiRectVec[i]);
        cv::drawKeypoints(markImage,
                          queryKeyPoints,
                          markImage,
                          cv::Scalar(0,0,255),
                          cv::DrawMatchesFlags::DRAW_OVER_OUTIMG
                          );
    }
}

void AlgorithmMidea::extractingTemplateFeature(const cv::Mat &srcImage, cv::Mat &dstImage)
{
    extractingTemplateFeature(srcImage,dstImage, templateSets, templateFeature);
}

int AlgorithmMidea::detectLackofLogo(const cv::Mat &srcImage, cv::Mat &dstImage, FeatureOfTemplate &tempData, ParamForLogo &paramSets, cv::Mat &blackDotMask)
{
    if (DEBUGMESSAGE) {
        qDebug()<<"MD Algorithm: "<<"function--detectLackofLogo()";
    }
    //preprocessing
    dstImage=srcImage.clone();
    blackDotMask=cv::Mat::zeros(srcImage.size(),CV_8UC1);
    cv::Mat grayImage;
    if (srcImage.channels()>1) {
        cv::cvtColor(srcImage,grayImage,CV_BGR2GRAY);
    } else {
        grayImage=srcImage.clone();
        cv::cvtColor(dstImage,dstImage,CV_GRAY2BGR);
    }
    //    GaussianBlur( grayImage, grayImage, cv::Size(7,7), 1);
    std::vector<std::vector<std::vector<cv::Point>>> contoursVec;
    std::vector<std::vector<int >> leftIndexVec;
    for (size_t i=0;i<tempData.trainSubImageRectVec.size();i++) {
        QTime timer;
        double preTime=0;
        if (DEBUGMESSAGE) {
            timer.start();
        }
        cv::Mat roiImage=grayImage(tempData.trainSubImageRectVec[i]);
        cv::threshold(roiImage,roiImage,255,255,cv::THRESH_OTSU|cv::THRESH_BINARY_INV);
        //        inRange	( m_testImage, 80, 150, binImag );
        //using ORB operator
        cv::Ptr<cv::AKAZE> orb =cv::AKAZE::create(FEATURETYPE,
                                                  0,
                                                  3,
                                                  0.01f,
                                                  4,
                                                  4,
                                                  cv::KAZE::DIFF_PM_G2 /*150*//*,1.f,1,31,0,2,ORB::HARRIS_SCORE,31,20*/);
        //        Ptr<ORB> orb =ORB::create(150,1.2f,8,101,0,2,ORB::HARRIS_SCORE,101,20);
        cv::BFMatcher matcher(cv::NORM_HAMMING/*,true*/);
        //compute keypoints and descriptors
        cv::Mat trainDescriptors;
        std::vector<cv::KeyPoint> trainKeyPoints;
        //        orb->detectAndCompute(grayImage, maskImage,trainKeyPoints,trainDescriptors);
        orb->detectAndCompute(roiImage, cv::Mat(),trainKeyPoints,trainDescriptors);
        if (DEBUGMESSAGE) {
            qDebug()<<"MD Algorithm: "<<i<<"th detectAndCompute--time ="<<timer.elapsed()-preTime<<"ms";
            qDebug()<<"MD Algorithm: "<<i<<"th detectAndCompute--key points number ="<<trainKeyPoints.size();
            preTime=timer.elapsed();
        }
        if (trainKeyPoints.size()<9) {
            continue;
        }
        std::vector<std::vector<cv::DMatch>> matchesVec;
        matcher.knnMatch(tempData.queryDescriptorsVec[i], trainDescriptors, matchesVec,2);
        if (DEBUGMESSAGE) {
            qDebug()<<"MD Algorithm: "<<i<<"th knnMatch--time ="<<timer.elapsed()-preTime<<"ms";
            preTime=timer.elapsed();
            qDebug()<<"MD Algorithm: "<<i<<"th original matches number="<<matchesVec.size();
        }
        if (matchesVec.size()<50) {
            return logoVanish;
        }
        std::vector<std::vector<cv::DMatch>>::iterator it;
        std::vector<cv::DMatch> matches;
        for (it = matchesVec.begin();it!=matchesVec.end();++it) {
            if ((*it)[0].distance/(*it)[1].distance<paramSets.ratioThredForTopTwo) {
                matches.push_back((*it)[0]);
            }
        }
        if (DEBUGMESSAGE) {
            qDebug()<<"MD Algorithm: "<<i<<"th matches number after filtering by ratio("<<paramSets.ratioThredForTopTwo<<") ="<<matches.size();
        }
        if (matches.size()<50) {
            return logoVanish;
        }
        //remove outliers using the RANSAC algorithm
        std::vector<cv::DMatch> bestMatches;
        removeOutlier(tempData.queryKeyPointsVec[i],trainKeyPoints,matches,bestMatches);
        if (DEBUGMESSAGE) {
            qDebug()<<"MD Algorithm: "<<i<<"th removeOutlier--time ="<<timer.elapsed()-preTime<<"ms";
            preTime=timer.elapsed();
            qDebug()<<"MD Algorithm: "<<i<<"th matches number after filtering by RANSAC"<<bestMatches.size();
        }
        if (bestMatches.size()<50) {
            return logoVanish;
        }
        if (IMAGESTORE) {
            cv::Mat matchedImage;
            cv::drawMatches( tempData.querySubBinImageVec[i],
                             tempData.queryKeyPointsVec[i],
                             roiImage,
                             trainKeyPoints,
                             bestMatches,
                             matchedImage
                             );
            cv::imwrite	(QDir::currentPath().toStdString()+"/"+ QString::number(QDateTime::currentMSecsSinceEpoch()).toStdString()+"_matchedKeyMap-"+QString::number(i).toStdString()+".jpg", matchedImage);
        }
        std::vector<cv::Point2f> matchedQueryPoints,matchedTrainPoints;
        cv::Point2f offsetLTP=cv::Point2f(tempData.trainSubImageRectVec[i].tl());
        for (std::vector<cv::DMatch>::const_iterator it=bestMatches.begin();it!=bestMatches.end();++it) {
            matchedQueryPoints.push_back(tempData.queryKeyPointsVec[i][it->queryIdx].pt);
            matchedTrainPoints.push_back(trainKeyPoints[it->trainIdx].pt+offsetLTP);
        }
        cv::Mat homographyMatrix=cv::findHomography(matchedQueryPoints,matchedTrainPoints,cv::RANSAC);
        cv::Mat transformedBinImage;
        cv::warpPerspective(tempData.querySubBinImageVec[i],transformedBinImage,homographyMatrix,grayImage.size());
        cv::bitwise_or(blackDotMask,transformedBinImage,blackDotMask);
        cv::threshold(transformedBinImage,transformedBinImage,100,255,cv::THRESH_BINARY);
        cv::bitwise_xor(grayImage,transformedBinImage,grayImage);

        //filter the interruption of ringe
        cv::Mat transformedRoiImage=transformedBinImage(tempData.trainSubImageRectVec[i]);
        cv::Mat kel=cv::getStructuringElement( cv::MORPH_RECT, cv::Size(7,7));
        cv::Mat bigImage;
        cv::dilate(transformedRoiImage,bigImage,kel);
        cv::Mat smallImage;
        cv::erode (transformedRoiImage,smallImage,kel);
        cv::Mat ringeImage;
        cv::bitwise_xor(smallImage,bigImage,ringeImage);
        cv::bitwise_not(ringeImage,ringeImage);
        cv::bitwise_and(roiImage,ringeImage,roiImage);

        if (DEBUGMESSAGE) {
            qDebug()<<"MD Algorithm: "<<i<<"th homographyMatrix--time ="<<timer.elapsed()-preTime<<"ms";
            preTime=timer.elapsed();
        }

        //searching which area is missed.
        std::vector<std::vector<cv::Point> > contours;
        std::vector<int > leftIndex;
        cv::findContours(roiImage,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE );
        cv::Mat markImage=dstImage(tempData.trainSubImageRectVec[i]);
        for (size_t l=0;l<contours.size();l++){
            double curArea=cv::contourArea(contours[l]);
            if(curArea>paramSets.logoAreaThred) {
                cv::drawContours( markImage, contours, l, cv::Scalar( 0,255,0 ), CV_FILLED );
                leftIndex.push_back(l);
            }
        }
        contoursVec.push_back(contours);
        leftIndexVec.push_back(leftIndex);
    }
    if (IMAGESTORE) {
        cv::imwrite	(QDir::currentPath().toStdString()+"/"+ QString::number(QDateTime::currentMSecsSinceEpoch()).toStdString() +"_feature.jpg", grayImage);
        cv::imwrite	(QDir::currentPath().toStdString()+"/"+ QString::number(QDateTime::currentMSecsSinceEpoch()).toStdString() +"_logoMask.jpg", blackDotMask);
    }
    for (size_t i=0;i<leftIndexVec.size();i++) {
        if (leftIndexVec[i].size()>0) {
            return logoLack;
        }
    }
    return correct;
}

void AlgorithmMidea::removeOutlier(std::vector<cv::KeyPoint> &queryPoints, std::vector<cv::KeyPoint> &trainPoints, std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &outputMatches)
{
    std::vector<cv::Point2f> matchedQueryPoints,matchedTrainPoints;
    for (std::vector<cv::DMatch>::const_iterator it=matches.begin();it!=matches.end();++it) {
        matchedQueryPoints.push_back(queryPoints[it->queryIdx].pt);
        matchedTrainPoints.push_back(trainPoints[it->trainIdx].pt);
    }
    std::vector<uchar> inliers;
    cv::findFundamentalMat(matchedQueryPoints,matchedTrainPoints,inliers,CV_FM_RANSAC);
    //extract the surviving (inliers) matches
    std::vector<uchar>::const_iterator itInlier=inliers.begin();
    std::vector<cv::DMatch>::const_iterator itMatches=matches.begin();
    for (;itInlier!=inliers.end();++itInlier,++itMatches)
        if(*itInlier)
            outputMatches.push_back(*itMatches);
}

void AlgorithmMidea::saveConfigureFile(QString fileName)
{
    if (fileName==NULL) {
        return;
    }
    cv::FileStorage fs(fileName.toStdString(), cv::FileStorage::WRITE);
    time_t rawtime; time(&rawtime);

    fs << "ParamForBlackDot"<<"[";
    fs  << "{:"
        << "blockSize" << blackDotSets.blockSize
        << "binThred" << blackDotSets.binThred
        << "dotAreaThred" << blackDotSets.dotAreaThred
        << "logoMaskWidth" << blackDotSets.logoMaskWidth
        << "foregroundMaskWidth" << blackDotSets.foregroundMaskWidth
        << "}";
    fs << "]";

    fs << "ParamForTemplate"<<"[";
    fs  << "{:"
        << "trainRectExtendFringe" << templateSets.trainRectExtendFringe
        << "}";
    fs << "]";

    fs << "FeatureOfTemplate"<<"[";
    fs  << "{:"
        << "trainSubImageRectVec" << templateFeature.trainSubImageRectVec
        << "undetectAreaVec" << templateFeature.undetectAreaVec
        << "backgroundAreaVec" << templateFeature.backgroundAreaVec
        << "querySubBinImageVec" << templateFeature.querySubBinImageVec
        << "querySubGrayImageVec" << templateFeature.querySubGrayImageVec
        << "queryDescriptorsVec" << templateFeature.queryDescriptorsVec
        << "queryKeyPointsVec" << templateFeature.queryKeyPointsVec
           //        << "coefficientPolyA_h" << templateFeature.coefficientPolyA_h
           //        << "coefficientPolyA_v" << templateFeature.coefficientPolyA_v
        << "baseImage" << templateFeature.baseImage
        << "preBaseImage" << templateFeature.preBaseImage
        << "}";
    fs << "]";

    fs << "ParamForLogo"<<"[";
    fs  << "{:"
        << "ratioThredForTopTwo" << logoSets.ratioThredForTopTwo
        << "logoAreaThred" << logoSets.logoAreaThred
        << "}";
    fs << "]";

    fs.release();
}

void AlgorithmMidea::loadConfigureFile(QString fileName)
{
    if (fileName==NULL) {
        return;
    }
    cv::FileStorage fs2(fileName.toStdString(), cv::FileStorage::READ);

    cv::FileNode ParamForBlackDot = fs2["ParamForBlackDot"];
    cv::FileNodeIterator it = ParamForBlackDot.begin(), it_end = ParamForBlackDot.end();
    // iterate through a sequence using FileNodeIterator
    for( ; it != it_end; ++it )
    {
        (*it)["blockSize"] >> blackDotSets.blockSize;
        (*it)["binThred"] >> blackDotSets.binThred;
        (*it)["dotAreaThred"] >> blackDotSets.dotAreaThred;
        (*it)["logoMaskWidth"] >> blackDotSets.logoMaskWidth;
        (*it)["foregroundMaskWidth"] >> blackDotSets.foregroundMaskWidth;
    }

    cv::FileNode ParamForTemplate = fs2["ParamForTemplate"];
    it = ParamForTemplate.begin(), it_end = ParamForTemplate.end();
    // iterate through a sequence using FileNodeIterator
    for( ; it != it_end; ++it )
    {
        (*it)["trainRectExtendFringe"] >> templateSets.trainRectExtendFringe;
    }

    cv::FileNode FeatureOfTemplate = fs2["FeatureOfTemplate"];
    it = FeatureOfTemplate.begin(), it_end = FeatureOfTemplate.end();
    // iterate through a sequence using FileNodeIterator
    for( ; it != it_end; ++it )
    {
        (*it)["trainSubImageRectVec"] >> templateFeature.trainSubImageRectVec;
        (*it)["undetectAreaVec"] >> templateFeature.undetectAreaVec;
        (*it)["backgroundAreaVec"] >> templateFeature.backgroundAreaVec;
        (*it)["querySubBinImageVec"] >> templateFeature.querySubBinImageVec;
        (*it)["querySubGrayImageVec"] >> templateFeature.querySubGrayImageVec;
        (*it)["queryDescriptorsVec"] >> templateFeature.queryDescriptorsVec;
        (*it)["queryKeyPointsVec"] >> templateFeature.queryKeyPointsVec;
        //        (*it)["coefficientPolyA_h"] >> templateFeature.coefficientPolyA_h;
        //        (*it)["coefficientPolyA_v"] >> templateFeature.coefficientPolyA_v;
        (*it)["baseImage"] >> templateFeature.baseImage;
        (*it)["preBaseImage"] >> templateFeature.preBaseImage;
    }

    cv::FileNode ParamForLogo = fs2["ParamForLogo"];
    it = ParamForLogo.begin(), it_end = ParamForLogo.end();
    // iterate through a sequence using FileNodeIterator
    for( ; it != it_end; ++it )
    {
        (*it)["ratioThredForTopTwo"] >> logoSets.ratioThredForTopTwo;
        (*it)["logoAreaThred"] >> logoSets.logoAreaThred;
    }
    fs2.release();
}

void AlgorithmMidea::binarizationForHSV(const cv::Mat &srcImage, cv::Mat &dstImage, bool bgWite)
{
    CV_Assert(srcImage.channels()==3);
    cv::Mat GaussImage;
    cv::GaussianBlur( srcImage, GaussImage,cv::Size(7,7), 1);
    //    inRange	( GaussImage, 50, 200, dstImage );
    cv::Mat Gray_image;
    cv::cvtColor(GaussImage,Gray_image,CV_BGR2GRAY);
    cv::Mat binImageBG;
    cv::threshold(Gray_image, binImageBG, 50, 255,cv::THRESH_BINARY);
    cv::Mat colorBGImage=cv::Mat::ones(binImageBG.size(),CV_8UC3)*255;
    GaussImage.copyTo(colorBGImage,binImageBG);
    cv::Mat HSV_image;
    cv::cvtColor(colorBGImage,HSV_image,CV_BGR2HSV);
    std::vector<cv::Mat> HSV_channelVec;
    cv::split(HSV_image,HSV_channelVec);
    cv::Mat binImageLogo;
    cv::threshold(HSV_channelVec[1], binImageLogo, 125, 255,cv::THRESH_BINARY);
    cv::bitwise_not(binImageLogo,binImageBG);
    cv::Mat S_imageBG=cv::Mat::ones(binImageBG.size(),CV_8UC1)*255;
    HSV_channelVec[1].copyTo(S_imageBG,binImageBG);
    cv::threshold(S_imageBG, dstImage, 25, 255,cv::THRESH_BINARY_INV);

    //    cv::Mat kel=getStructuringElement( cv::MORPH_ELLIPSE, cv::Size(55,55));
    //    dilate(dstImage,dstImage,kel);

    //    cv::Mat resultImage=cv::Mat::zeros(srcImage.size(),CV_32SC1);

    //    watershed(sourcImage,resultImage);

    //    cv::Mat HSV_image;
    //    cvtColor(GaussImage,HSV_image,CV_BGR2HSV);
    //    std::vector<cv::Mat> HSV_channelVec;
    //    split(HSV_image,HSV_channelVec);
    //    //binarize image using S channel
    //    dstImage=HSV_channelVec[1].clone();
    //    threshold(HSV_channelVec[1], dstImage, 26, 255,cv::THRESH_BINARY_INV);
    //    //using V channel again
    //    cv::Mat maskedChannelV=cv::Mat::zeros(binImageBG.size(),CV_8UC1);
    //    HSV_channelVec[2].copyTo(maskedChannelV,binImageBG);
    //    if (bgWite){
    //        threshold(maskedChannelV, dstImage, 30, 255,cv::THRESH_BINARY_INV);
    //    } else {
    //        threshold(maskedChannelV, dstImage, 30, 255,cv::THRESH_BINARY);
    //    }
}

void AlgorithmMidea::fuseImagesAndRomoveBG(const cv::Mat &srcImage1, const cv::Mat &srcImage2, cv::Mat &dstImage)
{
    QTime timer;
    double preTime=0;
    if (DEBUGMESSAGE) {
        timer.start();
    }
    cv::Mat grayImage1, grayImage2;
    if (srcImage1.channels()>1) {
        cv::cvtColor(srcImage1,grayImage1,CV_BGR2GRAY);
        cv::cvtColor(srcImage2,grayImage2,CV_BGR2GRAY);
    } else {
        grayImage1=srcImage1.clone();
        grayImage2=srcImage2.clone();
    }
    dstImage=grayImage1.clone();
    cv::MatIterator_<uchar> it = dstImage.begin<uchar>();	//the start position of iterator
    cv::MatIterator_<uchar> itend = dstImage.end<uchar>();	//the end position of iterator
    cv::MatConstIterator_<uchar> it1 = grayImage1.begin<uchar>();
    cv::MatConstIterator_<uchar> it2 = grayImage2.begin<uchar>();
    //iterating
    for (; it != itend; it++) {
        if ( *it1 > *it2 ) {
            *it=*it2;
        }
        it1++;
        it2++;
    }
    if (IMAGESTORE) {
        cv::imwrite	(QDir::currentPath().toStdString()+"/"+ QString::number(QDateTime::currentMSecsSinceEpoch()).toStdString() +"_fusedSrc.jpg", dstImage);
    }
    if (DEBUGMESSAGE) {
        qDebug()<<"MD Algorithm: "<<"fusing stage--time ="<<timer.elapsed()-preTime<<"ms";
        preTime=timer.elapsed();
    }

    int bgThred=0;
    //remove left BG
    cv::Mat leftROI=dstImage(cv::Rect(0,dstImage.rows/2-100,200,200));
    cv::Mat tempBinImage;
    double thredValue = threshold(leftROI,tempBinImage,255,255,cv::THRESH_OTSU|cv::THRESH_BINARY_INV);
    if (DEBUGMESSAGE){
        qDebug()<<"thredValue left--"<<thredValue;
    }
    if (thredValue<bgThred) {
        std::vector<std::vector<cv::Point> > maxContours;
        cv::findContours(tempBinImage,maxContours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
        if (maxContours.size()) {
            int maxIndex=0;
            double maxArea=0;
            for (size_t l=0;l<maxContours.size();l++){
                double curArea=contourArea(maxContours[l]);
                if(curArea>maxArea) {
                    maxArea=curArea;
                    maxIndex=l;
                }
            }
            cv::Rect leftBGRect=boundingRect(maxContours[maxIndex]);
            cv::rectangle(dstImage, cv::Point(0,0),cv::Point(leftBGRect.x+leftBGRect.width+blackDotSets.foregroundMaskWidth,dstImage.rows), cv::Scalar::all(255), CV_FILLED);
        }
    }
    //remove right BG
    cv::Mat rightROI=dstImage(cv::Rect(dstImage.cols-200,dstImage.rows/2-100,200,200));
    thredValue = threshold(rightROI,tempBinImage,255,255,cv::THRESH_OTSU|cv::THRESH_BINARY_INV);
    if (DEBUGMESSAGE){
        qDebug()<<"thredValue right--"<<thredValue;
    }
    if (thredValue<bgThred) {
        std::vector<std::vector<cv::Point> > maxContours;
        cv::findContours(tempBinImage,maxContours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
        if (maxContours.size()) {
            int maxIndex=0;
            double maxArea=0;
            for (size_t l=0;l<maxContours.size();l++){
                double curArea=contourArea(maxContours[l]);
                if(curArea>maxArea) {
                    maxArea=curArea;
                    maxIndex=l;
                }
            }
            cv::Rect leftBGRect=boundingRect(maxContours[maxIndex]);
            rectangle(dstImage, cv::Point(dstImage.cols-200+leftBGRect.x-blackDotSets.foregroundMaskWidth,0),cv::Point(dstImage.cols,dstImage.rows), cv::Scalar::all(255), CV_FILLED);
        }
    }
    //remove up BG
    cv::Mat upROI=dstImage(cv::Rect(dstImage.cols/2-100,0,200,200));
    thredValue = threshold(upROI,tempBinImage,255,255,cv::THRESH_OTSU|cv::THRESH_BINARY_INV);
    if (DEBUGMESSAGE){
        qDebug()<<"thredValue up--"<<thredValue;
    }
    if (thredValue<bgThred) {
        std::vector<std::vector<cv::Point> > maxContours;
        findContours(tempBinImage,maxContours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
        if (maxContours.size()) {
            int maxIndex=0;
            double maxArea=0;
            for (size_t l=0;l<maxContours.size();l++){
                double curArea=contourArea(maxContours[l]);
                if(curArea>maxArea) {
                    maxArea=curArea;
                    maxIndex=l;
                }
            }
            cv::Rect leftBGRect=boundingRect(maxContours[maxIndex]);
            cv::rectangle(dstImage, cv::Point(0,0),cv::Point(dstImage.cols,leftBGRect.y+leftBGRect.height+blackDotSets.foregroundMaskWidth), cv::Scalar::all(255), CV_FILLED);
        }
    }
    //remove down BG
    cv::Mat downROI=dstImage(cv::Rect(dstImage.cols/2-100,dstImage.rows-200,200,200));
    thredValue = threshold(downROI,tempBinImage,255,255,cv::THRESH_OTSU|cv::THRESH_BINARY_INV);
    if (DEBUGMESSAGE){
        qDebug()<<"thredValue down--"<<thredValue;
    }
    if (thredValue<bgThred) {
        std::vector<std::vector<cv::Point> > maxContours;
        cv::findContours(tempBinImage,maxContours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
        if (maxContours.size()) {
            int maxIndex=0;
            double maxArea=0;
            for (size_t l=0;l<maxContours.size();l++){
                double curArea=contourArea(maxContours[l]);
                if(curArea>maxArea) {
                    maxArea=curArea;
                    maxIndex=l;
                }
            }
            cv::Rect leftBGRect=boundingRect(maxContours[maxIndex]);
            cv::rectangle(dstImage, cv::Point(0,leftBGRect.x-blackDotSets.foregroundMaskWidth+dstImage.rows-200),cv::Point(dstImage.cols,dstImage.rows), cv::Scalar::all(255), CV_FILLED);
        }
    }
    if (IMAGESTORE) {
        cv::imwrite	(QDir::currentPath().toStdString()+"/"+ QString::number(QDateTime::currentMSecsSinceEpoch()).toStdString() +"_fusedSrc_noBG.jpg", dstImage);
    }
    if (DEBUGMESSAGE){
        qDebug()<<"MD Algorithm: "<<"remove BG stage--time ="<<timer.elapsed()-preTime<<"ms";
        preTime=timer.elapsed();
    }
}

void AlgorithmMidea::removeBasePlane(const cv::Mat &srcImage, cv::Mat &dstImage)
{
    removeBasePlane(srcImage, dstImage, templateFeature);
}

void AlgorithmMidea::removeBasePlane(const cv::Mat &srcImage, cv::Mat &dstImage, FeatureOfTemplate &featureTemplate)
{
    QTime timer;
    double preTime=0;
    if (DEBUGMESSAGE) {
        timer.start();
    }
    if (srcImage.channels()>1) {
        cv::cvtColor(srcImage,dstImage,CV_BGR2GRAY);
    } else {
        dstImage=srcImage.clone();
    }
    //remove the uneven of light
    if (featureTemplate.baseImage.data!=NULL) {
        if (IMAGESTORE) {
            cv::imwrite    (QDir::currentPath().toStdString()+"/"+ QString::number(QDateTime::currentMSecsSinceEpoch()).toStdString() +"_Base.jpg", featureTemplate.baseImage);
            cv::imwrite    (QDir::currentPath().toStdString()+"/"+ QString::number(QDateTime::currentMSecsSinceEpoch()).toStdString() +"_BG.jpg", featureTemplate.preBaseImage);
        }
        CV_Assert(featureTemplate.preBaseImage.size()==dstImage.size());

        cv::MatIterator_<float> it = featureTemplate.preBaseImage.begin<float>();	//the start position of iterator
        cv::MatIterator_<float> itend = featureTemplate.preBaseImage.end<float>();	//the end position of iterator
        cv::MatIterator_<uchar> it_dst = dstImage.begin<uchar>();	//the start position of iterator
        cv::MatIterator_<uchar> it_base = featureTemplate.baseImage.begin<uchar>();	//the start position of iterator
        //iterating
        for (; it != itend; it++) {
            if ( *it ) {
                float diff=float(*it_dst)-*it;
                if (diff>0) {
                    *it_dst+=*it_base;
                } else {
                    float scale=float(*it_dst)/(*it);
                    *it_dst+=(uchar)(scale*float(*it_base));
                }
            }else {
                *it_dst+=*it_base;
            }
            it_dst++;
            it_base++;
        }
        if (IMAGESTORE) {
            //            cv::imwrite    (QDir::currentPath().toStdString()+"/"+ QString::number(QDateTime::currentMSecsSinceEpoch()).toStdString() +"_scaleImage.jpg", scaleImage);
            cv::imwrite    (QDir::currentPath().toStdString()+"/"+ QString::number(QDateTime::currentMSecsSinceEpoch()).toStdString()+"_fusedSrc_noBG_noUneven.jpg", dstImage);

        }
        if (DEBUGMESSAGE) {
            qDebug()<<"MD Algorithm: "<<"remove base-plane stage--time ="<<timer.elapsed()-preTime<<"ms";
            preTime=timer.elapsed();
        }
    }
}

void AlgorithmMidea::fitPoly(std::vector<cv::Point> key_point, int n, cv::Mat& A)
{
    //Number of key points
    int N = key_point.size();

    //construct X
    cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
    for (int i = 0; i < n + 1; i++) {
        for (int j = 0; j < n + 1; j++) {
            for (int k = 0; k < N; k++) {
                X.at<double>(i, j) = X.at<double>(i, j) +
                        std::pow(key_point[k].x, i + j);
            }
        }
    }

    //construct Y
    cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
    for (int i = 0; i < n + 1; i++) {
        for (int k = 0; k < N; k++) {
            Y.at<double>(i, 0) = Y.at<double>(i, 0) +
                    std::pow(key_point[k].x, i) * key_point[k].y;
        }
    }

    A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
    //solve A
    cv::solve(X, Y, A, cv::DECOMP_LU);
}

bool AlgorithmMidea::inspectLineofPen(const cv::Mat &srcImage1, const cv::Mat &srcImage2, const cv::Mat &srcImage3, const cv::Mat &srcImage4, cv::Mat &dstImage)
{
    int bin_thred=50;
    int area_thred=50;
    cv::Mat grayImage1, grayImage2;
    fuseImagesAndRomoveBG(srcImage1,srcImage2,grayImage1);
    fuseImagesAndRomoveBG(srcImage3,srcImage4,grayImage2);
    cv::Mat subtractedImage;
    cv::subtract( grayImage1,
                  grayImage2,
                  subtractedImage
                  );
    cv::threshold(subtractedImage,dstImage,bin_thred,255,cv::THRESH_BINARY);
    cv::erode(dstImage,dstImage,cv::Mat());
    for (size_t i=0;i<templateFeature.undetectAreaVec.size();i++) {
        dstImage(templateFeature.undetectAreaVec[i])=0;
    }
    for (size_t i=0;i<templateFeature.backgroundAreaVec.size();i++) {
        dstImage(templateFeature.backgroundAreaVec[i])=0;
    }
    //    qDebug()<<"templateSets.undetectRoiRectVec.size() = "<<templateSets.undetectRoiRectVec.size();
    cv::Scalar sum_bin = sum(dstImage)/255;
    qDebug()<<sum_bin(0);
    if (sum_bin(0)>area_thred) {
        return true;
    }
    return false;
}

bool AlgorithmMidea::inspectLineofPen(const cv::Mat &srcImage1, const cv::Mat &srcImage2, cv::Mat &dstImage, bool is_up)
{
    int bin_thred=50;
    int area_thred=50;
    cv::Mat grayImage1, grayImage2;
    cv::Rect roi;
    if (is_up) {
        roi=cv::Rect(0,0,2000,srcImage1.rows-500);
    }else {
        roi=cv::Rect(0,500,2000,srcImage1.rows-500);
    }
    std::cout<<roi;
    if (srcImage1.channels()>1) {
        cv::cvtColor(srcImage1(roi),grayImage1,CV_BGR2GRAY);
        cv::cvtColor(srcImage2(roi),grayImage2,CV_BGR2GRAY);
    } else {
        grayImage1=srcImage1(roi).clone();
        grayImage2=srcImage2(roi).clone();
    }
    cv::Mat subtractedImage;
    cv::subtract( grayImage1,
                  grayImage2,
                  subtractedImage
                  );
    cv::threshold(subtractedImage,dstImage,bin_thred,255,cv::THRESH_BINARY);
    cv::erode(dstImage,dstImage,cv::Mat());
    cv::Scalar sum_bin = sum(dstImage)/255;
    qDebug()<<sum_bin(0);
    //    int maxIndex;
    //    double maxArea=0;
    //    std::vector<std::vector<cv::Point> > maxContours;
    //    findContours(binImage,maxContours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
    //    for (size_t l=0;l<maxContours.size();l++){
    //        double curArea=contourArea(maxContours[l]);
    //        if(curArea>maxArea) {
    //            maxArea=curArea;
    //            maxIndex=l;
    //        }
    //    }
    if (sum_bin(0)>area_thred) {
        return true;
    }
    return false;
}
