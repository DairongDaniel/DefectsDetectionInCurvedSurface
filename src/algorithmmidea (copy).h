#ifndef ALGORITHMMIDEA_H
#define ALGORITHMMIDEA_H

#include <opencv2/opencv.hpp>
#include <QObject>

enum errorCode {
    correct = 0,
    blackDot = 1,
    logoLack = 2,
    logoVanish = 4
};

struct BlackDotData {
    cv::Point loc;
    float area;
};

struct ParamForBlackDot {
    int blockSize;
    double binThred;
    double dotAreaThred;
    int logoMaskWidth;
    int foregroundMaskWidth;

    std::vector<BlackDotData> blackDotRect;
};

struct ParamForTemplate {
    std::vector<cv::Rect> roiRectVec;
    std::vector<cv::Rect> undetectRoiRectVec;
    int trainRectExtendFringe;
};

struct FeatureOfTemplate {
    std::vector<cv::Rect> trainSubImageRectVec;
    std::vector<cv::Mat> querySubBinImageVec;
    std::vector<cv::Mat> querySubGrayImageVec;
    std::vector<cv::Mat> queryDescriptorsVec;
    std::vector<std::vector<cv::KeyPoint>> queryKeyPointsVec;
//    cv::Mat coefficientPolyA_h;
//    cv::Mat coefficientPolyA_v;
    cv::Mat preBaseImage;
    cv::Mat baseImage;
};

struct ParamForLogo {
    double ratioThredForTopTwo;
    double logoAreaThred;
};

class AlgorithmMidea : public QObject
{
    Q_OBJECT
public:
    explicit AlgorithmMidea(QObject *parent = nullptr);
    /******************************************
     * detect all defects of products---errorCode
     * logoVanish: there is not logos finded in detecting area.
     * logoLack: The lack of logo have been detected
     * blackDot: The black dots have been detected
     * correct: otherwise
    ******************************************/
    static int detectAllDefects(const cv::Mat &srcImage,cv::Mat &dstImage,ParamForBlackDot &paramSetsForBlackDot,FeatureOfTemplate &tempData,ParamForLogo &paramSetsForLogo);
    int detectAllDefects(const cv::Mat &srcImage,cv::Mat &dstImage);

    /******************************************
     * fuse images first then detect all defects of products.
     * It is similiar with the function of detectAllDefects except the procedure for fusing.
    ******************************************/
    int fuseAndDetectAllDefects(const cv::Mat &srcImage1, const cv::Mat &srcImage2, cv::Mat &dstImage);

    /******************************************
     * blackDot: The black dots have been detected
     * correct: otherwise
    ******************************************/
    static int detectBlackDot(const cv::Mat &srcImage,cv::Mat &dstImage,ParamForBlackDot &paramSets,cv::Mat &mask);

    /******************************************
     * extracting features from template logo croped by users
    ******************************************/
    static void extractingTemplateFeature(const cv::Mat &srcImage,cv::Mat &dstImage,ParamForTemplate &extrtParam, FeatureOfTemplate &tempData);
    void extractingTemplateFeature(const cv::Mat &srcImage,cv::Mat &dstImage);
    /******************************************
     * return encoded value
     * logoVanish: there is not logos finded in detecting area.
     * logoLack: The lack of logo have been detected
     * correct: perfect products
    ******************************************/
    static int detectLackofLogo(const cv::Mat &srcImage,cv::Mat &dstImage,FeatureOfTemplate &tempData,ParamForLogo &paramSets,cv::Mat &blackDotMask);

    /**************************************************************************
     * @brief remove outliers using the RANSAC(random sampling consensus) algorithm
     * @input
     * @param queryPoints
     * @param trainPoints
     * @param Matches opencv struct that it indicate the Matched keypoint pairs.
     * @output
     * @param outputMatches
    ******************************************************************************/
    static void removeOutlier( std::vector<cv::KeyPoint> &queryPoints, std::vector<cv::KeyPoint> &trainPoints, std::vector<cv::DMatch> &Matches, std::vector<cv::DMatch> &outputMatches);

    /******************************************
     * save the configure files
    ******************************************/
    void saveConfigureFile(QString fileName);
    /******************************************
     * load the configure files
    ******************************************/
    void loadConfigureFile(QString fileName);

    /**************************************************************************
     * binarization for HSV Image
    ******************************************************************************/
    static void binarizationForHSV( const cv::Mat &srcImage, cv::Mat &dstImage, bool bgWite);

    /**************************************************************************
     * fuse two images, then remove background
    ******************************************************************************/
    void fuseImagesAndRomoveBG( const cv::Mat &srcImage1, const cv::Mat &srcImage2, cv::Mat &dstImage);

    /**************************************************************************
     * remove the base-plane
    ******************************************************************************/
    void removeBasePlane( const cv::Mat &srcImage, cv::Mat &dstImage);


    /**************************************************************************
     * fit curve line
    ******************************************************************************/
    static void fitPoly(std::vector<cv::Point> key_point, int n, cv::Mat& A);

    /**************************************************************************
     * inspect the line of pen
    ******************************************************************************/
    static bool inspectLineofPen(const cv::Mat &srcImage1, const cv::Mat &srcImage2, cv::Mat &dstImage,bool is_up);

signals:

public slots:

public:
    ParamForBlackDot    blackDotSets;
    ParamForTemplate    templateSets;
    FeatureOfTemplate   templateFeature;
    ParamForLogo        logoSets;
};

#endif // ALGORITHMMIDEA_H
