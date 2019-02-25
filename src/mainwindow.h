#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QSettings>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <opencv2/opencv.hpp>
#include "algorithmmidea.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void initData();
    void updateAllWidgets();
    void removeOutlier(
            std::vector<cv::KeyPoint> &queryPoints,
            std::vector<cv::KeyPoint> &trainPoints,
            std::vector<cv::DMatch> &matches,
            std::vector<cv::DMatch> &outputMatches);

private slots:
    void on_pushButton_clicked();
    void on_text_edited(const QString &);

    void on_processBTN_clicked();

    //getting interesting image vector
    void on_getInterestingCropImgaes(std::vector<cv::Rect> );

    void on_setQueryTemplate_clicked();

    void on_trainData_clicked();

    void on_storeParameter_clicked();

    void on_loadParameter_clicked();

    void on_pushButton_2_clicked();

    void on_process2Picture_clicked();

private:
    Ui::MainWindow *ui;

    QSettings *m_configureFile;
    //file path
    QString m_testPicFilePath;
    cv::Mat m_testImage;

    cv::Mat m_srcImage1;
    cv::Mat m_srcImage2;

    //algorithm parameter
//    int m_adaptiveBlokeSize;
//    double m_adaptiveThred;

    std::vector<cv::Rect> m_cropedRectVec;
    //query data
    AlgorithmMidea midea1;

};

#endif // MAINWINDOW_H
