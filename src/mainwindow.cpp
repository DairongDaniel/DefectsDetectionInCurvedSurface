#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QTime>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    initData();

    connect(ui->blockSizeLineEdit,SIGNAL(textEdited(const QString &)),this,SLOT(on_text_edited(const QString &)));
    connect(ui->thredLineEdit,SIGNAL(textEdited(const QString &)),this,SLOT(on_text_edited(const QString &)));
    connect(ui->displayWND,SIGNAL(s_cropedImageVec(std::vector<cv::Rect> )),this,SLOT(on_getInterestingCropImgaes(std::vector<cv::Rect> )));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::initData()
{
    //init setting file
    m_configureFile = new QSettings(
                QDir::currentPath()+"/initdata.ini",
                QSettings::NativeFormat
                );
    m_testPicFilePath=m_configureFile->value(
                "/FilePath/TestPictureFilePath", QDir::currentPath()
                ).toString();

    //    m_adaptiveBlokeSize=m_configureFile->value(
    //                "/AlgorithmParameter/adaptiveBlokeSize", 0
    //                ).toInt();
    //    m_adaptiveThred=m_configureFile->value(
    //                "/AlgorithmParameter/adaptiveThred", 0
    //                ).toDouble();
    updateAllWidgets();
}

void MainWindow::updateAllWidgets()
{
    ui->blockSizeLineEdit->setText(QString::number(midea1.blackDotSets.blockSize));
    ui->thredLineEdit->setText(QString::number(midea1.blackDotSets.binThred));
}

void MainWindow::on_pushButton_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(
                this,
                tr("Select the test picture"),
                m_testPicFilePath,
                tr("Images (*.png *.xpm *.jpg *.bmp *.jpeg)")
                );
    if (!fileName.isNull()) {
        QFileInfo fileInfo(fileName);
        m_testPicFilePath=fileInfo.absolutePath();
        qDebug()<<m_testPicFilePath;
        m_configureFile->setValue(
                    "/FilePath/TestPictureFilePath",
                    m_testPicFilePath);
        m_testImage=cv::imread(fileName.toStdString());
//        cv::Mat binImag;
//        AlgorithmMidea::binarizationForHSV(m_testImage,binImag,true);
        ui->displayWND->display(m_testImage);
    }
}

void MainWindow::on_text_edited(const QString &text)
{

    QObject *signalSender=sender();
    if (signalSender==ui->blockSizeLineEdit) {
        midea1.blackDotSets.blockSize=text.toInt()/2*2+1;
    } else if (signalSender==ui->thredLineEdit) {
        midea1.blackDotSets.binThred=text.toDouble();
    }
}

void MainWindow::on_processBTN_clicked()
{
    if (m_testImage.data) {
        QTime timer;
        double preTime=0;
        timer.start();
        cv::Mat tempImage;
        int eCode=midea1.detectAllDefects(
                    m_testImage,
                    tempImage
                    );
//        cv::Mat mask=cv::Mat::zeros(m_testImage.size(),CV_8UC1);
//        AlgorithmMidea::detectBlackDot(m_testImage,tempImage,midea1.blackDotSets,mask);
        ui->displayWND->display(tempImage);
        qDebug()<<"error code = "<<eCode;
        //        m_configureFile->setValue(
        //                    "/AlgorithmParameter/adaptiveBlokeSize",
        //                    m_adaptiveBlokeSize);
        //        m_configureFile->setValue(
        //                    "/AlgorithmParameter/adaptiveThred",
        //                    m_adaptiveThred);
        qDebug()<<"whole process cost time-----"<<timer.elapsed()-preTime<<"ms";
        preTime=timer.elapsed();
    }
    //    cv::Mat img1=cv::imread("/home/deniel/Documents/MD/Data/1.jpg");
    //    cv::Mat img2=cv::imread("/home/deniel/Documents/MD/Data/2.jpg");
    //    cvtColor(img1,img1,CV_BGR2GRAY);
    //    cvtColor(img2,img2,CV_BGR2GRAY);
    //    cv::Mat binImag;
    //    AlgorithmMidea::fuseImages(img1,img2,m_testImage);
    //    imwrite	("/home/deniel/Documents/MD/Data/fused.bmp", m_testImage);
    //    GaussianBlur( m_testImage, m_testImage,cv::Size(7,7), 1);
    //    inRange	( m_testImage, 80, 150, binImag );
    //    ui->displayWND->display(binImag);

}

void MainWindow::on_getInterestingCropImgaes(std::vector<cv::Rect> subImages)
{
    qDebug()<<"MainWindow::on_getInterestingCropImgaes(std::vector<Rect>)";
    m_cropedRectVec.clear();
    m_cropedRectVec=subImages;
}


void MainWindow::on_setQueryTemplate_clicked()
{
    cv::Mat tempImage;
    midea1.templateSets.roiRectVec=m_cropedRectVec;
    midea1.extractingTemplateFeature(
                m_testImage,
                tempImage);
    ui->displayWND->display(tempImage);
    imwrite	("/home/deniel/Documents/MD/Data/baseImage.bmp", midea1.templateFeature.baseImage);
}

void MainWindow::on_trainData_clicked()
{
    //    Mat matchedImage;
    cv::Mat tempImage;
    cv::Mat maskImage;
    AlgorithmMidea::detectLackofLogo(m_testImage,tempImage,midea1.templateFeature,midea1.logoSets,maskImage);
    ui->displayWND->display(tempImage);
}

void MainWindow::on_storeParameter_clicked()
{
    midea1.saveConfigureFile("/home/deniel/Projects/qtProjects/meidi/config.xml");
}

void MainWindow::on_loadParameter_clicked()
{
    midea1.loadConfigureFile("/home/deniel/Projects/qtProjects/meidi/config.xml");
    updateAllWidgets();
}

void MainWindow::on_pushButton_2_clicked()
{
    QStringList fileNames = QFileDialog::getOpenFileNames(
                this,
                tr("Select one or more test pictures"),
                m_testPicFilePath,
                tr("Images (*.png *.xpm *.jpg *.bmp *.jpeg)")
                );
    if (fileNames.size()>1) {

        QFileInfo fileInfo(fileNames.at(0));
        m_testPicFilePath=fileInfo.absolutePath();
        m_configureFile->setValue(
                    "/FilePath/TestPictureFilePath",
                    m_testPicFilePath);
        m_srcImage1=cv::imread(fileNames.at(0).toStdString());
        m_srcImage2=cv::imread(fileNames.at(1).toStdString());
        midea1.fuseImagesAndRomoveBG(m_srcImage1,m_srcImage2,m_testImage);
        ui->displayWND->display(m_testImage);
    }
}

void MainWindow::on_process2Picture_clicked()
{
    QStringList fileNames = QFileDialog::getOpenFileNames(
                this,
                tr("Select one or more test pictures"),
                m_testPicFilePath,
                tr("Images (*.png *.xpm *.jpg *.bmp *.jpeg)")
                );
    if (fileNames.size()>1) {

        QFileInfo fileInfo(fileNames.at(0));
        m_testPicFilePath=fileInfo.absolutePath();
        m_configureFile->setValue(
                    "/FilePath/TestPictureFilePath",
                    m_testPicFilePath);
        m_srcImage1=cv::imread(fileNames.at(0).toStdString());
        m_srcImage2=cv::imread(fileNames.at(1).toStdString());
        cv::Mat resultImage;
        midea1.fuseAndDetectAllDefects(m_srcImage1,m_srcImage2,resultImage);
//        AlgorithmMidea::inspectLineofPen(m_srcImage1,m_srcImage2,resultImage,true);
        ui->displayWND->display(resultImage);
    }
}
