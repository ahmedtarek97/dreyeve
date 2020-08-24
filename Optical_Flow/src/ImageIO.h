// Author: Ce Liu (c) Dec, 2009; celiu@mit.edu


#ifndef _ImageIO_h
#define _ImageIO_h

// No need of OpenCV; python wrapper handles it
// #include "cv.h"
// #include "highgui.h"
// #include "opencv2/core/core.hpp"
// #include "opencv2/highgui/highgui.hpp"

class ImageIO
{
public:
    enum ImageType{standard, derivative, normalized};
    ImageIO(void);
    ~ImageIO(void);
public:
    template <class T>
    static bool loadImage(const char* filename,T*& pImagePlane,int& width,int& height, int& nchannels);
    template <class T>
    static bool saveImage(const char* filename,const T* pImagePlane,int width,int height, int nchannels,ImageType imtype = standard);

};


#endif
