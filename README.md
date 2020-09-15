# dhe
c++ implement for dynamic histogram equalization.

## Installtion
All you need is opencv.

## Parameter
For my task, the best alpha is round 0.06. You can use grid search to find the most approprite one for your own task.

## Usage
```cpp
#include "dhe.hpp"

cv::Mat result;
cv::Mat img = cv::imread("your image file path");
img.convertTo(img,cv::CV_8U);
dhe(img,result,0.06);
```
