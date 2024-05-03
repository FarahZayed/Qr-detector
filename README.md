"Qr-detector" 
this is only phase one: 
1- function ma7tangha: this section holds only function that will be called in function that raise flags 
2- function that raise the flags: this section holds function for each noise or disorientation that it detects and return true or false
3- flags to do preprocessing: flag for each noise being tested that get raised by functions above
4- Function responsible for preprocessing: this section holds the function that corrects the noise and return the preprocessed image 
5- Preprocessing: this section call the function that correct the raised flag and check for any type of noise to correct at the end it plots the image after being corrected by any type of noise 
