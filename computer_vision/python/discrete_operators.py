from PIL import Image
from pylab import *

from numpy import *
from scipy import ndimage
from scipy.ndimage import filters

def plot_img(img,text):

	figure()
	gray()
	imshow(img)
	axis('equal')
	axis('off')
	title(text)
	
def apply_sobel(img):

	imx = zeros(img.shape)
	filters.sobel(img,1,imx)
	
	imy = zeros(img.shape)
	filters.sobel(img,0,imy)

	#the magnitude
	grad = sqrt(imx**2+imy**2)

	return imx,imy,grad

def apply_prewitt(img):
	imx = zeros(img.shape)
	filters.prewitt(img,1,imx)

	imy = zeros(img.shape)
	filters.prewitt(img,0,imy)

	#the magnitude
	grad = sqrt(imx**2+imy**2)

	return imx,imy,grad

def apply_robert(img):

	Dx  = array([[1,0],[0,-1]])
	Dy  = array([[0,1],[-1,0]])
	imx = ndimage.convolve(img,Dx)
	imy = ndimage.convolve(img,Dy)
	grad = sqrt(imx**2+imy**2)
	return imx,imy,grad
	

operator_name = input("Enter the name of the operator to apply (Sobel,Prewitt,Robert): ")

#read the image from the data and convert it to greyscale
img = array(Image.open('im/data/empire.jpg').convert('L'))

if operator_name == 'Sobel':

	plot_img(img,'Original')
	imx, imy, grad = apply_sobel(img)
	plot_img(imx,'Ix')
	plot_img(imy,'Iy')
	plot_img(grad,'Gradient')
	show()

elif operator_name == 'Prewitt':

	plot_img(img,'Original')
	imx, imy, grad = apply_prewitt(img)
	plot_img(imx,'Ix')
	plot_img(imy,'Iy')
	plot_img(grad,'Gradient')
	show()
elif operator_name == 'Robert':

	plot_img(img,'Original')
	imx, imy, grad = apply_robert(img)
	plot_img(imx,'Ix')
	plot_img(imy,'Iy')
	plot_img(grad,'Gradient')
	show()
	

	
