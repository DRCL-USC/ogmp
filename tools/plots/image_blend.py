# a opencv snippet to merge three images
import cv2 as cv

image0 = cv.imread("/home/lkrajan/Documents/paper_docs/rss24/figures/fig1/og0.png")
image1 = cv.imread("/home/lkrajan/Documents/paper_docs/rss24/figures/fig1/og1.png")
image2 = cv.imread("/home/lkrajan/Documents/paper_docs/rss24/figures/fig1/og2.png")
image3 = cv.imread("/home/lkrajan/Documents/paper_docs/rss24/figures/fig1/og3.png")


# add all images with equal wieghts 
image0123 = cv.addWeighted(image0, 0.5, image1, 0.5, 0)
# image0123 = cv.addWeighted(image0123, 0.5, image2, 0.5, 0)
# image0123 = cv.addWeighted(image0123, 0.5, image3, 0.5, 0)




# # add image 1 and 2 , alpha = 0.5
# image01 = cv.addWeighted(image0, 0.5, image1, 0.5, 0)
# # add image 12 and 3, alpha = 0.5
# image012 = cv.addWeighted(image01, 0.6, image2, 0.4, 0)
# # add image 012 and 4, alpha = 0.5
# image0123 = cv.addWeighted(image012, 0.5, image3, 0.5, 0)


# cv.imshow("image1", image1)
# cv.imshow("image2", image2)
# cv.imshow("image3", image3)
cv.imshow("result", image0123)
# save result
cv.imwrite("/home/lkrajan/Documents/paper_docs/rss24/figures/fig1/og0123.png",image0123)
cv.waitKey(0)
