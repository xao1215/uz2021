import numpy as np
import cv2
from matplotlib import pyplot as plt

def a():
    i = cv2.imread("../images/umbrellas.jpg")
    # i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)

    # height, width, channels = i.shape
    # print(i.shape)

    plt.imshow(i)
    plt.show()

def b():
    i = cv2.imread("../images/umbrellas.jpg")
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)

    newi = np.copy(i).astype(float)
    omg = ((newi[:,:,0] + newi[:,:,1] + newi[:,:,2]) / 3)

    for x in range(0, len(omg)):
        for y in range(0, len(omg[x])):
            num = omg[x,y]
            newi[x,y] = [num,num,num]

    newi = newi.astype(int)
    plt.imshow(newi)
    plt.show()

def c():
    i = cv2.imread("../images/umbrellas.jpg")
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)

    plt.subplot(1,2,1)
    plt.imshow(i)

    cutout = i[130:260,240:450,1]
    plt.subplot(1,2,2)
    plt.imshow(cutout, cmap="gray")

    plt.show()

def d():
    #invert part of iamge
    i = cv2.imread("../images/umbrellas.jpg")
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)

    plt.subplot(1,2,1)
    plt.imshow(i)

    invert = np.copy(i)
    invert[130:260,240:450,0] = 255-invert[130:260,240:450,0]
    invert[130:260,240:450,1] = 255-invert[130:260,240:450,1]
    invert[130:260,240:450,2] = 255-invert[130:260,240:450,2]
    
    plt.subplot(1,2,2)
    plt.imshow(invert)

    plt.show()

def e():
    i = cv2.imread("../images/umbrellas.jpg")
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    i = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)

    update = np.copy(i)
    update = update.astype(float)
    update[:,:] = update[:,:]/255*63
    update = update.astype(np.uint8)

    plt.subplot(1,2,1)
    plt.imshow(i,cmap="gray",vmax=255)
    plt.subplot(1,2,2)
    plt.imshow(update,cmap="gray",vmax=255)

    plt.show()

######################################################################

def aa():
    i = cv2.imread("../images/bird.jpg")
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    i = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)

    mask = np.copy(i)
    threshold = 74
    mask[mask < threshold] = 0
    mask[mask >= threshold] = 255

    maska = np.copy(i)
    maska = np.where( maska<threshold, 0, 1 )

    plt.subplot(2,3,1)
    plt.imshow(i,cmap="gray")
    plt.subplot(2,3,2)
    plt.imshow(mask,cmap="gray")
    plt.subplot(2,3,3)
    plt.imshow(maska,cmap="gray")

    plt.show()

def myhist(image,n):

    interval = 255 / n
    arr = image.reshape(-1)
    histogram = np.zeros(n)

    for x in range(len(arr)):
        histogram[ int(arr[x] / interval) - 1 ] += 1

    # histogram[:] = histogram[:] / len(arr)
    # print(histogram)

    # histogram = np.true_divide(histogram,len(arr))
    # a = 0
    # for x in range(len(histogram)):
    #     a += histogram[x]
    # print(a)

    return np.true_divide(histogram,len(arr))

def myhistnew(image,n):
    arr = image.reshape(-1)
    interval = (np.amax(arr) - np.amin(arr)) / n
    arr[:] = arr[:] - np.amin(arr)
    histogram = np.zeros(n)

    for x in range(len(arr)):
        histogram[ int(arr[x] / interval) - 1 ] += 1
    return np.true_divide(histogram,len(arr))

def bb():
    i = cv2.imread("../images/bird.jpg")
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    i = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)

    # print(myhist(i,25))

    plt.subplot(1,3,1)
    plt.imshow(i,cmap="gray")

    plt.subplot(1,3,2)
    # plt.hist(i.reshape(-1),density=True, bins=200,rwidth=0.7)
    plt.bar(np.arange(256), myhist(i,256))

    plt.subplot(1,3,3)
    plt.bar(np.arange(25), myhist(i,25))
    # plt.hist(i.reshape(-1),density=True, bins=25,rwidth=0.7)
    plt.show()

def cc():
    i = cv2.imread("../images/candy.jpg")
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    i = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)

    # myhistnew(i,10)

    plt.subplot(2,3,1)
    plt.imshow(i,cmap="gray")

    plt.subplot(2,3,2)
    # plt.hist(i.reshape(-1),density=True, bins=200,rwidth=0.7)
    plt.bar(np.arange(50), myhist(i,50))

    plt.subplot(2,3,3)
    plt.bar(np.arange(15), myhist(i,15))
    # plt.hist(i.reshape(-1),density=True, bins=25,rwidth=0.7)

    plt.subplot(2,3,4)
    plt.bar(np.arange(50), myhistnew(i,50))

    plt.subplot(2,3,5)
    plt.bar(np.arange(15), myhistnew(i,15))

    plt.show()

def dd():
    i = cv2.imread("../images/2.jpg")
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    i = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
    a = cv2.imread("../images/1.jpg")
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    b = cv2.imread("../images/4.jpg")
    b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)

    bins = 40

    plt.subplot(2,3,1)
    plt.imshow(i,cmap="gray")
    plt.subplot(2,3,4)
    plt.bar(np.arange(bins), myhist(i,bins))

    plt.subplot(2,3,2)
    plt.imshow(a,cmap="gray")
    plt.subplot(2,3,5)
    plt.bar(np.arange(bins), myhist(a,bins))

    plt.subplot(2,3,3)
    plt.imshow(b,cmap="gray")
    plt.subplot(2,3,6)
    plt.bar(np.arange(bins), myhist(b,bins))

    plt.show()

# OTSU
def mask(i):
    hist = myhist(i,256)

    total = np.sum(hist)
    besti = 0
    best = 0
    for x in range(1,len(hist)):
        none = np.sum(hist[:x]) / total
        ntwo = np.sum(hist[x:]) / total

        sumone = np.sum(hist[:x])
        sumtwo = np.sum(hist[x:])
        sone = 0
        stwo = 0

        if sumone != 0:
            sone = np.sum(np.multiply(np.arange(x),hist[:x])) / sumone
        if sumtwo != 0:
            stwo = np.sum(np.multiply(np.arange(x,len(hist)),hist[x:])) / sumtwo

        # stwo = np.sum(np.multiply(np.arange(x,len(hist)),hist[x:])) / np.sum(hist[x:])
        temp = np.square(sone-stwo) * none * ntwo
        if temp >= best:
            besti = x
            best = temp

    mask = np.copy(i)
    mask[mask < besti] = 0
    mask[mask >= besti] = 1

    return mask

def ee():
    i = cv2.imread("../images/bird.jpg")
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    i = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)

    hist = myhist(i,256)

    total = np.sum(hist)
    besti = 0
    best = 0
    for x in range(1,len(hist)):
        none = np.sum(hist[:x]) / total
        ntwo = np.sum(hist[x:]) / total

        sumone = np.sum(hist[:x])
        sumtwo = np.sum(hist[x:])
        sone = 0
        stwo = 0

        if sumone != 0:
            sone = np.sum(np.multiply(np.arange(x),hist[:x])) / sumone
        if sumtwo != 0:
            stwo = np.sum(np.multiply(np.arange(x,len(hist)),hist[x:])) / sumtwo

        # stwo = np.sum(np.multiply(np.arange(x,len(hist)),hist[x:])) / np.sum(hist[x:])
        temp = np.square(sone-stwo) * none * ntwo
        if temp >= best:
            besti = x
            best = temp

    mask = np.copy(i)
    mask[mask < besti] = 0
    mask[mask >= besti] = 1

    plt.subplot(1,2,1)
    plt.imshow(i,cmap="gray")
    plt.subplot(1,2,2)
    plt.imshow(mask,cmap="gray")
    plt.show()

######################################################################

def aaa(x):
    i = cv2.imread("../images/mask.png")
    # i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    # i = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)

    opening = np.copy(i)
    closing = i

    n = x
    SE = np.ones((n,n), np.uint8)
    #open
    opening = cv2.erode(opening, SE)
    opening = cv2.dilate(opening, SE)

    #close
    closing = cv2.dilate(closing, SE)
    closing = cv2.erode(closing, SE)

    plt.subplot(1,3,1)
    plt.imshow(i,cmap="gray")
    plt.subplot(1,3,2)
    plt.imshow(opening,cmap="gray")
    plt.subplot(1,3,3)
    plt.imshow(closing,cmap="gray")
    plt.show()

def bbb():
    i = cv2.imread("../images/bird.jpg")
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    i = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)

    a = mask(i)
    b = mask(i)

    n = 20
    SE = np.ones((n,n), np.uint8)
    SEE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))

    #close
    a = cv2.dilate(a, SE)
    a = cv2.erode(a, SE)

    b = cv2.dilate(b, SEE)
    b = cv2.erode(b, SEE)

    plt.subplot(1,3,1)
    plt.imshow(mask(i),cmap="gray")
    plt.subplot(1,3,2)
    plt.imshow(a,cmap="gray")
    plt.subplot(1,3,3)
    plt.imshow(b,cmap="gray")
    plt.show()

def immask(i,m):
    r = i[:,:,0]
    g = i[:,:,1]
    b = i[:,:,2]
    for x in range(len(m)):
        for y in range(len(m[x])):
            if m[x,y] == 0:
                r[x,y] = 0
                g[x,y] = 0
                b[x,y] = 0
    return np.dstack((r,g,b))

def ccc():
    i = cv2.imread("../images/bird.jpg")
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    g = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)

    maska = mask(g)

    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
    maska = cv2.dilate(maska, SE)
    maska = cv2.erode(maska, SE)

    new = immask(i,maska)

    plt.imshow(new)
    plt.show()

def ddd():
    i = cv2.imread("../images/eagle.jpg")
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)

    # i[:,:,0] = 255-i[:,:,0]
    # i[:,:,1] = 255-i[:,:,1]
    # i[:,:,2] = 255-i[:,:,2]

    g = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)

    maska = mask(g)
    # i[:,:,0] = 255-i[:,:,0]
    # i[:,:,1] = 255-i[:,:,1]
    # i[:,:,2] = 255-i[:,:,2]
    i = immask(i,maska)

    plt.subplot(1,2,1)
    # plt.bar(np.arange(256), myhist(g,256))
    plt.imshow(i)
    plt.show()

def eee():
    i = cv2.imread("../images/coins.jpg")
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    g = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)

    maska = mask(g)
    maska[:,:] = 1 - maska[:,:]

    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    new = cv2.dilate(maska, SE)
    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12))
    new = cv2.erode(new, SE)
    _,_,c,_ = cv2.connectedComponentsWithStats(new)

    im = np.copy(i)
    im[:,:,(0,1,2)] = (255,255,255)

    for x in range(1,len(c)):
        if c[x,4] < 700:
            im[ c[x,1]:c[x,1] + c[x,3], c[x,0]:c[x,0] + c[x,2] ,(0,1,2) ] = i[ c[x,1]:c[x,1] + c[x,3], c[x,0]:c[x,0] + c[x,2] ,(0,1,2) ] 

    plt.subplot(1,2,1)
    plt.imshow(im,cmap="gray")
    plt.subplot(1,2,2)
    plt.imshow(new,cmap="gray")

    plt.show()


aaa(5)