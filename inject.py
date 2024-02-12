from PIL import Image
import imageio
import sys
from scipy import fftpack
import numpy
import matplotlib.pyplot as plt

def getAnswers(answerPath):
    with open(str(answerPath), 'r') as file:
        answerContent = file.read()
        answerAppend = answerContent.split()
    
    finalAnswerList = []

    for i in range(0, len(answerAppend), 2):
        finalAnswerList.append([answerAppend[i], answerAppend[i + 1]])

    # print(finalAnswerList)

    return finalAnswerList
    # print(answerContent)
    # for answer in answerContent:
    #     print(answer)
    #     answerAppend = answer.split()
    #     print(answerAppend)
    #     print(answerAppend[0], answerAppend[1])
    # print(answerContent)

def getFrequencies(imagePath, injectAnswers):
    # print("here")
    # bbox = (250, 660, 280, 690)

    # bbox = (310, 660, 340, 690)

    # bbox = (500, 860, 530, 890)
    imageToInject = imageio.imread(str(imagePath), as_gray=True)

    fft2ToInject = fftpack.fftshift(fftpack.fft2(imageToInject))

    # print(fft2ToInject)
    # dic = {}
    # for i in range(0, len(fft2ToInject)):
    #     for j in range(0, len(fft2ToInject[0])):
    #         if numpy.round(abs(fft2ToInject[i][j]), 0) in dic:
    #             dic[numpy.round(abs(fft2ToInject[i][j]), 0)] = dic[numpy.round(abs(fft2ToInject[i][j]), 0)] + 1
    #         else:
    #             dic[numpy.round(abs(fft2ToInject[i][j]), 0)] = 1

    # print(dic)

    noiseBoxPixels = {
        "A": [0, 0],
        "B": [60, 60],
        "C": [120, 120],
        "D": [180, 180],
        "E": [250, 250]
    }

    # bbox = (250, 650, 530, 700)

    # 2nd Column
    # bbox = (700, 650, 980, 700)

    # 3rd column
    # bbox = (1150, 650, 1430, 700)
    
    answerStartCoordinates = [[250, 650, 530, 700], 
                              [700, 650, 980, 700],
                              [1150, 650, 1430, 700]]
    
    i = 0
    for startCoordinates in answerStartCoordinates:
        i = i + 1
        current = startCoordinates
        print("Second place")
        while i <= 85:
            # currentNumpy = numpy.array(current)
            # print(i)
            correctAnswer = ""
            # for answer in injectAnswers:
            for answer in injectAnswers:
                if answer[0] == str(i):
                    # print(answer[1])
                    correctAnswer = answer[1]
                    pass

            
            # print(i)
            for ans in range(0, len(correctAnswer)):
                print(i)

                startCoordinatesCrop = (current[0] + noiseBoxPixels[correctAnswer[ans]][0], current[1])
                # endCoordinateCrop = (current[2] + noiseBoxPixels[correctAnswer[ans]][1], current[3])

                print(startCoordinatesCrop)
                # print(endCoordinateCrop)

                for x in range(startCoordinatesCrop[0], startCoordinatesCrop[0] + 30):
                    for y in range(startCoordinatesCrop[1], startCoordinatesCrop[1] + 30):
                        # if  y >= 1700:
                        #     pass
                        # else:
                            # print(round(numpy.real(fft2ToInject[x][y])))
                            # fft2ToInject[x][y]  = round(numpy.real(fft2ToInject[x][y])) + 1.24011998
                            # print(numpy.real(fft2ToInject[x][y]))
                            # imageToInject[x][y] -= 10
                        imageToInject[y][x] -= 10

            
            if i%29 == 0 and startCoordinates !=  [1150, 650, 1430, 700]:
                break
            elif i%29 == 0 and startCoordinates ==  [1150, 650, 1430, 700]:
                break
            else:
                pass
            i = i + 1
            # print(i)
            current = [current[0], current[1] + 50, current[2], current[3] + 50]

    count = 0
    for i in range(310, 340):
        for j in range(650, 680):
            if imageToInject[i][j] == 245:
                count = count + 1

    print(count)
    # plt.figure(figsize = (12, 6))
    # plt.imshow((numpy.log(abs(fft2ToInject))* 255 /numpy.amax(numpy.log(abs(fft2ToInject)))).astype(numpy.uint8), cmap='viridis', aspect='auto', origin='lower')
    # plt.show()

    # fft2 = fftpack.fftshift(fftpack.fft2(injectImage))

    # # imageio.imsave('fft.png', (numpy.log(abs(fft2))* 255 /numpy.amax(numpy.log(abs(fft2)))).astype(numpy.uint8))

    # print(fft2)
    # print("Intensity might")


    # plt.figure(figsize = (12, 6))
    # plt.imshow((numpy.log(abs(fft2))* 255 /numpy.amax(numpy.log(abs(fft2)))).astype(numpy.uint8), cmap='viridis', aspect='auto', origin='lower')
    # plt.show()
    # for i in range(0, len(fft2ToInject)):
    #     for j in range(0, len(fft2ToInject[0])):
    #         # print("*")
    #         answ = numpy.real(fft2ToInject[i][j])
    #         if answ > 0:
    #             a = answ - int(answ)
    #         else:
    #             a = -1*answ - (-1*int(answ))
    #         if a == 0.24011998:
    #             print(answ) 
    #         # print(numpy.real(fft2ToInject[i][j]))
 
    # ifft2 = abs(fftpack.ifft2(fftpack.ifftshift(fft2ToInject)))

    imageio.imsave('afterInjection3.png', imageToInject )

def getboxLocations():
    pass

if __name__ == '__main__':
    # injectFrequency = getFrequencies(sys.argv[1])
    injectForm = Image.open(sys.argv[1])

    print("Image is %s pixels wide." % injectForm.width)
    print("Image is %s pixels high." % injectForm.height)
    print("Image mode is %s." % injectForm.mode)

    print("Pixel value at (10,10) is %s" % str(injectForm.getpixel((10,10))))

    # For 5 boxes
    # 1st column
    # bbox = (250, 650, 530, 700)

    # 2nd Column
    # bbox = (700, 650, 980, 700)

    # 3rd column
    # bbox = (1150, 650, 1430, 700)

    # Pixels of one box shape 30 x 30
    # bbox = (250, 710, 280, 740)

    # bbox = (310, 710, 340, 740)

    bbox = (1150, 1550, 1430, 1600)

    pixels = injectForm.crop(bbox)
    pixel_values = list(pixels.getdata())
    pixels.save("cropped_image.jpg")

    injectAnswers = getAnswers(sys.argv[2])

    getFrequencies(sys.argv[1], injectAnswers)


    # print(injectForm)
    # print("hello")