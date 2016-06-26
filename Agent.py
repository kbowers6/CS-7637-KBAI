# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
from PIL import Image
import PIL.ImageOps
import PIL.ImageMath
from PIL import ImageFilter
from PIL import ImageChops
import numpy as np

class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        self.MCimageNames = ['1', '2', '3', '4', '5', '6']
        self.MCimageNames3x3 = ['1', '2', '3', '4', '5', '6', '7', '8']
        #self.problemImageNames = ['A', 'B', 'C']
        self.confidenceThreshold = .70

    def MeasureImageSimilarityNumpy(self, imgA, imgB, showImage):
        m1 = np.matrix(imgA).astype(float)
        m2 = np.matrix(imgB).astype(float)

        sumDiffPixels = (np.abs(np.subtract(m1,m2)) > 127).sum() # sums all the whitish pixels
        numPixels = imgA.size[0] * imgA.size[1]

        similarityPercentage = 1.0 - (sumDiffPixels / (numPixels * 1.0))

        if showImage:
            print  'Similarity = ', similarityPercentage, '%'

        return similarityPercentage

    # Opens Image with blurred inverted b/w. Objects in image are in white, background in black.
    def OpenImage(self, problem, imageName, blurred= True):
        image = Image.open(problem.figures[imageName].visualFilename).convert('L')
        image = PIL.ImageOps.invert(image) # invert so background is black, objects are white

        # Blur image a little bit
        if blurred:
            gaussFilter = ImageFilter.GaussianBlur(radius=1)
            image = image.filter(gaussFilter)

        return image

    def GetObjectPixelCount(self, image, whiteness=250):
        npImage = np.matrix(image).astype(float)
        pixelCount = (npImage > whiteness).sum()
        return pixelCount

    # Loops the MC options and picks the one with highest confidence compared to the generatedImage
    # Returns tuple of confidence and the best Answer
    def FindBestAnswer(self, problem, generatedImage, showImage, is3x3=False):
        isFirst = True
        bestConfidence = 0.0
        bestAnswer = 0

        counter = 1
        # generatedImage.show()
        # input("Press Enter to continue...")
        # Loop all the MC answers and pick the most confident
        if is3x3:
            imageSet = self.MCimageNames3x3
        else:
            imageSet = self.MCimageNames
        for imageName in imageSet:
            image = self.OpenImage(problem, imageName)

            similarity = self.MeasureImageSimilarityNumpy(generatedImage, image, showImage)

            print 'Answer', counter, 'has sim of ', similarity

            if isFirst:
                isFirst = False
                bestAnswer = counter
                bestConfidence = similarity
            else:
                if similarity > bestConfidence:
                    bestAnswer = counter
                    bestConfidence = similarity
            counter += 1
        return (bestConfidence, bestAnswer)

    # Loops the MC options and picks the one with lowest difference compared to the sumOfPixels
    # Returns tuple of confidence and the best Answer
    def FindBestSumAnswer(self, problem, sumOfPixels, showImage, threshold= 200, is3x3=False, whiteness=250):
        isFirst = True
        bestConfidence = 0.0
        bestAnswer = -1

        counter = 1
        # Loop all the MC answers and pick the most confident
        if is3x3:
            imageSet = self.MCimageNames3x3
        else:
            imageSet = self.MCimageNames
        for imageName in imageSet:
            image = self.OpenImage(problem, imageName, blurred=False)

            npImage = np.matrix(image).astype(float)
            sumPixelsImage = (npImage > whiteness).sum() # sums all the whitish pixels

            diffCount = abs(sumOfPixels - sumPixelsImage)

            print 'Answer', counter, 'has diff of', diffCount, 'pixels (', sumPixelsImage,')'

            if diffCount < threshold:
                if isFirst:
                    isFirst = False
                    bestAnswer = counter
                    bestConfidence = diffCount
                else:
                    if diffCount < bestConfidence:
                        bestAnswer = counter
                        bestConfidence = diffCount
            counter += 1
        return (bestConfidence, bestAnswer)

    # Tests pixel count diff add as the transform: (B - A) + B
    def PixelCountDiffAddTest_3x3(self, problem, problemImages, showImage):
        whiteness = 200
        print 'diff add...'
        A = problemImages[0]
        B = problemImages[1]
        C = problemImages[2]

        D = problemImages[3]
        E = problemImages[4]
        F = problemImages[5]

        G = problemImages[6]
        H = problemImages[7]

        # Get pixel counts for all images
        sumA = self.GetObjectPixelCount(A, whiteness=whiteness)
        sumB = self.GetObjectPixelCount(B, whiteness=whiteness)
        sumC = self.GetObjectPixelCount(C, whiteness=whiteness)
        sumD = self.GetObjectPixelCount(D, whiteness=whiteness)
        sumE = self.GetObjectPixelCount(E, whiteness=whiteness)
        sumF = self.GetObjectPixelCount(F, whiteness=whiteness)
        sumG = self.GetObjectPixelCount(G, whiteness=whiteness)
        sumH = self.GetObjectPixelCount(H, whiteness=whiteness)

        # Test horizontal relationship
        threshold = 300
        confidenceHorizontal = 0.0
        if (sumC - threshold < (sumB - sumA) + sumB < sumC + threshold) and (sumF - threshold < (sumE - sumD) + sumE < sumF + threshold):
            confidenceHorizontal = 1.0

        # Test vertical relationship
        confidenceVertical = 0.0
        if (sumG - threshold < (sumD - sumA) + sumD < sumG + threshold) and (sumH - threshold < (sumE - sumB) + sumE < sumH + threshold):
            confidenceVertical = 1.0

        if True:
            print 'conf LR = ', confidenceHorizontal
            print 'conf TB = ', confidenceVertical

        print 'sumG', sumG, 'sumH', sumH
        # transformedImage is the resulting image from transforming either C (for C->D) or B (for B->D)
        if confidenceHorizontal >= confidenceVertical:
            targetSumPixels = (sumH - sumG) + sumH
        else:
            targetSumPixels = (sumF - sumC) + sumF

        return max(confidenceHorizontal, confidenceVertical), self.FindBestSumAnswer(problem, targetSumPixels, showImage, threshold=threshold, is3x3=True, whiteness=whiteness)

    # Tests entire flip as the transform, vertical and horizontal
    def EntireFlip_3x3(self, problem, problemImages, showImage):
        print 'entire flip LR...'
        A = problemImages[0]
        B = problemImages[1]
        C = problemImages[2]

        D = problemImages[3]
        E = problemImages[4]
        F = problemImages[5]

        G = problemImages[6]
        H = problemImages[7]

        # Test horizontal relationship
        mirrorA = PIL.ImageOps.mirror(A)
        mirrorD = PIL.ImageOps.mirror(D)
        confidenceAC = self.MeasureImageSimilarityNumpy(mirrorA, C, showImage)
        confidenceDF = self.MeasureImageSimilarityNumpy(mirrorD, F, showImage)

        # Test vertical relationship
        flipA = PIL.ImageOps.flip(A)
        flipB = PIL.ImageOps.flip(B)
        confidenceAG = self.MeasureImageSimilarityNumpy(flipA, G, showImage)
        confidenceBH = self.MeasureImageSimilarityNumpy(flipB, H, showImage)

        print 'conf LR = ', confidenceAC, confidenceDF
        print 'conf TB = ', confidenceAG, confidenceBH

        # Get the more confident relationship direction
        if min(confidenceAC, confidenceDF) > min(confidenceAG, confidenceBH): # horizontal is better
            targetImage = PIL.ImageOps.mirror(G)
            relationshipConfidence = min(confidenceAC, confidenceDF)
        else: # vertical is better
            targetImage = PIL.ImageOps.flip(C)
            relationshipConfidence = min(confidenceAG, confidenceBH)

        return relationshipConfidence, self.FindBestAnswer(problem, targetImage, showImage, is3x3=True)

    # Tests no change as the transform
    def WallDoublingPixelsTest_3x3(self, problem, problemImages, showImage):
        print 'wall doubling...'
        A = problemImages[0]
        B = problemImages[1]
        C = problemImages[2]

        D = problemImages[3]
        F = problemImages[5]

        G = problemImages[6]
        H = problemImages[7]

        # Get pixel counts for all images
        sumA = self.GetObjectPixelCount(A)
        sumB = self.GetObjectPixelCount(B)
        sumC = self.GetObjectPixelCount(C)
        sumD = self.GetObjectPixelCount(D)
        sumF = self.GetObjectPixelCount(F)
        sumG = self.GetObjectPixelCount(G)
        sumH = self.GetObjectPixelCount(H)

        # Test horizontal wall doubling
        threshold = 200
        confidenceHorizontal = 0.0
        if (sumC - threshold < sumA*2 < sumC + threshold) and (sumF - threshold < sumD*2 < sumF + threshold):
            confidenceHorizontal = 1.0

        # Test vertical wall doubling
        confidenceVertical = 0.0
        if (sumG - threshold < sumA * 2 < sumG + threshold) and (sumH - threshold < sumB * 2 < sumH + threshold):
            confidenceVertical = 1.0

        if True:
            print 'conf LR = ', confidenceHorizontal
            print 'conf TB = ', confidenceVertical

        # transformedImage is the resulting image from mirroring either C (for C->D) or B (for B->D)
        if confidenceHorizontal >= confidenceVertical:
            targetSumPixels = sumG * 2
        else:
            targetSumPixels = sumC * 2

        return max(confidenceHorizontal, confidenceVertical), self.FindBestSumAnswer(problem, targetSumPixels, showImage, threshold=threshold, is3x3=True)

    # Tests mirror as the transform (left to right mirror)
    def MirrorTest(self, problem, problemImages, showImage):
        print 'mirror...'
        A = problemImages[0]
        B = problemImages[1]
        C = problemImages[2]

        # Test A -> B mirror and A -> C mirror
        mirroredA = PIL.ImageOps.mirror(A)
        confidenceAB = self.MeasureImageSimilarityNumpy(mirroredA, B, showImage)
        confidenceAC = self.MeasureImageSimilarityNumpy(mirroredA, C, showImage)

        if True:
            print 'conf AB = ', confidenceAB
            print 'conf AC = ', confidenceAC

        # Not good enough confidence
        if max(confidenceAC, confidenceAB) < self.confidenceThreshold:
            return (max(confidenceAC, confidenceAB), (max(confidenceAC, confidenceAB), -1))

        # transformedImage is the resulting image from mirroring either C (for C->D) or B (for B->D)
        if confidenceAB >= confidenceAC:
            transformedImage = PIL.ImageOps.mirror(C)
        else:
            transformedImage = PIL.ImageOps.mirror(B)

        return (max(confidenceAC, confidenceAB), self.FindBestAnswer(problem, transformedImage, showImage))

    # Tests flip as the transform (up to down flip)
    def FlipTest(self, problem, problemImages, showImage):
        print 'flip...'
        A = problemImages[0]
        B = problemImages[1]
        C = problemImages[2]

        # Test A -> B flip and A -> C flip
        flipA = PIL.ImageOps.flip(A)
        confidenceAB = self.MeasureImageSimilarityNumpy(flipA, B, showImage)
        confidenceAC = self.MeasureImageSimilarityNumpy(flipA, C, showImage)

        if True:
            print 'conf AB = ', confidenceAB
            print 'conf AC = ', confidenceAC

        # Not good enough confidence
        if max(confidenceAC, confidenceAB) < self.confidenceThreshold:
            return (max(confidenceAC, confidenceAB), (max(confidenceAC, confidenceAB), -1))

        # transformedImage is the resulting image from flipping either C (for C->D) or B (for B->D)
        if confidenceAB >= confidenceAC:
            transformedImage = PIL.ImageOps.flip(C)
        else:
            transformedImage = PIL.ImageOps.flip(B)

        return (max(confidenceAC, confidenceAB), self.FindBestAnswer(problem, transformedImage, showImage))

    # Tests for new or removed objects as the transform
    def BooleanTest(self, problem, problemImages, showImage):
        print 'boolean...'
        A = problemImages[0]
        B = problemImages[1]
        C = problemImages[2]

        # Test A -> B add/remove and A -> C add/remove
        testImageA = np.matrix(A).astype(float)
        testImageB = np.matrix(B).astype(float)
        testImageC = np.matrix(C).astype(float)

        testImageBsubA = testImageB - testImageA
        testImageCsubA = testImageC - testImageA

        # Apply transform to the other row/column
        AtoBtransformResult = testImageC + testImageBsubA
        AtoCtransformResult = testImageB + testImageCsubA

        # Convert To Image
        AtoBtransformResult = Image.fromarray(AtoBtransformResult)
        AtoCtransformResult = Image.fromarray(AtoCtransformResult)

        resultAtoB = self.FindBestAnswer(problem, AtoBtransformResult, showImage)
        resultAtoC = self.FindBestAnswer(problem, AtoCtransformResult, showImage)

        # Return the best Answer
        if resultAtoB[0] >= resultAtoC[0]:
            return (resultAtoB[0], resultAtoB)
        else:
            return (resultAtoC[0], resultAtoC)

    # Tests for new or removed objects as the transform
    # if good enough match, confidence is set to maximum
    def FillTest(self, problem, problemImages, showImage):
        fillthreshold = 500
        print 'fill...'
        A = problemImages[0]
        B = problemImages[1]
        C = problemImages[2]

        # Get the sum of the white pixels in each image
        npA = np.matrix(A).astype(float)
        npB = np.matrix(B).astype(float)
        npC = np.matrix(C).astype(float)
        sumPixelsA = (npA > 180).sum() # sums all the whitish pixels
        sumPixelsB = (npB > 180).sum() # sums all the whitish pixels
        sumPixelsC = (npC > 180).sum() # sums all the whitish pixels

        # To detect if an image fills an object, treat the previous sum of white pixels as a circumference of a circle
        # Then solve for the radius and find the area, i.e. how many pixels are needed to fill the object
        # If this area sum is close to the sum of the new image, the object(s) have been filled
        # circumference to area simplifies to C ** 2 / 4*pi
        lineThickness = 4
        sumAfilled = ((sumPixelsA/lineThickness)** 2) / (4 * np.pi)
        sumBfilled = ((sumPixelsB/lineThickness) ** 2) / (4 * np.pi)
        sumCfilled = ((sumPixelsC/lineThickness) ** 2) / (4 * np.pi)

        sumAunfilled = 2 * np.sqrt(np.pi * sumPixelsA)
        sumBunfilled = 2 * np.sqrt(np.pi * sumPixelsB)
        sumCunfilled = 2 * np.sqrt(np.pi * sumPixelsC)

        print 'sum=', sumPixelsA, sumPixelsB, sumPixelsC, 'fill=', sumAfilled, sumBfilled, sumCfilled, 'unfill=', sumAunfilled, sumBunfilled, sumCunfilled
        # Find the most confident match for filling or unfilling
        if sumPixelsB - fillthreshold <= sumAfilled <= sumPixelsB + fillthreshold:
            resultAtoBfilled = self.FindBestSumAnswer(problem, sumCfilled, showImage)
            return (1.0, (1.0, resultAtoBfilled[1]))
        if sumPixelsC - fillthreshold <= sumAfilled <= sumPixelsC + fillthreshold:
            resultAtoCfilled = self.FindBestSumAnswer(problem, sumBfilled, showImage)
            return (1.0, (1.0, resultAtoCfilled[1]))
        if sumPixelsB - fillthreshold <= sumAunfilled <= sumPixelsB + fillthreshold:
            resultAtoBunfilled = self.FindBestSumAnswer(problem, sumCunfilled, showImage)
            return (1.0, (1.0, resultAtoBunfilled[1]))
        if sumPixelsC - fillthreshold <= sumAunfilled <= sumPixelsC + fillthreshold:
            resultAtoCunfilled = self.FindBestSumAnswer(problem, sumBunfilled, showImage)
            return (1.0, (1.0, resultAtoCunfilled[1]))
        return (0.0, -1)


    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints 
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.

    # problem input is a RavensProblem object.
    # problem.figures is dictionary of the question. "A" -> "C" for question, "1" -> "6" for answer
    # 3x3 matrices are "A" -> "H" for question, "1" -> "8" for answer
    def Solve(self,problem):
        print '\nBegin ' + problem.name + ' --------------------------------------------'

        # Is this a 2x2 or 3x3 problem
        is3x3 = len(problem.figures) > 9 # 2x2s have 9 images: 3 for the 2x2 and 6 MCs

        if not is3x3:
            imgA = self.OpenImage(problem,'A')
            imgB = self.OpenImage(problem,'B')
            imgC = self.OpenImage(problem,'C')

            problemImagesBlurred = [imgA, imgB, imgC]

            imgARaw = self.OpenImage(problem,'A', False)
            imgBRaw = self.OpenImage(problem,'B', False)
            imgCRaw = self.OpenImage(problem,'C', False)

            problemImagesRaw = [imgARaw, imgBRaw, imgCRaw]

            ####### Run 2x2 Test Suite ###################################################################################

            # Result is tuple of confidence (float) and answer (int)
            resultMirror = self.MirrorTest(problem, problemImagesBlurred, False)
            resultFlip = self.FlipTest(problem, problemImagesBlurred, False)
            resultBoolean = self.BooleanTest(problem, problemImagesBlurred, False)
            resultFill = self.FillTest(problem, problemImagesRaw, False)

            # Set Add/remove Object as best choice
            bestRelationshipConfidence = resultBoolean[0]
            bestAnswerConfidence = resultBoolean[1][0]
            bestAnswer = resultBoolean[1][1]
            chosenTransform = 'boolean'

            # Set Mirror as best choice
            if resultMirror[0] >= bestRelationshipConfidence and resultMirror[1][0] >= bestAnswerConfidence:
                bestRelationshipConfidence = resultMirror[0]
                bestAnswerConfidence = resultMirror[1][0]
                bestAnswer = resultMirror[1][1]
                chosenTransform = 'mirroring'

            # Set Flip as best choice
            if resultFlip[0] >= bestRelationshipConfidence and resultFlip[1][0] >= bestAnswerConfidence:
                bestRelationshipConfidence = resultFlip[0]
                bestAnswerConfidence = resultFlip[1][0]
                bestAnswer = resultFlip[1][1]
                chosenTransform = 'flipping'

            # Set Fill as best choice
            if resultFill[0] >= bestRelationshipConfidence and resultFill[1][0] >= bestAnswerConfidence:
                bestRelationshipConfidence = resultFill[0]
                bestAnswerConfidence = resultFill[1][0]
                bestAnswer = resultFill[1][1]
                chosenTransform = 'filling'

            print 'Answer is', bestAnswer, 'with confidence', bestAnswerConfidence, 'solved by', chosenTransform
            return bestAnswer

        else: # 3x3 Test Suite
            imgA = self.OpenImage(problem, 'A')
            imgB = self.OpenImage(problem, 'B')
            imgC = self.OpenImage(problem, 'C')
            imgD = self.OpenImage(problem, 'D')
            imgE = self.OpenImage(problem, 'E')
            imgF = self.OpenImage(problem, 'F')
            imgG = self.OpenImage(problem, 'G')
            imgH = self.OpenImage(problem, 'H')

            problemImagesBlurred = [imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH]

            imgARaw = self.OpenImage(problem, 'A', blurred=False)
            imgBRaw = self.OpenImage(problem, 'B', blurred=False)
            imgCRaw = self.OpenImage(problem, 'C', blurred=False)
            imgDRaw = self.OpenImage(problem, 'D', blurred=False)
            imgERaw = self.OpenImage(problem, 'E', blurred=False)
            imgFRaw = self.OpenImage(problem, 'F', blurred=False)
            imgGRaw = self.OpenImage(problem, 'G', blurred=False)
            imgHRaw = self.OpenImage(problem, 'H', blurred=False)

            problemImagesRaw = [imgARaw, imgBRaw, imgCRaw, imgDRaw, imgERaw, imgFRaw, imgGRaw, imgHRaw]

            ####### Run 3x3 Test Suite ###################################################################################

            # Result is tuple of confidence (float) and answer (int)
            resultWallDoubling = self.WallDoublingPixelsTest_3x3(problem, problemImagesRaw, False)
            resultPixelCountDiffAdd = self.PixelCountDiffAddTest_3x3(problem, problemImagesRaw, False)
            resultEntireFlip = self.EntireFlip_3x3(problem, problemImagesBlurred, False)

            # Set Skip as best choice
            bestRelationshipConfidence = .50
            bestAnswerConfidence = 0.0
            bestAnswer = -1
            chosenTransform = 'skip'

            # Set Wall Doubling as best choice
            if resultWallDoubling[0] >= bestRelationshipConfidence and resultWallDoubling[1][0] >= bestAnswerConfidence:
                bestRelationshipConfidence = resultWallDoubling[0]
                bestAnswerConfidence = resultWallDoubling[1][0]
                bestAnswer = resultWallDoubling[1][1]
                chosenTransform = 'doubling'

            # Set Diff Add as best choice
            if resultPixelCountDiffAdd[0] >= bestRelationshipConfidence and resultPixelCountDiffAdd[1][0] >= bestAnswerConfidence:
                bestRelationshipConfidence = resultPixelCountDiffAdd[0]
                bestAnswerConfidence = resultPixelCountDiffAdd[1][0]
                bestAnswer = resultPixelCountDiffAdd[1][1]
                chosenTransform = 'diff add'

            # Set Entire Flip as best choice
            if resultEntireFlip[0] >= bestRelationshipConfidence and resultEntireFlip[1][0] >= bestAnswerConfidence:
                bestRelationshipConfidence = resultEntireFlip[0]
                bestAnswerConfidence = resultEntireFlip[1][0]
                bestAnswer = resultEntireFlip[1][1]
                chosenTransform = 'entire flip'

            print resultPixelCountDiffAdd
            print 'Answer is', bestAnswer, 'with answer confidence', bestAnswerConfidence, 'and relationship confidence', bestRelationshipConfidence, 'solved by', chosenTransform
            return bestAnswer