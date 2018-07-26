import cv2
import numpy as np
import glob
import pytesseract
import scipy.ndimage.morphology as morp
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'


def main():
    cont = 1

    for file in glob.glob("Sellos/*.png"):

        print("leyendo el ejemplo ", cont, " de 128")

        img = cv2.imread(file)

        img = cv2.resize(img, (320, 240))
        #img = cv2.resize(img, (640, 480))

        cv2.imshow('original', img)
        cv2.waitKey(0)

        edges = cv2.Canny(img, 100, 200)

        #cv2.imshow('bordes', edges)
        #cv2.waitKey(0)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        #cv2.imshow('HSV', hsv)
        #cv2.waitKey(0)


        # Rango de colores detectados:
        # Rojos:
        rojo_bajo = np.array([0, 44, 170], dtype=np.uint8)
        rojo_alto = np.array([10, 255, 255], dtype=np.uint8)
        # Verdes:
        apagado_bajo = np.array([0, 0, 139])
        apagado_alto = np.array([255, 42, 238])
        # Azules:
        azul_bajo = np.array([112, 11, 0], dtype=np.uint8)
        azul_alto = np.array([152, 140, 255], dtype=np.uint8)
        # Amarillos:
        amarillo_bajo = np.array([16, 30, 70], dtype=np.uint8)
        amarillo_alto = np.array([28, 215, 210], dtype=np.uint8)

        # Crear las mascaras
        #mascara_rojo1 = cv2.inRange(hsv, rojo_bajo1, rojo_alto1)
        mascara_rojo = cv2.inRange(hsv, rojo_bajo, rojo_alto)
        #cv2.imshow('mascara_rojo', mascara_rojo)
        #cv2.waitKey(0)
        mascara_apagado = cv2.inRange(hsv, apagado_bajo, apagado_alto)
        #cv2.imshow('mascara_apagado', mascara_apagado)
        #cv2.waitKey(0)
        mascara_amarillo = cv2.inRange(hsv, amarillo_bajo, amarillo_alto)
        #cv2.imshow('mascara_amarillo', mascara_amarillo)
        #cv2.waitKey(0)
        mascara_azul = cv2.inRange(hsv, azul_bajo, azul_alto)
        #cv2.imshow('mascara_azul', mascara_azul)
        #cv2.waitKey(0)

        final = cv2.hconcat((mascara_azul, mascara_amarillo, mascara_apagado, mascara_rojo))

        cv2.imshow('mascaras azul amarillo apagado y rojo', final)
        cv2.waitKey(0)

        # Juntar todas las mascaras
        mask = cv2.add(mascara_rojo, mascara_amarillo)
        mask = cv2.add(mask, mascara_apagado)
        mask = cv2.add(mask, mascara_azul)

        # Mostrar la mascara final y la imagen
        #cv2.imshow('mascara_color', mask)
        #cv2.waitKey(0)

        imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

        #cv2.imshow('YCbCr', imgYCC)
        #cv2.waitKey(0)

        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #cv2.imshow('YCbCr', img_g)
        #cv2.waitKey(0)

        #ret, img_th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh, img_bw = cv2.threshold(img_g, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        #cv2.imshow('binarizada', img_bw)
        #cv2.waitKey(0)

        #esq = skeletonize(img_bw)

        #kernel = np.ones((3, 3), np.uint8)

        #dilation = cv2.dilate(np.double(esq), kernel, iterations=1)
        #erosion = cv2.erode(dilation, kernel, iterations=3)

        #cv2.imshow('esqueletizada', erosion)
        #cv2.waitKey(0)

        mat = np.matrix([[0,0,0,0,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,0,0,0,0]])

        kernel = np.ones((3, 3), np.uint8)

        er = cv2.erode(img_bw, kernel, iterations=2)

        #cv2.imshow('er1', er)
        #cv2.waitKey(0)

        kernel = np.ones((5, 5), np.uint8)

        dil = cv2.dilate(er, mat, iterations=6)

        #cv2.imshow('dil1', dil)
        #cv2.waitKey(0)

        er = cv2.erode(dil, mat, iterations=6)

        #cv2.imshow('dil + er', er)
        #cv2.waitKey(0)

        cont = cont +1


def skeletonize(img):

    struct =  np.array([  [[[0, 0, 0], [0, 1, 0], [1, 1, 1]],
                           [[1, 1, 1], [0, 0, 0], [0, 0, 0]]],

                          [[[0, 0, 0], [1, 1, 0], [0, 1, 0]],
                           [[0, 1, 1], [0, 0, 1], [0, 0, 0]]],

                          [[[0, 0, 1], [0, 1, 1], [0, 0, 1]],
                           [[1, 0, 0], [1, 0, 0], [1, 0, 0]]],

                          [[[0, 0, 0], [0, 1, 1], [0, 1, 0]],
                           [[1, 1, 0], [1, 0, 0], [0, 0, 0]]],

                          [[[1, 1, 1], [0, 1, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [1, 1, 1]]],

                          [[[0, 1, 0], [0, 1, 1], [0, 0, 0]],
                           [[0, 0, 0], [1, 0, 0], [1, 1, 0]]],

                          [[[1, 0, 0], [1, 1, 0], [1, 0, 0]],
                           [[0, 0, 1], [0, 0, 1], [0, 0, 1]]],

                          [[[0, 1, 0], [1, 1, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 1], [0, 1, 1]]]])



    img = img.copy()
    last = ()
    while np.any(img != last):
        last = img
        for s in struct:
            img = np.logical_and(img, np.logical_not(morp.binary_hit_or_miss(img, *s)))
    return img




main()